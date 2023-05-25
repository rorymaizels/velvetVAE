"""Main module."""
from velvet.submodule import VectorField, NeighborhoodConstraint
from velvet.constants import REGISTRY_KEYS_VT
from velvet.utils import NoWarningZINB

from typing import Callable, Iterable, Optional
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from scvi._compat import Literal
from scvi._types import LatentDataType
from scvi.distributions import NegativeBinomial, Poisson
from scvi.module.base import BaseLatentModeModuleClass, LossOutput, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, LinearDecoderSCVI, one_hot

torch.backends.cudnn.benchmark = True


class VelVAE(BaseLatentModeModuleClass):
    """
    Variational Autoencoder (VAE) model with velocity modeling.

    Inputs:
    - transcriptome: The transcriptome data.
    - n_input: The number of input dimensions.
    - n_batch: The number of batch dimensions (default: 0).
    - n_labels: The number of label dimensions (default: 0).
    - n_hidden: The number of hidden units in the encoder and decoder layers (default: 128).
    - n_latent: The dimensionality of the latent space (default: 50).
    - n_layers: The number of layers in the encoder and decoder (default: 1).
    - n_continuous_cov: The number of continuous covariates (default: 0).
    - n_cats_per_cov: An iterable containing the number of categories for each covariate (default: None).
    - dropout_rate: The dropout rate for the encoder and decoder (default: 0.0).
    - dispersion: The dispersion parameter for gene-cell distribution (default: "gene").
    - log_variational: Whether to take the logarithm of the input data for variational inference (for stability, not transformation) (default: True).
    - gene_likelihood: The likelihood distribution for gene expression modeling ("zinb", "nb", "poisson", "normal") (default: "zinb").
    - latent_distribution: The distribution type for the latent space (default: "normal").
    - encode_covariates: Whether to encode covariates in the model (default: False).
    - deeply_inject_covariates: Whether to deeply inject covariates in the encoder (default: True).
    - use_batch_norm: The type of batch normalization to use ("encoder", "decoder", "none", "both") (default: "none").
    - use_layer_norm: The type of layer normalization to use ("encoder", "decoder", "none", "both") (default: "none").
    - use_linear_decoder: Whether to use a linear decoder (default: True).
    - use_size_factor_key: Whether to use the size factor key (default: False).
    - use_observed_lib_size: Whether to use the observed library size or inferred (default: True).
    - library_log_means: The mean of the log library sizes (default: None).
    - library_log_vars: The variance of the log library sizes (default: None).
    - var_activation: The activation function for the variance (default: None).
    - latent_data_type: The type of latent data. Not yet implemented ("dist", None) (default: None).
    - labelling_time: The time for labeling, in hours (default: 2.0).
    - neighborhood_loss: The type of neighborhood loss to use, MSE or cosine similarity ("mse", "cs") (default: "cs").
    - neighborhood_space: The space for neighborhood constraints, none being off. ("latent_space", "gene_space", "none") (default: "latent_space").
    - biophysical_model: The type of biophysical model, simple=steady state ratio, full=biophysical model ("simple", "full") (default: "full").
    - gamma_mode: The mode for gamma, whether to learn or not ("fixed", "learned") (default: "learned").
    - recon_lambda: The lambda coefficient for the reconstruction loss (default: 1.0).
    - velocity_lambda: The lambda coefficient for the velocity loss (default: 10.0).
    - neighborhood_lambda: The lambda coefficient for the neighborhood loss (default: 10.0).
    - kld_lambda: The lambda coefficient for the Kullback-Leibler divergence loss (default: 0.1).
    - neighborhood_kwargs: Additional keyword arguments for neighborhood constraints (default: {}).
    - vectorfield_kwargs: Additional keyword arguments for the vector field (default: {}).
    - gamma_kwargs: Additional keyword arguments for gamma (default: {}).
    - verbose: The verbosity level (default: 0).

    Outputs: None
    """

    def __init__(
        self,
        transcriptome,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 50,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.0,
        dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb", "poisson", "normal"] = "zinb",
        latent_distribution: str = "normal",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_linear_decoder: bool = True,
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        var_activation: Optional[Callable] = None,
        latent_data_type: Optional[LatentDataType] = None,
        labelling_time: float = 2.0,
        neighborhood_loss: Literal["mse", "cs"] = "cs",
        neighborhood_space: Literal["latent_space", "gene_space", "none"] = "latent_space",
        biophysical_model: Literal["simple", "full"] = "full",
        gamma_mode: Literal["fixed", "learned"] = "learned",
        recon_lambda: float = 1.0,
        velocity_lambda: float = 10.0,
        neighborhood_lambda: float = 10.0,
        kld_lambda: float = 0.1,
        neighborhood_kwargs={},
        vectorfield_kwargs={},
        gamma_kwargs={},
        verbose=0,
    ):
        super().__init__()

        # structuring learning
        self.verbose = verbose
        self.kld_lambda = kld_lambda
        self.rcn_lambda = recon_lambda
        self.nbr_lambda = neighborhood_lambda
        self.vel_lambda = velocity_lambda
        self.nbr_loss = neighborhood_loss

        # velocity modelling
        self.gamma_mode = gamma_mode
        self.bmode = biophysical_model
        self.space = neighborhood_space
        self.t = labelling_time

        # model hyperparams
        self.use_linear_decoder = use_linear_decoder
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self._latent_data_type = latent_data_type
        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size

        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, " "must provide library_log_means and library_log_vars."
                )

            self.register_buffer("library_log_means", torch.from_numpy(library_log_means).float())
            self.register_buffer("library_log_vars", torch.from_numpy(library_log_vars).float())

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None

        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent + n_continuous_cov
        if self.use_linear_decoder:
            self.decoder = LinearDecoderSCVI(
                n_input_decoder,
                n_input,
                n_cat_list=cat_list,
                use_batch_norm=use_batch_norm_decoder,
                use_layer_norm=use_layer_norm_decoder,
            )
        else:
            self.decoder = DecoderSCVI(
                n_input_decoder,
                n_input,
                n_cat_list=cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                inject_covariates=deeply_inject_covariates,
                use_batch_norm=use_batch_norm_decoder,
                use_layer_norm=use_layer_norm_decoder,
                scale_activation="softplus" if use_size_factor_key else "softmax",
            )

        self.vf = VectorField(latent_dim=n_latent, **vectorfield_kwargs)

        self.nc = NeighborhoodConstraint(X=transcriptome, **neighborhood_kwargs)
        self.median_library_size = torch.tensor(
            transcriptome.sum(1), device="cuda" if torch.cuda.is_available() else "cpu"
        ).median()

    def _get_inference_input(
        self,
        tensors,
    ):
        batch_index = tensors[REGISTRY_KEYS_VT.BATCH_KEY]

        cont_key = REGISTRY_KEYS_VT.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS_VT.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        if self.latent_data_type is None:
            x = tensors[REGISTRY_KEYS_VT.X_KEY]
            input_dict = dict(x=x, batch_index=batch_index, cont_covs=cont_covs, cat_covs=cat_covs)
        else:
            """
            if self.latent_data_type == "dist":
                qzm = tensors[REGISTRY_KEYS.LATENT_QZM_KEY]
                qzv = tensors[REGISTRY_KEYS.LATENT_QZV_KEY]
                input_dict = dict(qzm=qzm, qzv=qzv)
            else:
                raise ValueError(f"Unknown latent data type: {self.latent_data_type}")
            """
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        vz = inference_outputs["vz"]
        library = inference_outputs["library"]
        batch_index = tensors[REGISTRY_KEYS_VT.BATCH_KEY]
        y = tensors[REGISTRY_KEYS_VT.LABELS_KEY]

        cont_key = REGISTRY_KEYS_VT.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS_VT.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        size_factor_key = REGISTRY_KEYS_VT.SIZE_FACTOR_KEY
        size_factor = torch.log(tensors[size_factor_key]) if size_factor_key in tensors.keys() else None

        input_dict = dict(
            z=z,
            vz=vz,
            library=library,
            batch_index=batch_index,
            y=y,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
            size_factor=size_factor,
        )
        return input_dict

    def _compute_local_library_params(self, batch_index):
        """
        Computes local library parameters.
        Compute two tensors of shape (batch_index.shape[0], 1) where each
        element corresponds to the mean and variances, respectively, of the
        log library sizes in the batch the cell corresponds to.
        """
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(one_hot(batch_index, n_batch), self.library_log_means)
        local_library_log_vars = F.linear(one_hot(batch_index, n_batch), self.library_log_vars)
        return local_library_log_means, local_library_log_vars

    @auto_move_data
    def _regular_inference(self, x, batch_index, cont_covs=None, cat_covs=None, n_samples=1):
        """
        High level inference method.
        Runs the inference (encoder) model.
        """
        x_ = x

        # prepare for encoder
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        # run inference
        qz, z = self.z_encoder(encoder_input, batch_index, *categorical_input)
        vz = self.vf(z)

        ql = None
        if not self.use_observed_lib_size:
            ql, library_encoded = self.l_encoder(encoder_input, batch_index, *categorical_input)
            library = library_encoded

        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand((n_samples, library.size(0), library.size(1)))
            else:
                library = ql.sample((n_samples,))

        outputs = dict(z=z, vz=vz, qz=qz, ql=ql, library=library)
        return outputs

    def _cached_inference(self, qzm, qzv, n_samples=1):
        """
        if self.latent_data_type == "dist":
            dist = Normal(qzm, qzv.sqrt())
            # use dist.sample() rather than rsample because we aren't optimizing
            # the z in latent/cached mode
            untran_z = dist.sample() if n_samples == 1 else dist.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
        else:
            raise ValueError(f"Unknown latent data type: {self.latent_data_type}")
        outputs = dict(z=z, qz_m=qzm, qz_v=qzv, ql=None, library=None)
        return outputs"""
        pass  # TODO

    @auto_move_data
    def generative(
        self, z, vz, library, batch_index, cont_covs=None, cat_covs=None, size_factor=None, y=None, transform_batch=None
    ):
        """Runs the generative model."""
        ### formatting
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library

        z_future = z + vz

        if cont_covs is None:
            decoder_input = z
            decoder_input_future = z_future

        elif z.dim() != cont_covs.dim():
            decoder_input = torch.cat([z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1)
            decoder_input_future = torch.cat(
                [z_future, cont_covs.unsqueeze(0).expand(z_future.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)
            decoder_input_future = torch.cat([z_future, cont_covs], dim=-1)

        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            decoder_input,
            size_factor,
            batch_index,
            *categorical_input,
            y,
        )

        px_scale_future, px_r_future, px_rate_future, px_dropout_future = self.decoder(
            self.dispersion,
            decoder_input_future,
            size_factor,
            batch_index,
            *categorical_input,
            y,
        )

        vel = px_rate_future - px_rate

        if self.bmode == "full":
            t, g = self.t, self.loggamma.exp()
            new = ((1 - (-t * g).exp()) / g) * (vel + g * px_rate)
            new = torch.clip(new, 0, 100)
            new = torch.nan_to_num(new, nan=0, neginf=0, posinf=0)
        elif self.bmode == "simple":
            new = vel + self.loggamma.exp() * px_rate
            new = torch.clip(new, 0, 100)
            new = torch.nan_to_num(new, nan=0, neginf=0, posinf=0)

        if self.dispersion == "gene-label":
            px_r = F.linear(one_hot(y, self.n_labels), self.px_r)  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        if self.gene_likelihood == "zinb":
            px = NoWarningZINB(mu=px_rate, theta=px_r, zi_logits=px_dropout, scale=px_scale, validate_args=False)

        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale, validate_args=False)
        elif self.gene_likelihood == "poisson":
            px = Poisson(px_rate, scale=px_scale)
        elif self.gene_likelihood == "normal":
            px = Normal(loc=px_rate, scale=px_r)

        # Priors
        if self.use_observed_lib_size:
            pl = None
        else:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        return dict(
            px=px,
            vel=vel,
            new=new,
            pl=pl,
            pz=pz,
        )

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
    ):
        """Computes the loss function for the model."""
        x = tensors[REGISTRY_KEYS_VT.X_KEY]

        ### perform vae freezing
        if self.current_epoch > self.freeze_vae_after_epochs:
            self.freeze_mapping()
            vae_on = torch.tensor(0.0, device=x.device)
        else:
            vae_on = torch.tensor(1.0, device=x.device)

        n = torch.log1p(tensors[REGISTRY_KEYS_VT.N_KEY])
        n_hat = torch.log1p(generative_outputs["new"])
        vel_loss = nn.MSELoss()(n, n_hat) * self.vel_lambda

        if self.current_epoch > self.constrain_vf_after_epochs:
            if self.space == "gene_space":
                v_hat = generative_outputs["vel"]
                vhc = v_hat.clone().detach()
                k = tensors[REGISTRY_KEYS_VT.KNN_KEY].clone().detach()
                ts = tensors[REGISTRY_KEYS_VT.TS_KEY].clone().detach()

                v_proj = self.nc.project(x, vhc, k, ts)
                if self.nbr_loss == "mse":
                    nbr_loss = nn.MSELoss()(v_hat, v_proj) * self.nbr_lambda
                elif self.nbr_loss == "cs":
                    nbr_loss = torch.nn.CosineEmbeddingLoss()(
                        v_hat, v_proj, torch.ones(v_hat.shape[0], device=v_hat.device)
                    )
                    nbr_loss = nbr_loss * self.nbr_lambda

            elif self.space == "latent_space":
                z = inference_outputs["z"]
                vz = inference_outputs["vz"]
                zf = self._regular_inference(self.nc.X, self.nc.b)["qz"].loc

                zc = z.clone().detach()
                vzc = vz.clone().detach()
                zfc = zf.clone().detach()
                kc = tensors[REGISTRY_KEYS_VT.KNN_KEY].clone().detach()
                ts = tensors[REGISTRY_KEYS_VT.TS_KEY].clone().detach()
                vz_proj = self.nc.project(zc, vzc, kc, ts, zfull=zfc)

                if self.nbr_loss == "mse":
                    nbr_loss = nn.MSELoss()(vz, vz_proj) * self.nbr_lambda
                elif self.nbr_loss == "cs":
                    nbr_loss = torch.nn.CosineEmbeddingLoss()(vz, vz_proj, torch.ones(vz.shape[0], device=vz.device))
                    nbr_loss = nbr_loss * self.nbr_lambda
            elif self.space == "none":
                nbr_loss = torch.tensor(0.0, device=x.device)
        else:
            nbr_loss = torch.tensor(0.0, device=x.device)

        ### vae losses
        kld_z = kl(inference_outputs["qz"], generative_outputs["pz"]).sum(dim=1)
        if not self.use_observed_lib_size:
            kld_l = kl(
                inference_outputs["ql"],
                generative_outputs["pl"],
            ).sum(dim=1)
        else:
            kld_l = torch.tensor(0.0, device=x.device)

        kld_loss = self.kld_lambda * kld_z + kld_l

        rcn_loss = -generative_outputs["px"].log_prob(x).sum(-1)
        rcn_loss = rcn_loss * self.rcn_lambda

        loss = torch.mean(rcn_loss * vae_on + kld_loss * vae_on + vel_loss + nbr_loss)

        kl_local = dict(kl_divergence_l=kld_l, kl_divergence_z=kld_z)

        if self.verbose > 0 and self.global_step % self.verbose == 0:
            print(
                f"(pre-weighted) Loss: {loss.mean()}, rloss: {reconst_loss.mean()}, kloss{weighted_kl_local.mean()}, vloss: {velocity_loss.mean()}, nloss {neighborhood_loss.mean()}."
            )

        return LossOutput(loss=loss, reconstruction_loss=rcn_loss, kl_local=kl_local)

    def freeze_mapping(self):
        for param in self.z_encoder.parameters():
            param.requires_grad = False

        for param in self.l_encoder.parameters():
            param.requires_grad = False

        for param in self.decoder.parameters():
            param.requires_grad = False


class SplicingVelVAE(BaseLatentModeModuleClass):
    """
    Variational Autoencoder (VAE) model with velocity modeling.

    Args:
        transcriptome: The transcriptome data.
        n_input (int): The number of input dimensions.
        n_batch (int): The number of batch dimensions (default: 0).
        n_labels (int): The number of label dimensions (default: 0).
        n_hidden (int): The number of hidden units in the encoder and decoder layers (default: 128).
        n_latent (int): The dimensionality of the latent space (default: 50).
        n_layers (int): The number of layers in the encoder and decoder (default: 1).
        n_continuous_cov (int): The number of continuous covariates (default: 0).
        n_cats_per_cov (Optional[Iterable[int]]): The number of categories per covariate (default: None).
        dropout_rate (float): The dropout rate (default: 0.0).
        dispersion (str): The dispersion method (default: "gene").
        log_variational (bool): Whether to log-transform the input data for the variational encoder (default: True).
        gene_likelihood (Literal["zinb", "nb", "poisson", "normal"]): The likelihood model for gene expression (default: "zinb").
        latent_distribution (str): The distribution of the latent space (default: "normal").
        encode_covariates (bool): Whether to encode covariates (default: False).
        deeply_inject_covariates (bool): Whether to deeply inject covariates into the encoder and decoder (default: True).
        use_batch_norm (Literal["encoder", "decoder", "none", "both"]): The usage of batch normalization (default: "none").
        use_layer_norm (Literal["encoder", "decoder", "none", "both"]): The usage of layer normalization (default: "none").
        use_linear_decoder (bool): Whether to use a linear decoder (default: True).
        use_size_factor_key (bool): Whether to use a size factor key (default: False).
        use_observed_lib_size (bool): Whether to use the observed library size (default: True).
        library_log_means (Optional[np.ndarray]): The log means of the library size (default: None).
        library_log_vars (Optional[np.ndarray]): The log variances of the library size (default: None).
        var_activation (Optional[Callable]): The activation function for the variance (default: None).
        latent_data_type (Optional[LatentDataType]): The data type of the latent space (default: None).
        neighborhood_loss (Literal["mse", "cs"]): The neighborhood loss function (default: "cs").
        neighborhood_space (Literal["latent_space", "gene_space", "none"]): The space for neighborhood computation (default: "latent_space").
        gamma_mode (Literal["fixed", "learned"]): The mode for the gamma model (default: "learned").
        recon_lambda (float): The weight for the reconstruction loss (default: 1.0).
        velocity_lambda (float): The weight for the velocity loss (default: 10.0).
        neighborhood_lambda (float): The weight for the neighborhood loss (default: 10.0).
        kld_lambda (float): The weight for the Kullback-Leibler divergence loss (default: 0.1).
        neighborhood_kwargs: Additional keyword arguments for the NeighborhoodConstraint class (default: {}).
        vectorfield_kwargs: Additional keyword arguments for the VectorField class (default: {}).
        gamma_kwargs: Additional keyword arguments for the gamma model (default: {}).
        verbose (int): The verbosity level (default: 0).
    """

    def __init__(
        self,
        transcriptome,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 50,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.0,
        dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb", "poisson", "normal"] = "zinb",
        latent_distribution: str = "normal",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_linear_decoder: bool = True,
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        var_activation: Optional[Callable] = None,
        latent_data_type: Optional[LatentDataType] = None,
        neighborhood_loss: Literal["mse", "cs"] = "cs",
        neighborhood_space: Literal["latent_space", "gene_space", "none"] = "latent_space",
        gamma_mode: Literal["fixed", "learned"] = "learned",
        recon_lambda: float = 1.0,
        velocity_lambda: float = 10.0,
        neighborhood_lambda: float = 10.0,
        kld_lambda: float = 0.1,
        neighborhood_kwargs={},
        vectorfield_kwargs={},
        gamma_kwargs={},
        verbose=0,
    ):
        super().__init__()

        # structuring learning
        self.verbose = verbose
        self.kld_lambda = kld_lambda
        self.rcn_lambda = recon_lambda
        self.nbr_lambda = neighborhood_lambda
        self.vel_lambda = velocity_lambda
        self.nbr_loss = neighborhood_loss

        # velocity modelling
        self.gamma_mode = gamma_mode
        self.space = neighborhood_space

        # model hyperparams
        self.use_linear_decoder = use_linear_decoder
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self._latent_data_type = latent_data_type
        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size

        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, " "must provide library_log_means and library_log_vars."
                )

            self.register_buffer("library_log_means", torch.from_numpy(library_log_means).float())
            self.register_buffer("library_log_vars", torch.from_numpy(library_log_vars).float())

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None

        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent + n_continuous_cov
        if self.use_linear_decoder:
            self.decoder = LinearDecoderSCVI(
                n_input_decoder,
                n_input,
                n_cat_list=cat_list,
                use_batch_norm=use_batch_norm_decoder,
                use_layer_norm=use_layer_norm_decoder,
            )
        else:
            self.decoder = DecoderSCVI(
                n_input_decoder,
                n_input,
                n_cat_list=cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                inject_covariates=deeply_inject_covariates,
                use_batch_norm=use_batch_norm_decoder,
                use_layer_norm=use_layer_norm_decoder,
                scale_activation="softplus" if use_size_factor_key else "softmax",
            )

        self.vf = VectorField(latent_dim=n_latent, **vectorfield_kwargs)

        self.nc = NeighborhoodConstraint(X=transcriptome, **neighborhood_kwargs)

    def _get_inference_input(
        self,
        tensors,
    ):
        batch_index = tensors[REGISTRY_KEYS_VT.BATCH_KEY]

        cont_key = REGISTRY_KEYS_VT.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS_VT.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        if self.latent_data_type is None:
            x = tensors[REGISTRY_KEYS_VT.X_KEY]
            input_dict = dict(x=x, batch_index=batch_index, cont_covs=cont_covs, cat_covs=cat_covs)
        else:
            """
            if self.latent_data_type == "dist":
                qzm = tensors[REGISTRY_KEYS.LATENT_QZM_KEY]
                qzv = tensors[REGISTRY_KEYS.LATENT_QZV_KEY]
                input_dict = dict(qzm=qzm, qzv=qzv)
            else:
                raise ValueError(f"Unknown latent data type: {self.latent_data_type}")
            """
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        vz = inference_outputs["vz"]
        library = inference_outputs["library"]
        batch_index = tensors[REGISTRY_KEYS_VT.BATCH_KEY]
        y = tensors[REGISTRY_KEYS_VT.LABELS_KEY]

        cont_key = REGISTRY_KEYS_VT.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS_VT.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        size_factor_key = REGISTRY_KEYS_VT.SIZE_FACTOR_KEY
        size_factor = torch.log(tensors[size_factor_key]) if size_factor_key in tensors.keys() else None

        input_dict = dict(
            z=z,
            vz=vz,
            library=library,
            batch_index=batch_index,
            y=y,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
            size_factor=size_factor,
        )
        return input_dict

    def _compute_local_library_params(self, batch_index):
        """
        Computes local library parameters.
        Compute two tensors of shape (batch_index.shape[0], 1) where each
        element corresponds to the mean and variances, respectively, of the
        log library sizes in the batch the cell corresponds to.
        """
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(one_hot(batch_index, n_batch), self.library_log_means)
        local_library_log_vars = F.linear(one_hot(batch_index, n_batch), self.library_log_vars)
        return local_library_log_means, local_library_log_vars

    @auto_move_data
    def _regular_inference(self, x, batch_index, cont_covs=None, cat_covs=None, n_samples=1):
        """
        High level inference method.
        Runs the inference (encoder) model.
        """
        x_ = x

        # prepare for encoder
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        # run inference
        qz, z = self.z_encoder(encoder_input, batch_index, *categorical_input)
        vz = self.vf(z)

        ql = None
        if not self.use_observed_lib_size:
            ql, library_encoded = self.l_encoder(encoder_input, batch_index, *categorical_input)
            library = library_encoded

        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand((n_samples, library.size(0), library.size(1)))
            else:
                library = ql.sample((n_samples,))

        outputs = dict(z=z, vz=vz, qz=qz, ql=ql, library=library)
        return outputs

    def _cached_inference(self, qzm, qzv, n_samples=1):
        """
        if self.latent_data_type == "dist":
            dist = Normal(qzm, qzv.sqrt())
            # use dist.sample() rather than rsample because we aren't optimizing
            # the z in latent/cached mode
            untran_z = dist.sample() if n_samples == 1 else dist.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
        else:
            raise ValueError(f"Unknown latent data type: {self.latent_data_type}")
        outputs = dict(z=z, qz_m=qzm, qz_v=qzv, ql=None, library=None)
        return outputs"""
        pass  # TODO

    @auto_move_data
    def generative(
        self, z, vz, library, batch_index, cont_covs=None, cat_covs=None, size_factor=None, y=None, transform_batch=None
    ):
        """Runs the generative model."""
        ### formatting
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library

        z_future = z + vz

        if cont_covs is None:
            decoder_input = z
            decoder_input_future = z_future

        elif z.dim() != cont_covs.dim():
            decoder_input = torch.cat([z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1)
            decoder_input_future = torch.cat(
                [z_future, cont_covs.unsqueeze(0).expand(z_future.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)
            decoder_input_future = torch.cat([z_future, cont_covs], dim=-1)

        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            decoder_input,
            size_factor,
            batch_index,
            *categorical_input,
            y,
        )

        px_scale_future, px_r_future, px_rate_future, px_dropout_future = self.decoder(
            self.dispersion,
            decoder_input_future,
            size_factor,
            batch_index,
            *categorical_input,
            y,
        )

        vel = px_rate_future - px_rate

        uns = vel + self.loggamma.exp() * px_rate
        uns = torch.clip(uns, 0, 100)
        uns = torch.nan_to_num(uns, nan=0, neginf=0, posinf=0)

        if self.dispersion == "gene-label":
            px_r = F.linear(one_hot(y, self.n_labels), self.px_r)  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        if self.gene_likelihood == "zinb":
            px = NoWarningZINB(mu=px_rate, theta=px_r, zi_logits=px_dropout, scale=px_scale, validate_args=False)

        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale, validate_args=False)
        elif self.gene_likelihood == "poisson":
            px = Poisson(px_rate, scale=px_scale)
        elif self.gene_likelihood == "normal":
            px = Normal(loc=px_rate, scale=px_r)

        # Priors
        if self.use_observed_lib_size:
            pl = None
        else:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        return dict(
            px=px,
            vel=vel,
            uns=uns,
            pl=pl,
            pz=pz,
        )

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
    ):
        """Computes the loss function for the model."""
        x = tensors[REGISTRY_KEYS_VT.X_KEY]

        ### perform vae freezing
        if self.current_epoch > self.freeze_vae_after_epochs:
            self.freeze_mapping()
            vae_on = torch.tensor(0.0, device=x.device)
        else:
            vae_on = torch.tensor(1.0, device=x.device)

        u = torch.log1p(tensors[REGISTRY_KEYS_VT.U_KEY])
        u_hat = torch.log1p(generative_outputs["uns"])
        vel_loss = nn.MSELoss()(u, u_hat) * self.vel_lambda

        if self.current_epoch > self.constrain_vf_after_epochs:
            if self.space == "gene_space":
                v_hat = generative_outputs["vel"]
                vhc = v_hat.clone().detach()
                k = tensors[REGISTRY_KEYS_VT.KNN_KEY].clone().detach()
                ts = tensors[REGISTRY_KEYS_VT.TS_KEY].clone().detach()

                v_proj = self.nc.project(x, vhc, k, ts)
                if self.nbr_loss == "mse":
                    nbr_loss = nn.MSELoss()(v_hat, v_proj) * self.nbr_lambda
                elif self.nbr_loss == "cs":
                    nbr_loss = torch.nn.CosineEmbeddingLoss()(
                        v_hat, v_proj, torch.ones(v_hat.shape[0], device=v_hat.device)
                    )
                    nbr_loss = nbr_loss * self.nbr_lambda

            elif self.space == "latent_space":
                z = inference_outputs["z"]
                vz = inference_outputs["vz"]
                zf = self._regular_inference(self.nc.X, self.nc.b)["qz"].loc

                zc = z.clone().detach()
                vzc = vz.clone().detach()
                zfc = zf.clone().detach()
                kc = tensors[REGISTRY_KEYS_VT.KNN_KEY].clone().detach()
                ts = tensors[REGISTRY_KEYS_VT.TS_KEY].clone().detach()

                vz_proj = self.nc.project(zc, vzc, kc, ts, zfull=zfc)

                if self.nbr_loss == "mse":
                    nbr_loss = nn.MSELoss()(vz, vz_proj) * self.nbr_lambda
                elif self.nbr_loss == "cs":
                    nbr_loss = torch.nn.CosineEmbeddingLoss()(vz, vz_proj, torch.ones(vz.shape[0], device=vz.device))
                    nbr_loss = nbr_loss * self.nbr_lambda
            elif self.space == "none":
                nbr_loss = torch.tensor(0.0, device=x.device)
        else:
            nbr_loss = torch.tensor(0.0, device=x.device)

        ### vae losses
        kld_z = kl(inference_outputs["qz"], generative_outputs["pz"]).sum(dim=1)
        if not self.use_observed_lib_size:
            kld_l = kl(
                inference_outputs["ql"],
                generative_outputs["pl"],
            ).sum(dim=1)
        else:
            kld_l = torch.tensor(0.0, device=x.device)

        kld_loss = self.kld_lambda * kld_z + kld_l

        rcn_loss = -generative_outputs["px"].log_prob(x).sum(-1)
        rcn_loss = rcn_loss * self.rcn_lambda

        loss = torch.mean(rcn_loss * vae_on + kld_loss * vae_on + vel_loss + nbr_loss)

        kl_local = dict(kl_divergence_l=kld_l, kl_divergence_z=kld_z)

        if self.verbose > 0 and self.global_step % self.verbose == 0:
            print(
                f"(pre-weighted) Loss: {loss.mean()}, rloss: {reconst_loss.mean()}, kloss{weighted_kl_local.mean()}, vloss: {velocity_loss.mean()}, nloss {neighborhood_loss.mean()}."
            )

        return LossOutput(loss=loss, reconstruction_loss=rcn_loss, kl_local=kl_local)

    def freeze_mapping(self):
        for param in self.z_encoder.parameters():
            param.requires_grad = False

        for param in self.l_encoder.parameters():
            param.requires_grad = False

        for param in self.decoder.parameters():
            param.requires_grad = False
