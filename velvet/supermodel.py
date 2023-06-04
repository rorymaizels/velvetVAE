"""code for VelvetSDE model"""
from typing import List, Optional, Tuple
from anndata import AnnData

import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
import torchsde

from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.module.base import BaseModuleClass, LossOutput
from scvi.model.base import ArchesMixin, BaseModelClass, RNASeqMixin, VAEMixin

from velvet.constants import REGISTRY_KEYS_SDE
from velvet.mixins import VelvetMixin, SDETrainingMixin, SimulationMixin

class VelvetSDE(
    RNASeqMixin, VAEMixin, ArchesMixin, SDETrainingMixin, BaseModelClass, VelvetMixin, SimulationMixin
):
    """
    A class for the VelvetSDE model, which includes various mixin classes for
    modeling dynamics as a neural stochastic differential equation.
    """

    def __init__(self, model, sde_module, markov_module, **kwargs) -> None:
        """
        Initialize the VelvetSDE model.

        Parameters:
        model:
            The initial Velvet model to be used.
        sde_module:
            The nSDE module.
        markov_module:
            The Markov module.
        **kwargs:
            Additional keyword arguments for SDVAE.
        """
        super().__init__(model.adata)

        self.prior_model = model
        use_time = REGISTRY_KEYS_SDE.TIME_KEY in self.adata_manager.data_registry

        self.module = SDVAE(
            adata=model.adata, 
            prior_module=model.module, 
            sde_module=sde_module, 
            markov_module=markov_module,
            use_time=use_time, 
            **kwargs
        )

        self._model_summary_string = "Velvet SDE model based on VAE model"
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def setup_anndata(
        cls,
        model,
        x_layer: Optional[str] = None,
        t_key: Optional[str] = None,
        index_key: Optional[str] = None,
        batch_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """
        Set up anndata for the VelvetSDE model.

        Parameters:
        model: Type of the model, replace `Any` with specific type.
            The model to be used.
        x_layer: str, optional
            The layer to be used for transcriptome. Default is None.
        t_key: str, optional
            The key to be used for time. Default is None.
        index_key: str, optional
            The key to be used for knn index. Default is None.
        batch_key: str, optional
            The key to be used for batch. Default is None.
        categorical_covariate_keys: list of str, optional
            The keys to be used for categorical covariates. Default is None.
        continuous_covariate_keys: list of str, optional
            The keys to be used for continuous covariates. Default is None.
        **kwargs: Any
            Additional keyword arguments.
        """
        adata = model.adata
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS_SDE.X_KEY, x_layer, is_count_data=False),
            NumericalObsField(REGISTRY_KEYS_SDE.TIME_KEY, t_key, required=False),
            NumericalObsField(REGISTRY_KEYS_SDE.INDEX_KEY, index_key, required=True),
            CategoricalObsField(REGISTRY_KEYS_SDE.BATCH_KEY, batch_key),
            CategoricalJointObsField(REGISTRY_KEYS_SDE.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS_SDE.CONT_COVS_KEY, continuous_covariate_keys),
        ]

        # register new fields for latent mode if needed
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)



class SDVAE(BaseModuleClass):
    def __init__(
        self,
        adata: AnnData,
        prior_module: nn.Module,
        sde_module: nn.Module,
        markov_module: nn.Module,
        use_time: bool = False,
    ) -> None:
        """
        Initialize the SDVAE class.

        Args:
            adata (AnnData): Input anndata.
            prior_module (nn.Module): Velvet module for the model.
            sde_module (nn.Module): nSDE module for the model.
            markov_module (nn.Module): Markov module for the model.
            include_vel_loss (bool, optional): Flag to include velocity loss. Default is False.
        """
        super().__init__()
        # core components
        self.adata = adata
        self.sde = sde_module
        self.mp = markov_module
        self.core = prior_module
        self.z_encoder = prior_module.z_encoder
        self.n_latent = prior_module.n_latent
        self.use_time = use_time

    def _get_inference_input(
        self,
        tensors,
    ):
        batch_index = tensors[REGISTRY_KEYS_SDE.BATCH_KEY]
        if self.use_time:
            time_index = tensors[REGISTRY_KEYS_SDE.TIME_KEY]
        else:
            time_index = None

        cell_index = tensors[REGISTRY_KEYS_SDE.INDEX_KEY]

        cont_key = REGISTRY_KEYS_SDE.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS_SDE.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        x = tensors[REGISTRY_KEYS_SDE.X_KEY]
        input_dict = dict(
            x=x,
            t=time_index,
            cell_index=cell_index,
            batch_index=batch_index,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z_sort = inference_outputs["z_sort"]
        init_idx = inference_outputs["init_idx"]
        init_cells = inference_outputs["init_cells"]
        cell_index = inference_outputs["cell_index"]

        input_dict = dict(z_sort=z_sort, init_idx=init_idx, init_cells=init_cells, cell_index=cell_index)
        return input_dict

    def inference(
        self,
        x,
        t,
        cell_index,
        batch_index,
        cont_covs=None,
        cat_covs=None,
    ):
        x_ = x
        if self.core.log_variational:
            x_ = torch.log(1 + x_)

        if cont_covs is not None and self.core.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.core.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        # run inference without gradients
        with torch.no_grad():
            qz, z = self.z_encoder(encoder_input, batch_index, *categorical_input)

        index_sort = cell_index.squeeze().argsort()
        z_sort = z[index_sort, :]
        if self.use_time:
            t_sort = t[index_sort, :]
        else:
            t_sort = None

        init_idx, init_cells = self.select_initial_states(z_sort, t_sort)

        outputs = dict(z_sort=z_sort, init_idx=init_idx, init_cells=init_cells, cell_index=cell_index)
        return outputs

    def generative(self, z_sort, init_idx, init_cells, cell_index):
        self.timespan = torch.linspace(0, self.t_max, self.n_steps, device=init_cells.device)

        # Markov approximation of Fokker-Planck dynamics
        z_markov = self.mp.random_walk(
            z=z_sort,
            initial_states=init_idx,
            n_jumps=self.n_markov_steps,
            n_steps=self.n_steps,
        ).permute(
            1, 0, 2
        )  # just to equate to SDE, for symmetry downstream

        # SDE simulation of Fokker-Planck system
        z_diffeq = torchsde.sdeint_adjoint(self.sde, init_cells, self.timespan, method="midpoint", dt=self.dt)

        # reshape to: n_cells, n_simulations, n_timesteps, n_latent
        # then remove the first time step (initial conditions)
        z_m = z_markov.reshape(self.n_steps, self.n_trajectories, self.n_simulations, self.n_latent).permute(
            1, 2, 0, 3
        )[:, :, 1:, :]

        z_s = z_diffeq.reshape(self.n_steps, self.n_trajectories, self.n_simulations, self.n_latent).permute(
            1, 2, 0, 3
        )[:, :, 1:, :]

        return dict(
            z_m=z_m,
            z_s=z_s,
        )

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
    ):
        z_s = generative_outputs["z_s"]
        z_m = generative_outputs["z_m"].detach()

        rcn_loss = self.kld_loss(z_m, z_s)
        loss = torch.mean(rcn_loss)

        return LossOutput(
            loss=loss, reconstruction_loss=rcn_loss, kl_local={"kld": torch.tensor(0.0, device=self.device)}
        )

    def kld_loss(self, z_m: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
        """
        Compute the Kullback-Leibler divergence loss.

        Args:
            z_m (torch.Tensor): Tensor representing z_m in the model.
            z_s (torch.Tensor): Tensor representing z_s in the model.

        Returns:
            torch.Tensor: Computed Kullback-Leibler divergence loss.
        """
        zm_mu = z_m.mean(dim=1, keepdims=True)
        if self.mp.use_terminal_states:
            zm_sd = z_m.std(dim=1, keepdims=True).add(0.1)
        else:
            zm_sd = z_m.std(dim=1, keepdims=True)

        pxm = Normal(loc=zm_mu, scale=zm_sd)
        zs_mu = z_s.mean(dim=1, keepdims=True)
        zs_sd = z_s.std(dim=1, keepdims=True)
        pxs = Normal(loc=zs_mu, scale=zs_sd)
        kld = kl(pxs, pxm).sum(-1)
        return kld

    def select_initial_states(self, z: torch.Tensor, t):
        """
        Select initial states for simulation.

        Args:
            z (torch.Tensor): Latent variable tensor.
            t (torch.Tensor): Time tensor.

        Returns:
            tuple: Tuple containing indices of selected initial cells and their corresponding cells.
        """
        if self.use_time:
            time_index = torch.argsort(t.squeeze())
            z_timesorted = z[time_index, :]
            valid_indices = int((self.beginning_pct / 100) * z.shape[0])
            initial_indices = torch.randperm(valid_indices)[: self.n_trajectories]
            initial_cells = z_timesorted[initial_indices, :]
            chosen_cell_indices = time_index[initial_indices]
        else:
            valid_indices = z.shape[0]
            chosen_cell_indices = torch.randperm(valid_indices)[: self.n_trajectories]
            initial_cells = z[chosen_cell_indices, :]

        repeated_indices = chosen_cell_indices.repeat_interleave(self.n_simulations)
        repeated_cells = initial_cells.repeat_interleave(self.n_simulations, dim=0)
        return repeated_indices, repeated_cells

    def freeze_mapping(self) -> None:
        """
        Freeze the mapping by making all parameters in the encoders and decoder non-trainable.
        """
        for param in self.z_encoder.parameters():
            param.requires_grad = False

        for param in self.l_encoder.parameters():
            param.requires_grad = False

        for param in self.decoder.parameters():
            param.requires_grad = False
