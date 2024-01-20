"""higher level model class for Velvet"""
import logging
from typing import List, Optional

from anndata import AnnData
from scipy.sparse import issparse

import torch
from torch import nn
from scvi._compat import Literal
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField,
)
from scvi.model._utils import _init_library_size
from scvi.model.base import ArchesMixin, BaseModelClass, RNASeqMixin, VAEMixin
from scvi.utils import setup_anndata_dsp

from velvetvae.module import VelVAE, SplicingVelVAE
from velvetvae.constants import REGISTRY_KEYS_VT
from velvetvae.mixins import ModifiedUnsupervisedTrainingMixin, VelvetMixin, ModellingMixin

logger = logging.getLogger(__name__)

_SCVI_LATENT_QZM = "_scvi_latent_qzm"
_SCVI_LATENT_QZV = "_scvi_latent_qzv"

logger = logging.getLogger(__name__)


class Velvet(
    RNASeqMixin,
    VAEMixin,
    ArchesMixin,
    ModifiedUnsupervisedTrainingMixin,
    BaseModelClass,
    VelvetMixin,
    ModellingMixin,
):
    """
    Variational Estimation of Latent Velocity from Expression with Temporal resolution

    Parameters:
        adata (AnnData): AnnData object that has been registered via `SCVI.setup_anndata`.
        n_hidden (int): Number of nodes per hidden layer.
        n_latent (int): Dimensionality of the latent space.
        n_layers (int): Number of hidden layers used for encoder and decoder NNs.
        dropout_rate (float): Dropout rate for neural networks.
        dispersion (Literal["gene", "gene-batch", "gene-label", "gene-cell"]): One of the following dispersion parameter options:
            - "gene": Dispersion parameter of NB is constant per gene across cells.
            - "gene-batch": Dispersion can differ between different batches.
            - "gene-label": Dispersion can differ between different labels.
            - "gene-cell": Dispersion can differ for every gene in every cell.
        gene_likelihood (Literal["zinb", "nb", "poisson", "normal"]): One of the following gene likelihood models:
            - "nb": Negative binomial distribution.
            - "zinb": Zero-inflated negative binomial distribution.
            - "poisson": Poisson distribution.
            - "normal": Normal distribution.
        latent_distribution (Literal["normal", "ln"]): One of the following latent distributions:
            - "normal": Normal distribution.
            - "ln": Logistic normal distribution (Normal(0, I) transformed by softmax).
        vectorfield_kwargs: Keyword arguments passed to the vectorfield within VAE.
        **model_kwargs: Keyword arguments for VAE.

    Attributes:
        module: Velvet model module.
        _model_summary_string: Summary string describing the model parameters.
        init_params_: Initial model parameters.

    Methods:
        setup_model(gamma_kwargs={}): Set up the model with optional gamma parameters.
        setup_anndata(...): Set up the AnnData object for modeling.
        _get_latent_adata_from_adata(...): Get latent data from the AnnData object.
        _get_latent_fields(...): Get latent fields based on the mode.
        to_latent_mode(...): Convert the model to latent mode.
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 50,
        n_layers: int = 1,
        dropout_rate: float = 0.0,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson", "normal"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        linear_decoder: bool = True,
        batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        neighborhood_space: Literal["latent_space", "gene_space", "none"] = "latent_space",
        biophysical_model: Literal["simple", "full"] = "full",
        gamma_mode: Literal["fixed", "learned"] = "learned",
        labelling_time = None,
        neighborhood_kwargs={},
        vectorfield_kwargs={},
        **model_kwargs,
    ):
        super().__init__(adata)

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS_VT.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS_VT.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch
        use_size_factor_key = REGISTRY_KEYS_VT.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        use_similarity_graph = REGISTRY_KEYS_VT.TS_KEY in self.adata_manager.data_registry
        library_log_means, library_log_vars = None, None
        if not use_size_factor_key:
            library_log_means, library_log_vars = _init_library_size(self.adata_manager, n_batch)

        self.labelling_time = labelling_time
        self.gamma_mode = gamma_mode
        self.biophysical_model = biophysical_model
        self.n_latent = n_latent

        self.module = VelVAE(
            transcriptome=self.adata_manager.get_from_registry("X"),
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_labels=self.summary_stats.n_labels,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            use_size_factor_key=use_size_factor_key,
            use_batch_norm=batch_norm,
            use_layer_norm=layer_norm,
            use_linear_decoder=linear_decoder,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            neighborhood_space=neighborhood_space,
            biophysical_model=biophysical_model,
            gamma_mode=gamma_mode,
            neighborhood_kwargs=neighborhood_kwargs,
            vectorfield_kwargs=vectorfield_kwargs,
            labelling_time=labelling_time,
            use_similarity_graph=use_similarity_graph,
            **model_kwargs,
        )
        self._model_summary_string = (
            "Velvet model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}, dispersion: {}, gene_likelihood: {}, latent_distribution: {}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            dispersion,
            gene_likelihood,
            latent_distribution,
        )
        self.init_params_ = self._get_init_params(locals())

    def setup_model(
        self,
        gamma_kwargs={},
    ):
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        total = self.adata_manager.get_from_registry("X")
        total = total.A if issparse(total) else total

        new = self.adata_manager.get_from_registry("N")
        new = new.A if issparse(new) else new

        if REGISTRY_KEYS_VT.TIME_KEY in self.adata_manager.data_registry:
            t = self.adata_manager.get_from_registry(REGISTRY_KEYS_VT.TIME_KEY)
            self.variable_labelling = True
        elif self.labelling_time is not None:
            t = self.labelling_time
            self.variable_labelling = False
        else:
            raise ValueError(
                "Please supply labelling time either in setup_anndata or as argument to model instantiation."
            )
            
        gamma_numpy = self.find_gamma(
            total=total, new=new, t=t, model=self.biophysical_model, **gamma_kwargs
        )
        gamma = torch.tensor(gamma_numpy, device=torch_device, dtype=torch.float32)

        if self.gamma_mode == "learned":
            self.module.loggamma = nn.Parameter(gamma.log())
            self.module.ss_gamma = gamma
        else:
            self.module.loggamma = gamma.log()

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        x_layer: Optional[str] = None,
        n_layer: Optional[str] = None,
        knn_layer: Optional[str] = None,
        time_key: Optional[str] = None,
        ts_layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        %(summary)s.
        Parameters
        ----------
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())        
        anndata_fields = [
            LayerField(REGISTRY_KEYS_VT.X_KEY, x_layer, is_count_data=False),
            LayerField(REGISTRY_KEYS_VT.N_KEY, n_layer, is_count_data=False),
            ObsmField(REGISTRY_KEYS_VT.KNN_KEY, knn_layer),
            CategoricalObsField(REGISTRY_KEYS_VT.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS_VT.LABELS_KEY, labels_key),
            NumericalObsField(REGISTRY_KEYS_VT.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS_VT.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS_VT.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        if ts_layer is not None:
            anndata_fields.append(ObsmField(REGISTRY_KEYS_VT.TS_KEY, ts_layer))
        if time_key is not None:
            anndata_fields.append(
                NumericalObsField(REGISTRY_KEYS_VT.TIME_KEY, time_key, required=False)
            )   

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

class Svelvet(
    RNASeqMixin,
    VAEMixin,
    ArchesMixin,
    ModifiedUnsupervisedTrainingMixin,
    BaseModelClass,
    VelvetMixin,
    ModellingMixin,
):
    """
    Variational Estimation of Latent Velocity from Expression with Temporal resolution for Splicing data

    Parameters:
        adata (AnnData): AnnData object that has been registered via `SCVI.setup_anndata`.
        n_hidden (int): Number of nodes per hidden layer.
        n_latent (int): Dimensionality of the latent space.
        n_layers (int): Number of hidden layers used for encoder and decoder NNs.
        dropout_rate (float): Dropout rate for neural networks.
        dispersion (Literal["gene", "gene-batch", "gene-label", "gene-cell"]): One of the following dispersion parameter options:
            - "gene": Dispersion parameter of NB is constant per gene across cells.
            - "gene-batch": Dispersion can differ between different batches.
            - "gene-label": Dispersion can differ between different labels.
            - "gene-cell": Dispersion can differ for every gene in every cell.
        gene_likelihood (Literal["zinb", "nb", "poisson", "normal"]): One of the following gene likelihood models:
            - "nb": Negative binomial distribution.
            - "zinb": Zero-inflated negative binomial distribution.
            - "poisson": Poisson distribution.
            - "normal": Normal distribution.
        latent_distribution (Literal["normal", "ln"]): One of the following latent distributions:
            - "normal": Normal distribution.
            - "ln": Logistic normal distribution (Normal(0, I) transformed by softmax).
        vectorfield_kwargs: Keyword arguments passed to the vectorfield within VAE.
        **model_kwargs: Keyword arguments for VAE.

    Attributes:
        module: VelvetSplicing model module.
        _model_summary_string: Summary string describing the model parameters.
        init_params_: Initial model parameters.

    Methods:
        setup_model(gamma_kwargs={}): Set up the model with optional gamma parameters.
        setup_anndata(...): Set up the AnnData object for modeling.
        _get_latent_adata_from_adata(...): Get latent data from the AnnData object.
        _get_latent_fields(...): Get latent fields based on the mode.
        to_latent_mode(...): Convert the model to latent mode.
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 50,
        n_layers: int = 1,
        dropout_rate: float = 0.0,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson", "normal"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        linear_decoder: bool = True,
        batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        neighborhood_space: Literal["latent_space", "gene_space", "none"] = "latent_space",
        gamma_mode: Literal["fixed", "learned"] = "learned",
        neighborhood_kwargs={},
        vectorfield_kwargs={},
        **model_kwargs,
    ):
        super().__init__(adata)

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS_VT.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS_VT.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch
        use_size_factor_key = REGISTRY_KEYS_VT.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        use_similarity_graph = REGISTRY_KEYS_VT.TS_KEY in self.adata_manager.data_registry
        library_log_means, library_log_vars = None, None
        if not use_size_factor_key:
            library_log_means, library_log_vars = _init_library_size(self.adata_manager, n_batch)

        self.gamma_mode = gamma_mode
        self.labelling_time = None

        self.module = SplicingVelVAE(
            transcriptome=self.adata_manager.get_from_registry("X"),
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_labels=self.summary_stats.n_labels,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            use_size_factor_key=use_size_factor_key,
            use_batch_norm=batch_norm,
            use_layer_norm=layer_norm,
            use_linear_decoder=linear_decoder,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            neighborhood_space=neighborhood_space,
            gamma_mode=gamma_mode,
            neighborhood_kwargs=neighborhood_kwargs,
            vectorfield_kwargs=vectorfield_kwargs,
            use_similarity_graph=use_similarity_graph,
            **model_kwargs,
        )
        self._model_summary_string = (
            "Velvet model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}, dispersion: {}, gene_likelihood: {}, latent_distribution: {}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            dispersion,
            gene_likelihood,
            latent_distribution,
        )
        self.init_params_ = self._get_init_params(locals())

    def setup_model(
        self,
        gamma_kwargs={},
    ):
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        total = self.adata_manager.get_from_registry("X")
        total = total.A if issparse(total) else total

        unspliced = self.adata_manager.get_from_registry("U")
        unspliced = unspliced.A if issparse(unspliced) else unspliced

        gamma_numpy = self.find_gamma_splicing(total=total, unspliced=unspliced, **gamma_kwargs)
        gamma = torch.tensor(gamma_numpy, device=torch_device)

        if self.gamma_mode == "learned":
            self.module.loggamma = nn.Parameter(gamma.log())
            self.module.ss_gamma = gamma
        else:
            self.module.loggamma = gamma.log()

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        x_layer: Optional[str] = None,
        u_layer: Optional[str] = None,
        knn_layer: Optional[str] = None,
        ts_layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        %(summary)s.
        Parameters
        ----------
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS_VT.X_KEY, x_layer, is_count_data=False),
            LayerField(REGISTRY_KEYS_VT.U_KEY, u_layer, is_count_data=False),
            ObsmField(REGISTRY_KEYS_VT.KNN_KEY, knn_layer),
            CategoricalObsField(REGISTRY_KEYS_VT.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS_VT.LABELS_KEY, labels_key),
            NumericalObsField(REGISTRY_KEYS_VT.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS_VT.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS_VT.CONT_COVS_KEY, continuous_covariate_keys),
        ]   
        # register new fields for latent mode if needed
        if ts_layer is not None:
            anndata_fields.append(ObsmField(REGISTRY_KEYS_VT.TS_KEY, ts_layer))
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)