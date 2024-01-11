"""mixins to support training and wider functionality."""
from velvet.submodule import MarkovProcess, SDE
from velvet.preprocessing import neighbors

from typing import Optional, Union, Tuple, List

from anndata._core.anndata import AnnData
from anndata._core.views import ArrayView
from numpy import ndarray
from scipy.sparse._csr import csr_matrix

import anndata as ann
import matplotlib.pyplot as plt
import scvelo as scv

import numpy as np
import torch

from scvi.dataloaders import DataSplitter
from scvi.train import TrainingPlan, TrainRunner

from scipy.sparse import issparse

from sklearn.decomposition import PCA
from scvelo.core import LinearRegression as scveloLR
from tqdm import tqdm, trange

import torchsde
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans


class VelvetMixin:
    """
    Mixin class with methods to learn steady-state parameters
    for the biophysical model of metabolic labelling.
    """

    def find_gamma(
        self,
        total: np.ndarray,
        new: np.ndarray,
        t = None,
        model: str = "full",
        gamma_min: float = 0.01,
        gamma_max: float = 10,
    ) -> np.ndarray:
        """
        Finds gamma, a parameter for the biophysical model of metabolic labelling.

        Parameters
        ----------
        total : np.ndarray
            The total count data.
        new : np.ndarray
            The new count data.
        t : float, optional
            Time variable for the full model, equivalent to labelling duration.
        model : str, optional
            The model to be used, default is 'full'. If 'simple', gamma = K. If 'full', gamma is calculated with time variable t.
        gamma_min : float, optional
            The minimum limit for gamma, default is 0.01.
        gamma_max : float, optional
            The maximum limit for gamma, default is 10.

        Returns
        -------
        gamma : np.ndarray
            The calculated gamma value(s).
        """
        if self.variable_labelling and model=='full':
            # learn a different gamma estimate per labelling duration, and average
            t = np.array(t).reshape(-1, 1)
            gammas = np.zeros((len(np.unique(t)), total.shape[1]))
            for i, ti in enumerate(np.unique(t)):
                tot_i = total[t.flatten()==ti]
                new_i = new[t.flatten()==ti]
                lr = scveloLR(fit_intercept=False, percentile=[5,95])
                lr.fit(tot_i, new_i)
                Ki = lr.coef_
                gamma_i = -np.log(1 - Ki) / ti
                gamma_i = np.clip(gamma_i, gamma_min, gamma_max)
                gammas[i] = gamma_i
            gamma = np.mean(gammas, axis=0) 
        else:
            lr = scveloLR(fit_intercept=False, percentile=[5,95])
            lr.fit(total, new)
            K = lr.coef_
            if model=='simple':
                gamma = K
            elif model=='full':
                gamma = -np.log(1 - K) / t
                gamma = np.clip(gamma, gamma_min, gamma_max)
        return gamma

    def find_gamma_splicing(
        self,
        total: np.ndarray,
        unspliced: np.ndarray,
        gamma_min: float = 0.01,
        gamma_max: float = 100,
    ) -> np.ndarray:
        """
        Finds gamma for splicing, a parameter for the biophysical model of metabolic labelling.

        Parameters
        ----------
        total : np.ndarray
            The total count data.
        unspliced : np.ndarray
            The unspliced count data.
        gamma_min : float, optional
            The minimum limit for gamma, default is 0.01.
        gamma_max : float, optional
            The maximum limit for gamma, default is 10.

        Returns
        -------
        np.ndarray
            The calculated gamma value(s) for splicing.
        """
        lr = scveloLR(fit_intercept=False, percentile=[5, 95])
        lr.fit(total, unspliced)
        K = lr.coef_
        return np.clip(K, gamma_min, gamma_max)

    def plot_latent_pca(self, color: str = "cell_type", title: str = "PCA of Latent Space") -> None:
        """
        Plots PCA of the latent space of the given data.

        Parameters
        ----------
        color : str, optional
            The cell type used for coloring the plot, default is 'cell_type'.
        title : str, optional
            The title of the plot, default is 'PCA of Latent Space'.

        Returns
        -------
        None
        """
        # Get the data from the registry
        X = self.adata_manager.get_from_registry("X")

        # Check if the data is sparse, convert to a regular array if it is
        X = X.A if issparse(X) else X

        torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        x = torch.tensor(X, device=torch_device)
        b = torch.arange(X.shape[0], device=torch_device)
        z = self.module.inference(x, b)["z"]
        v = self.module.vf(z)

        # Perform PCA on the latent state and velocity
        pca = PCA(n_components=2)
        zs = z.detach().cpu().numpy()
        zfs = (z + v).detach().cpu().numpy()

        # Fit and transform the PCA model on latent space data
        z_pca = pca.fit_transform(zs)
        zf_pca = pca.transform(zfs)
        v_pca = zf_pca - z_pca

        copy = self.adata.copy()
        copy.obsm["X_vae"] = z_pca
        copy.obsm["velocity_vae"] = v_pca
        copy.uns["velocity_params"] = {"embeddings": "vae"}

        # Plot velocity embedding stream
        scv.pl.velocity_embedding_stream(copy, basis="vae", title=title, color=color, show=True)

    def predict_velocity(self, numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """
        Predicts the velocity of model's adata.

        Parameters
        ----------
        numpy : bool, optional
            If True, converts the result into a numpy array.
            If False, the result remains a PyTorch tensor.
            Default is True.

        Returns
        -------
        np.ndarray or torch.Tensor
            The predicted velocity of the data, in the form of a numpy array if numpy=True,
            else in the form of a PyTorch tensor.
        """
        X = self.adata_manager.get_from_registry("X")
        X = X.A if issparse(X) else X

        if "t" in self.adata_manager.data_registry:
            t = self.adata_manager.get_from_registry("t")
        else: 
            try:
                t = self.labelling_time
            except AttributeError:
                print("No labelling time information supplied. Assumed to be 2 hours.")
                t = 2.0

        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.tensor(X, device=torch_device)
        b = torch.arange(X.shape[0], device=torch_device)

        # Use the model to infer the state and generate the output
        with torch.no_grad():
            inf = self.module.inference(x, b)
            gen = self.module.generative(inf["z"], inf["vz"], inf["library"], t, b)

        if numpy:
            return gen["vel"].detach().cpu().numpy()
        else:
            return gen["vel"]

    def generate_samples(self, n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates samples of the latent velocity and posterior velocity from model's data.

        Parameters
        ----------
        n_samples : int, optional
            The number of samples to generate, default is 100.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the latent velocity and the posterior velocity.
        """
        X = self.adata_manager.get_from_registry("X")

        X = X.A if issparse(X) else X

        torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        if "t" in self.adata_manager.data_registry:
            t = self.adata_manager.get_from_registry("t")
        elif self.labelling_time is not None:
            t = self.labelling_time
        else:
            print("No labelling time information supplied. Assumed to be 2 hours.")
            t = 2.0

        x = torch.tensor(X, device=torch_device)
        b = torch.arange(X.shape[0], device=torch_device)

        # Initialize latent and posterior velocities with zeros
        latent_velocity = torch.zeros((n_samples, x.shape[0], self.module.n_latent))
        posterior_velocity = torch.zeros((n_samples, *x.shape))

        # Loop through n_samples times and generate velocities
        with torch.no_grad():
            for i in trange(n_samples):
                # Get inferred state
                inf = self.module.inference(x, b)
                # Generate output
                gen = self.module.generative(inf["z"], inf["vz"], inf["library"], t, b)
                # Assign velocities
                latent_velocity[i] = inf["vz"]
                posterior_velocity[i] = gen["vel"]
        return latent_velocity, posterior_velocity

    def calculate_cellwise_uncertainty(self, log: bool = True) -> "AnnData":
        """
        Calculate the cell-wise uncertainty of the data.

        Parameters
        ----------
        log : bool, optional
            Whether to take the logarithm of the uncertainty values. Default is True.

        Returns
        -------
        AnnData
            The AnnData object with the added 'uncertainty' field in the observations.
        """
        # Generate samples and calculate uncertainty
        latent_velocity, _ = self.generate_samples()
        cell_var = latent_velocity.var(0).mean(1)
        scaled_cell_var = cell_var / (cell_var.mean() + 1e-6)
        if log:
            scaled_cell_var = scaled_cell_var.log()

        adata = self.adata_manager.get_from_registry("X")
        adata.obs["uncertainty"] = scaled_cell_var
        return adata

    def get_latent_dynamics(self, module=None, return_data: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the latent dynamics of the data.

        Parameters
        ----------
        module : Module, optional
            The module used to infer the state of the data. If None, the instance's module is used. Default is None.
        return_data : bool, optional
            Whether to return the inferred state and velocity. Default is True.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the inferred state and the velocity if return_data is True, otherwise no return value.
        """
        if module is None:
            module = self.module

        X = self.adata_manager.get_from_registry("X")
        X = X.A if issparse(X) else X
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        x = torch.tensor(X, device=torch_device)
        b = torch.arange(X.shape[0], device=torch_device)
        module.to(torch_device)
        with torch.no_grad():
            z = module.inference(x, b)["z"]
            vz = module.vf(z)

        self.adata.obsm["X_z"] = z.detach().cpu().numpy()
        self.adata.obsm["velocity_z"] = vz.detach().cpu().numpy()
        if return_data:
            return z, vz

    def infer_pseudotime(
        self,
        n_neighbors: int = 30,
        time_key: str = "t",
        latent_key: str = "X_z",
        velocity_key: str = "velocity_z",
        n_jobs: int = 1,
        verbose: bool = True,
    ) -> None:
        """
        Infer pseudotime using the provided model. Pseudotime is calculated by scVelo.

        Args:
            model (Any): The model to infer pseudotime.
            color (str, optional): The color to be used for scatter plots. Default is None.
            n_neighbors (int, optional): The number of neighbors for the velocity graph. Default is 30.
            time_key (str, optional): Key for the time in the model's adata. Default is 't'.
            latent_key (str, optional): Key for the latent data in the model's adata. Default is 'X_z'.
            velocity_key (str, optional): Key for the velocity data in the model's adata. Default is 'velocity_z'.
            n_jobs (int, optional): Number of jobs to run in parallel for velocity graph calculation. Default is 1.
            verbose (bool, optional): If True, print details during the calculation process. Default is True.

        Returns:
            None. This function modifies the model's adata in-place by adding pseudotime, end_points, and root_cells to it.
        """
        if verbose:
            print("Wrapper for velocity pseudotime calculated by scVelo.")
            print("Authors: Volker Bergen, Marius Lange, Stefan Peidli, F. Alexander Wolf & Fabian J. Theis")
            print("Paper: https://www.nature.com/articles/s41587-020-0591-3")

        self.get_latent_dynamics(return_data=False)
        sub = ann.AnnData(
            X=self.adata.obsm[latent_key],
            layers={"X": self.adata.obsm[latent_key], "velocity": self.adata.obsm[velocity_key]},
            obs=self.adata.obs.copy(),
        )
        neighbors(sub, total_layer="X", n_neighbors=n_neighbors, include_self=True)
        scv.tl.velocity_graph(sub, xkey="X", vkey="velocity", n_jobs=n_jobs)
        scv.tl.velocity_pseudotime(sub)
        scv.pp.pca(sub)

        fig = plt.figure(figsize=(18, 6))
        ax1, ax2, ax3 = fig.subplots(1, 3)

        scv.pl.scatter(
            sub,
            basis="pca",
            color="velocity_pseudotime",
            cmap="gnuplot",
            fontsize=22,
            size=40,
            legend_loc="on data",
            legend_fontsize=22,
            ax=ax1,
            show=False,
        )
        scv.pl.scatter(
            sub,
            basis="pca",
            color="end_points",
            cmap="gnuplot",
            fontsize=22,
            size=40,
            legend_loc="on data",
            legend_fontsize=22,
            ax=ax2,
            show=False,
        )
        scv.pl.scatter(
            sub,
            basis="pca",
            color="root_cells",
            cmap="gnuplot",
            fontsize=22,
            size=40,
            legend_loc="on data",
            legend_fontsize=22,
            ax=ax3,
        )

        self.adata.obs[time_key] = sub.obs.velocity_pseudotime
        self.adata.obs["end_points"] = sub.obs.end_points
        self.adata.obs["root_cells"] = sub.obs.root_cells

class ModellingMixin():
    """functions for downstream modelling & simulations"""
    def quantify_uncertainty(self, n_samples=100, return_data=False):
        lv, pv = self.generate_samples(n_samples=n_samples)
        sample_variance = lv.var(0)

        cellwise_var = sample_variance.mean(1)
        dwise_var = sample_variance.mean(0)

        cellwise_var_scaled = cellwise_var / cellwise_var.mean()
        dwise_var_scaled = dwise_var / dwise_var.mean()

        self.adata.obs['uncertainty'] = cellwise_var_scaled.detach().cpu().numpy()
        self.adata.uns['dimensionwise_uncertainty'] = dwise_var_scaled.detach().cpu().numpy()

        if return_data:
            return cellwise_var_scaled, dwise_var_scaled
        
    def get_gene_expression(
        self,
        x
    ):
        x = x.to(self.device)
        batch_index = torch.arange(x.shape[0], device=self.device)
        size_factor = torch.full(
            size=(x.shape[0], 1), 
            fill_value=self.module.median_library_size.log(),
            device=self.device
        )
        with torch.no_grad():
            _, _, gex, _ = self.module.decoder(
                self.module.dispersion,
                x,
                size_factor,
                batch_index,
            )
        return gex
        
    def get_trajectory_gene_expression(
        self,
        trajectories
    ):
        mapped_trajectories = []
        for traj in trajectories:
            traj = traj.to(self.device)
            traj_rate = self.get_gene_expression(traj)
            mapped_trajectories.append(traj_rate[None,:,:])
        gex = torch.vstack(mapped_trajectories)
        return gex
    
    def extract_genes(self, gex, genes):
        gex_numpy = gex.detach().cpu().numpy()
        valid_genes = set(genes).intersection(self.adata.var_names)
        gene_idx = [np.where(self.adata.var_names==gene)[0][0] for gene in valid_genes]
        return gex_numpy[:,:,gene_idx].squeeze()


class SimulationMixin:
    def simulate(
        self,
        initial_cells: Union[AnnData, csr_matrix, ndarray, ArrayView],
        n_samples_per_cell: int = 1,
        n_steps: int = 200,
        t_max: float = 50,
        t_start: float = 0,
        dt: float = 0.1,
        latent_key: str = "X_z",
        store_data: bool = False,
        return_data: bool = True,
        n_chunks: int = 100,
    ) -> Optional[Tuple[torch.Tensor, ndarray]]:
        """
        Simulate the SDE model on the initial cells data.

        Parameters
        ----------
        initial_cells : AnnData, csr_matrix, ndarray, or ArrayView
            The initial state of the cells to simulate.
        n_samples_per_cell : int, optional
            The number of samples per cell. Default is 1.
        n_steps : int, optional
            The number of simulation steps. Default is 200.
        t_max : float, optional
            The maximum time for the simulation. Default is 50.
        t_start : float, optional
            The start time for the simulation. Default is 0.
        dt : float, optional
            The time step size. Default is 0.1.
        latent_key : str, optional
            The key to access the latent data in the initial cells AnnData object. Default is 'X_z'.
        store_data : bool, optional
            Whether to store the trajectories and cell IDs in the instance. Default is False.
        return_data : bool, optional
            Whether to return the trajectories and cell IDs. Default is True.
        n_chunks : int, optional
            The number of chunks to divide the initial samples into for simulation. Default is 100.

        Returns
        -------
        Optional[Tuple[torch.Tensor, ndarray]]
            A tuple containing the trajectories and cell IDs if return_data is True, otherwise no return value.
        """
        if isinstance(initial_cells, AnnData):
            initial_cells = initial_cells.obsm[latent_key]
        if isinstance(initial_cells, csr_matrix):
            initial_cells = initial_cells.A
        if (isinstance(initial_cells, ndarray)) or (isinstance(initial_cells, ArrayView)):
            initial_cells = torch.tensor(initial_cells, device=self.device)

        initial_samples = torch.repeat_interleave(initial_cells, n_samples_per_cell, dim=0)
        sample_cell_ids = np.array([[i] * n_samples_per_cell for i in range(len(initial_cells))]).flatten()

        timespan = torch.linspace(t_start, t_max, n_steps)
        timespan.to(self.device)
        self.module.sde.to(self.device)
        initial_samples.to(self.device)

        trajectories = [None] * n_chunks
        for i, chunk in enumerate(pbar := tqdm(initial_samples.chunk(n_chunks))):
            pbar.set_description(f"Simulating Chunk {i}")
            traj = torchsde.sdeint_adjoint(self.module.sde, chunk, timespan, method="midpoint", dt=dt)
            trajectories[i] = torch.permute(traj, [1, 0, 2])
        trajectories = torch.vstack(trajectories)

        if store_data:
            self.trajectories = trajectories
            self.cell_ids = sample_cell_ids
        if return_data:
            return trajectories, sample_cell_ids

    def dtw_cluster_trajectories(
        self,
        n_clusters: int,
        trajectories: Union[ndarray, None] = None,
        metric: str = "dtw",
        max_iter: int = 10,
        n_jobs: int = 1,
        verbose: bool = False,
        return_model: bool = False,
    ) -> Union[ndarray, Tuple[ndarray, TimeSeriesKMeans]]:
        """
        Cluster time series trajectories using the k-means algorithm.

        Parameters:
        n_clusters (int): The number of clusters to form.
        trajectories (numpy.ndarray, optional): The trajectories to be clustered. If not provided, uses self.trajectories.
        metric (str, optional): The metric used for clustering. Default is 'dtw' (Dynamic Time Warping).
        max_iter (int, optional): Maximum number of iterations of the k-means algorithm for a single run. Default is 10.
        n_jobs (int, optional): The number of jobs to run in parallel. Default is 1.
        verbose (bool, optional): Whether to be verbose. Default is False.
        return_model (bool, optional): Whether to return the clustering model. Default is False.

        Returns:
        numpy.ndarray: Cluster labels for each trajectory.
        TimeSeriesKMeans (optional): The fitted TimeSeriesKMeans model.
        """
        traj_data = to_time_series_dataset(trajectories)
        ts_kmc = TimeSeriesKMeans(
            n_clusters=n_clusters, metric=metric, max_iter=max_iter, n_jobs=n_jobs, verbose=verbose
        )
        ts_kmc.fit(traj_data)
        predictions = ts_kmc.labels_
        if return_model:
            return predictions, ts_kmc
        else:
            return predictions


class ModifiedTrainingPlan(TrainingPlan):
    """
    A TrainingPlan class modified to pass additional attributes to the module during training.

    This class modifies the training_step method to set current_epoch and global_step on the module.

    Attributes:
    module (LightningModule): The module being trained.
    plan_kwargs (dict): Additional keyword arguments for the TrainingPlan.
    """

    def __init__(self, module, **plan_kwargs):
        super().__init__(module, **plan_kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        # the modification:
        self.module.current_epoch = self.current_epoch
        self.module.global_step = self.global_step

        # as before:
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})

        _, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        self.log("train_loss", scvi_loss.loss, on_epoch=True)
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
        return scvi_loss.loss


class ModifiedUnsupervisedTrainingMixin:
    """
    General purpose unsupervised train method.
    Modified to return epoch number during training.
    """

    def train(
        self,
        max_epochs: Optional[int] = 1000,
        freeze_vae_after_epochs: int = 200,
        constrain_vf_after_epochs: int = 200,
        lr: float = 0.001,
        weight_decay: float = 1e-3,
        optimizer: str = "AdamW",
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 1.0,
        validation_size: Optional[float] = None,
        batch_size: Optional[float] = None,
        early_stopping: bool = False,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """
        Train the model.
        MODIFIED: accepts and passes onto module vectorfield warmup args +
        monkey patches the training plan's training step to share epoch with module

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, sets to
            `np.min([round((20000 / n_cells) * 400), 400])`, though default is 1000
        freeze_vae_after_epochs
            Number of epochs after which, we should freeze the VAE and train only the VF.
        constrain_vf_after_epochs
            Number of epochs after which neighborhood constraint of VF will be introduced.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        self.module.max_epochs = max_epochs
        self.module.freeze_vae_after_epochs = freeze_vae_after_epochs
        self.module.constrain_vf_after_epochs = constrain_vf_after_epochs

        n_cells = self.adata.n_obs
        if batch_size is None:
            batch_size = n_cells

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()
        plan_kwargs["lr"] = lr
        plan_kwargs["weight_decay"] = weight_decay
        plan_kwargs["optimizer"] = optimizer

        data_splitter = DataSplitter(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = ModifiedTrainingPlan(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        return runner()


class SDETrainingMixin:
    """Unsupervised training method specifically for VelvetSDE."""

    def train(
        self,
        max_epochs: Optional[int] = 1000,
        n_trajectories: Optional[int] = 100,
        n_simulations: Optional[int] = 20,
        n_steps: Optional[int] = 50,
        n_markov_steps: Optional[int] = 10,
        t_max: Optional[float] = 10,
        dt: Optional[float] = 0.5,
        beginning_pct: Optional[float] = None,
        lr: float = 0.001,
        weight_decay: float = 1e-3,
        optimizer: str = "AdamW",
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 1.0,
        validation_size: Optional[float] = None,
        batch_size: Optional[float] = None,
        early_stopping: bool = False,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        if batch_size is None:
            batch_size = self.adata.shape[0]
        self.module.max_epochs = max_epochs

        # dynamical params defined at training to allow training regimes
        self.module.n_trajectories = n_trajectories
        self.module.n_simulations = n_simulations
        self.module.n_steps = n_steps
        self.module.n_markov_steps = n_markov_steps
        self.module.t_max = t_max
        self.module.beginning_pct = beginning_pct
        self.module.dt = dt

        n_cells = self.adata.n_obs
        if max_epochs is None:
            max_epochs = 100
        if batch_size is None:
            batch_size = n_cells

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()
        plan_kwargs["lr"] = lr
        plan_kwargs["weight_decay"] = weight_decay
        plan_kwargs["optimizer"] = optimizer

        data_splitter = DataSplitter(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = ModifiedTrainingPlan(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        return runner()
