"""components used for Velvet module and VelvetSDE"""
from velvet.preprocessing import neighborhood

from typing import Optional, List

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scanpy as sc

import torch
from torch import nn
from scvi.module.base import auto_move_data
import torchsde
import anndata as ann

from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
from tqdm import tqdm

from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

from torch import nn, Tensor
import torch


class VectorField(nn.Module):
    """
    This class implements a VectorField model which is a PyTorch Module.
    It consists of multiple layers of Linear transformations and ReLU activation functions, ending with a final Linear layer.

    Attributes
    ----------
    latent_dim : int
        The dimensionality of the latent space.
    n_layers : int, optional
        The number of layers in the neural network.
    n_hidden : int, optional
        The number of hidden units in each layer.
    torch_device : torch.device
        The device to which tensors will be moved.
    drift : torch.nn.Sequential
        The sequential model representing the main body of the neural network.

    Methods
    -------
    forward(z: Tensor) -> Tensor:
        Performs the forward pass of the VectorField.
    """

    def __init__(
        self,
        latent_dim: int,
        n_layers: int = 3,
        n_hidden: int = 128,
    ) -> None:
        super(VectorField, self).__init__()

        self.latent_dim = latent_dim
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        drift_list = []
        drift_list.extend([nn.Linear(latent_dim, n_hidden), nn.ReLU()])
        for k in range(n_layers - 1):
            drift_list.extend([nn.Linear(n_hidden, n_hidden), nn.ReLU()])
        drift_list.append(nn.Linear(n_hidden, latent_dim))

        self.drift = nn.Sequential(*drift_list)

        for m in self.drift.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, z: Tensor) -> Tensor:
        v = self.drift(z)
        return v


class NeighborhoodConstraint(nn.Module):
    """object for manifold constraining of velocity vectors."""

    def __init__(self, X, batch_index=None, inverse_sigma: int = 10, similarity_strength: float = 0.2) -> None:
        """
        Initializes the object with total dataset.

        Parameters
        ----------
        X : np.array or torch.Tensor
            The input tensor.

        batch_index : torch.Tensor, optional
            The batch index tensor. If None, it is initialized to zeros.
        inverse_sigma : int, optional
            A parameter for scaling in the project method. Default is 10.
        similarity_strengh: float, optional
            If a similarity transition matrix is supplied, how much relative weight to give it

        """
        super(NeighborhoodConstraint, self).__init__()
        self.X = torch.tensor(X, device="cuda" if torch.cuda.is_available() else "cpu")
        if batch_index is None:
            self.b = torch.zeros(self.X.shape[0], device=self.X.device)
        else:
            self.b = batch_index
        self.cs = nn.CosineSimilarity(dim=2)
        self.inverse_sigma = inverse_sigma
        self.ss = similarity_strength

    @auto_move_data
    def project(
        self, x: Tensor, v: Tensor, k: Tensor, Ts: Optional[Tensor] = None, zfull: Optional[Tensor] = None
    ) -> Tensor:
        if zfull is not None:
            X = zfull
        else:
            X = self.X

        deltas = X[k.to(torch.long)] - x.unsqueeze(1)
        v_unsq = v.unsqueeze(1)
        cs_scores = self.cs(v_unsq, deltas)

        graph = cs_scores
        Tv = torch.expm1(graph * self.inverse_sigma)
        Tv = Tv / torch.abs(Tv).sum(1)[:, None]

        if Ts is not None:
            T = (self.ss * Ts) + ((1 - self.ss) * Tv)
        else:
            T = Tv

        dX = X[k.to(torch.long)] - x.unsqueeze(1)
        dX[torch.isnan(dX)] = 0

        subtractor = T.mean(dim=1, keepdims=True) * dX.sum(1)
        v_proj = torch.einsum("ncg, nc -> ng", dX, T) - subtractor

        return v_proj


class MarkovProcess:
    def __init__(
        self,
        model,
        n_neighbors: int = 30,
        use_space: str = "latent_space",
        use_spline: bool = False,
        terminal_state_key: Optional[str] = None,
        deterministic_scaling: float = 10.0,
        use_similarity: bool = True,
        similarity_strength: float = 0.2,
    ) -> None:
        """
        Initializes the Markov Process, setting up transition matrices and data objects.

        Args:
            model: A model object.
            n_neighbors: Number of neighbors for KNN.
            use_space: Type of space to be used, either 'latent_space' or 'gene_space'.
            use_spline: If true, applies spline to walks to increase number of steps.
            terminal_state_key: Key to the terminal state, None if not applicable.
            deterministic_scaling: The larger it is, the more deterministic the transition matrix will be.
            use_similarity: If true, uses similarity-based transition matrix alongside velocity.
            similarity_strength: weight of similarity T relative to velocity T.

        Returns:
            None.
        """
        self.adata = model.adata
        self.deterministic_scaling = deterministic_scaling
        self.device = model.device

        if use_space == "latent_space":
            subdata = ann.AnnData(
                X=self.adata.obsm["X_z"],
                obs=self.adata.obs,
                layers={
                    "total": self.adata.obsm["X_z"],
                    "total_for_neighbors": self.adata.obsm["X_z"],
                    "velocity": self.adata.obsm["velocity_z"],
                },
            )
        elif use_space == "gene_space":
            subdata = ann.AnnData(
                X=self.adata.layers["total"],
                obs=self.adata.obs,
                layers={
                    "total": self.adata.layers["total"],
                    "total_for_neighbors": np.log1p(self.adata.layers["total"]),
                    "velocity": self.adata.layers["velocity"],
                },
            )
        if terminal_state_key is not None:
            self.terminal_states = torch.tensor(np.where(self.adata.obs[terminal_state_key])[0], device=model.device)
            self.use_terminal_states = True
        else:
            self.use_terminal_states = False

        neighborhood(subdata, xkey="total_for_neighbors", n_neighbors=n_neighbors, symmetric=False, calculate_transition=False, verbose=False)
        self.T = self.velocity_transition_matrix(subdata)
        if use_similarity:
            Ts = self.similarity_transition_matrix(self.adata)
            self.T = (similarity_strength * Ts) + ((1 - similarity_strength) * self.T)

        self.use_spline = use_spline
        self.subdata = subdata

    def random_walk(
        self, z: torch.Tensor, initial_states: torch.Tensor, n_jumps: int, n_steps: int, deterministic: bool = False
    ) -> torch.Tensor:
        """
        Performs a random walk in the latent space.

        Args:
            z: The latent space states.
            initial_states: Initial states for the walk.
            n_jumps: Number of jumps in the walk (markov steps).
            n_steps: Number of steps in the walk (steps for spline interpolation).
            deterministic: If true, the walk is deterministic rather than random.

        Returns:
            Tensor containing the walks.
        """
        assert n_steps >= n_jumps, "step before you can jump"
        initial_states = initial_states.reshape(-1, 1)
        paths = initial_states.repeat(1, n_jumps)

        for i in range(n_jumps - 1):
            if deterministic:
                paths[:, i + 1] = self.T[paths[:, i]].argmax(1).flatten()
            else:
                paths[:, i + 1] = self.T[paths[:, i]].multinomial(1).flatten()

        if self.use_terminal_states:
            paths = self.apply_terminal_states(paths)

        walks = z[paths]

        if self.use_spline:
            walks = self.apply_spline(walks, n_jumps, n_steps)

        return walks.detach()

    def apply_terminal_states(self, paths: torch.Tensor) -> torch.Tensor:
        """
        Applies terminal states to the paths, freezing the trajectory when it reaches a terminal state.
        Implementation performs this rapidly through broadcasting.
        Args:
            paths: Paths to apply terminal states to.

        Returns:
            Paths with terminal states applied.
        """
        diff = paths.unsqueeze(2).repeat(1, 1, self.terminal_states.shape[0]) - self.terminal_states.reshape(1, 1, -1)
        B = (diff == 0).sum(2) == 1  # B = where path[i,j] is a terminal state
        B[:, -1] = True  # every last step should be considered a terminal state
        replace_idx = B.cumsum(1) > 0
        value_idx = B.cumsum(1).cumsum(1) == 1
        adjusted_values = paths[value_idx].reshape(-1, 1).repeat(1, paths.shape[1]) * replace_idx
        retained_values = paths * (~replace_idx)
        adjusted_paths = retained_values + adjusted_values
        return adjusted_paths

    def apply_spline(self, walks: torch.Tensor, n_jumps: int, n_steps: int) -> torch.Tensor:
        """
        Applies spline to walks.

        Args:
            walks: Walks to apply spline to.
            n_jumps: Number of jumps in the walk (markov jumps).
            n_steps: Number of steps in the walk (output steps after interpolation).

        Returns:
            Walks with spline applied.
        """
        coeffs = natural_cubic_spline_coeffs(torch.linspace(0, 1, n_jumps, device=self.device), walks)
        splines = NaturalCubicSpline(coeffs)
        spline_time = torch.linspace(0, 1, n_steps, device=walks.device)
        walks = splines.evaluate(spline_time)
        return walks

    def similarity_transition_matrix(self, adata: ann.AnnData) -> torch.Tensor:
        """
        Retrieves a similarity transition matrix.

        Args:
            subdata: adata to create similarity transition matrix from.

        Returns:
            Similarity transition matrix.
        """
        # this should be precomputed using the pp.neighborhood function
        Ts = adata.uns["neighbors"]["similarity_transition"].A
        return torch.tensor(Ts, device=self.device)

    def velocity_transition_matrix(self, subdata: ann.AnnData) -> torch.Tensor:
        """
        Creates a velocity transition matrix.

        Args:
            subdata: Subdata to create velocity transition matrix from.

        Returns:
            Velocity transition matrix.
        """
        x = torch.tensor(subdata.layers["total"])
        v = torch.tensor(subdata.layers["velocity"])
        k = torch.tensor(subdata.obsm["knn_index"])

        deltas = x[k.to(torch.long)] - x.unsqueeze(1)
        v_unsq = v.unsqueeze(1)

        graph = torch.nn.CosineSimilarity(dim=2)(v_unsq, deltas)

        T = torch.exp(graph * self.deterministic_scaling)
        T = T / torch.abs(T).sum(1)[:, None]

        T_expanded = torch.sparse_coo_tensor(
            indices=torch.vstack((torch.arange(k.shape[0]).repeat_interleave(k.shape[1]), k.flatten())),
            values=T.flatten(),
        ).to_dense()

        return T_expanded.detach().to(self.device)

    def plot(
        self,
        initial_indices,
        n_markov_steps: int,
        n_steps: int,
        z=None,
        cell_alpha: float = 0.2,
        cell_size: int = 200,
        line_alpha: float = 0.7,
        components: List[int] = [0, 1],
        color: Optional[str] = None,
    ) -> None:
        """
        Plots the random walk trajectories in PCA space.

        Args:
            initial_indices: Initial states for the walk.
            n_markov_steps: Number of Markov steps in the walk.
            n_steps: Number of steps in the walk.
            z: The latent space coordinates, if None uses total layer.
            cell_alpha: Transparency level of cells.
            cell_size: Size of cells.
            line_alpha: Transparency level of trajectory lines.
            components: Principal components to plot.
            color: Color key for cells.

        Returns:
            None.
        """
        if z is None:
            z = torch.tensor(self.subdata.layers["total"], device=self.T.device)

        trajectories = self.random_walk(
            z=z,
            initial_states=initial_indices,
            n_jumps=n_markov_steps,
            n_steps=n_steps,
        )

        pca = PCA()
        z_pca = pca.fit_transform(z.detach().cpu().numpy())

        fig, ax = plt.subplots(figsize=(12, 8))
        self.adata.obsm["X_viz"] = z_pca
        self.adata.uns["velocity_params"] = {"embeddings": "viz"}
        sc.pl.scatter(
            self.adata,
            basis="viz",
            color=color,
            title="",
            ax=ax,
            size=cell_size,
            alpha=cell_alpha,
            show=False,
            components=f"{components[0]+1},{components[1]+1}",
        )

        for t in trajectories:
            t_pca = pca.transform(t.detach().cpu().numpy())
            ax.plot(t_pca[:, components[0]], t_pca[:, components[1]], alpha=line_alpha, label="Markov")


class SDE(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        prior_vectorfield=None,
        noise_scalar: float = 0.1,
        n_layers: int = 3,
        n_hidden: int = 128,
        device: str = "cuda",
    ) -> None:
        """
        Initializes the SDE class.
        Assumes scalar Brownian noise, for now.

        Args:
            latent_dim: The latent dimension.
            prior_vectorfield: If provided, it is used to update SDE's weights.
            noise_scalar: Scalar for the noise.
            n_layers: Number of layers.
            n_hidden: Number of hidden units.
            device: Device to use, either 'cuda' or 'cpu'.

        Returns:
            None.
        """
        super(SDE, self).__init__()
        self.latent_dim = latent_dim
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        drift_list = []
        drift_list.extend([nn.Linear(latent_dim, n_hidden), nn.ReLU()])
        for _ in range(n_layers - 1):
            drift_list.extend([nn.Linear(n_hidden, n_hidden), nn.ReLU()])
        drift_list.append(nn.Linear(n_hidden, latent_dim))

        self.drift = nn.Sequential(*drift_list)

        for m in self.drift.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

        if prior_vectorfield is not None:
            self.load_state_dict(prior_vectorfield.state_dict())
        self.drift.to(device)

        self.noise_scalar = noise_scalar
        self.noise_type = "diagonal"
        self.sde_type = "stratonovich"

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Drift function for the SDE.

        Args:
            t: Time tensor.
            y: The tensor representing the state of the system.

        Returns:
            Drift of the system at state y and time t.
        """
        return self.drift(y)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Diffusion function for the SDE.

        Args:
            t: Time tensor.
            y: The tensor representing the state of the system.

        Returns:
            Scalar value for the diffusion of the system at state y and time t.
        """
        return torch.full_like(y, self.noise_scalar)

    def plot(
        self,
        n_steps: int,
        t_max: float,
        dt: float,
        initial_indices: Optional[torch.Tensor] = None,
        initial_cells: Optional[torch.Tensor] = None,
        z=None,
        cell_alpha: float = 0.2,
        cell_size: int = 200,
        line_alpha: float = 0.7,
        components: List[int] = [0, 1],
        color: Optional[str] = None,
    ) -> None:
        """
        Plots the SDE trajectories in PCA space.

        Args:
            n_steps: Number of steps.
            t_max: Maximum time.
            dt: Time step size.
            initial_indices: Initial state indices, if None the total layer is used.
            initial_cells: Initial state cells, if None initial indices are used.
            z: The latent space coordinates, if None uses total layer.
            cell_alpha: Transparency level of cells.
            cell_size: Size of cells.
            line_alpha: Transparency level of trajectory lines.
            components: Principal components to plot.
            color: Color of cells.

        Returns:
            None.
        """
        if z is None:
            z = torch.tensor(self.data.layers["total"], device=self.T.device)

        if initial_cells is None:
            initial_cells = z[initial_indices.squeeze(), :]

        timespan = torch.linspace(0, t_max, n_steps, device=z.device)

        self.eval()
        with torch.no_grad():
            trajectories = torchsde.sdeint_adjoint(self, initial_cells, timespan, method="midpoint", dt=dt).permute(
                (1, 0, 2)
            )
        self.train()

        pca = PCA()
        z_pca = pca.fit_transform(z.detach().cpu().numpy())

        fig, ax = plt.subplots(figsize=(12, 8))
        self.adata.obsm["X_viz"] = z_pca
        self.adata.uns["velocity_params"] = {"embeddings": "viz"}
        sc.pl.scatter(
            self.adata,
            basis="viz",
            color=color,
            title="",
            ax=ax,
            size=cell_size,
            alpha=cell_alpha,
            show=False,
            components=f"{components[0]+1},{components[1]+1}",
        )

        for t in trajectories:
            t_pca = pca.transform(t.detach().cpu().numpy())
            ax.plot(t_pca[:, components[0]], t_pca[:, components[1]], alpha=line_alpha)
        plt.show()


class ObsNet(nn.Module):
    def __init__(
        self,
        n_latent: int,
        n_layers: Optional[int] = 2,
        n_hidden: Optional[int] = 128,
    ) -> None:
        """
        Initializes the ObsNet class.

        Args:
            n_latent: The latent dimension size.
            n_layers: The number of layers in the network (default is 2).
            n_hidden: The number of hidden units in each layer (default is 128).

        Returns:
            None.
        """
        super().__init__()
        self.latent_dim = n_latent
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layer_list = []
        layer_list.extend([nn.Linear(n_latent, n_hidden), nn.ReLU()])
        for _ in range(n_layers - 1):
            layer_list.extend([nn.Linear(n_hidden, n_hidden), nn.ReLU()])
        layer_list.append(nn.Linear(n_hidden, 1))

        self.net = nn.Sequential(*layer_list)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor.

        Returns:
            Tensor after passing through the network.
        """
        return self.net(x)


class DelaunayEstimator:
    def __init__(self, cells: torch.Tensor, n_neighbors: int):
        """
        Initializes the DelaunayEstimator with the given cells and number of neighbors.

        Parameters:
        cells: torch.Tensor
            Tensor representing cells.
        n_neighbors: int
            The number of neighbors.
        """
        self.cells = cells
        self.n_neighbors = n_neighbors

    def __call__(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Method to estimate local neighbourhood inclusion points for a trajectory

        Parameters:
        trajectory: torch.Tensor
            The trajectory tensor.

        Returns:
        torch.Tensor:
            The tensor of manifold indices.
        """
        neighborhood = torch.argsort(
            torch.linalg.norm((self.cells[None, :, :] - trajectory[:, None, :]), dim=2), dim=1
        )[:, :15]
        neighbors = self.cells[neighborhood].detach()
        manifold_index = np.zeros(trajectory.shape[0])
        for i, (step, neigh) in enumerate(zip(trajectory, neighbors)):
            convex_hull = Delaunay(neigh.cpu().numpy())
            s = step.detach().cpu().numpy()
            manifold_index[i] = convex_hull.find_simplex(s) >= 0
        manifold_index = torch.tensor(manifold_index, device=trajectory.device)
        return manifold_index


class ManifoldEstimator:
    def __init__(self, cell_rep: torch.Tensor, n_neighbors: int):
        """
        For a set of trajectory representations (i.e. PCA of trajs), estimates which
        points exist in the data manifold. For clipping/pruning predictions.

        Parameters:
        cell_rep: torch.Tensor
            Tensor representing cells.
        n_neighbors: int
            The number of neighbors.
        """
        self.manifold = DelaunayEstimator(cell_rep, n_neighbors=n_neighbors)

    def __call__(self, trajectories_rep: torch.Tensor) -> torch.Tensor:
        """
        Call method for ManifoldEstimator.

        Parameters:
        trajectories_rep: torch.Tensor
            The trajectories tensor.
        n_jobs: int, optional
            The number of jobs to run in parallel. Default is 1.

        Returns:
        torch.Tensor:
            The tensor of manifold indices.
        """
        manifold_index = torch.zeros(trajectories_rep.shape[0], trajectories_rep.shape[1])
        for i, t in enumerate(pbar := tqdm(trajectories_rep)):
            pbar.set_description(f"Updating manifold...")
            manifold_index[i] = self.manifold(t)
        return manifold_index
