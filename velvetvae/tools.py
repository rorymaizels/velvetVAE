"""tools"""
from typing import Tuple
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.utils.data as data_utils
from sklearn.decomposition import PCA

from velvetvae.spline import natural_cubic_spline_coeffs, NaturalCubicSpline
from velvetvae.submodule import ObsNet, ManifoldEstimator

class Cleaner:
    def __init__(
        self,
        model,
        trajectories,
        cell_ids,
        n_neighbors: int,
        latent_cells=None,
        latent_key: str = "X_z",
        n_pca_components: int = 3,
    ):
        """
        Initialize a Cleaner instance.

        Args:
            model (Any): Model instance.
            trajectories (Any): Trajectories data.
            cell_ids (Any): Identifiers for the cells.
            n_neighbors (int): Number of neighbors for the ManifoldEstimator.
            latent_cells (Any, optional): Latent cells data. If None, it is extracted from the model. Default is None.
            latent_key (str, optional): Key for latent data in the model's adata. Default is 'X_z'.
            n_jobs (int, optional): Number of jobs to run in parallel. Default is 1.
            n_pca_components (int, optional): Number of components for PCA transformation. Default is 3.
        """
        self.device = model.device

        if latent_cells is None:
            if model is None:
                print("Provide one of model or latent_cells.")
                return
            latent_cells = model.adata.obsm[latent_key]

        if n_pca_components is not None:
            self.pca = PCA(n_components=n_pca_components)
            self.cells_rep = torch.tensor(self.pca.fit_transform(latent_cells), device=trajectories.device)

            self.trajectories_rep = torch.tensor(
                np.array([self.pca.transform(t.detach().cpu().numpy()) for t in trajectories]),
                device=trajectories.device,
            )

        else:
            self.cells_rep = torch.tensor(latent_cells, device=trajectories.device)
            self.trajectories_rep = trajectories.clone()

        self.trajectories = trajectories
        self.cell_ids = cell_ids
        self.estimator = ManifoldEstimator(self.cells_rep, n_neighbors)
        self.manifold_index = self.estimator(self.trajectories_rep)

    def clip(
        self,
        interpolate: bool = True,
        update_manifold_index: bool = True,
    ):
        """
        Clip trajectories.

        Args:
            interpolate (bool, optional): Whether to interpolate the trajectories. Default is True.
            update_manifold_index (bool, optional): Whether to update the manifold index. Default is True.

        Returns:
            Any: Clipped trajectories.
        """
        n_traj, n_steps, n_latent = self.trajectories.shape
        m = self.manifold_index
        clip_index = (m.cumsum(1) - m.cumsum(1).max(1, keepdims=True).values) - m * 1.0 < 0

        clipped_trajectories = []
        for trajectory, index in (pbar := tqdm(zip(self.trajectories, clip_index), total=len(clip_index))):
            pbar.set_description("Clipping trajectories...")
            trajectory = trajectory[index]
            if interpolate:
                if len(trajectory) < n_steps:
                    coeffs = natural_cubic_spline_coeffs(
                        torch.linspace(0, 1, len(trajectory), device=self.device), trajectory.to(self.device)
                    )
                    splines = NaturalCubicSpline(coeffs)
                    spline_time = torch.linspace(0, 1, n_steps, device=self.device)
                    trajectory = splines.evaluate(spline_time).reshape(1, n_steps, n_latent)
                else:
                    trajectory = trajectory.reshape(1, n_steps, n_latent)
            clipped_trajectories.append(trajectory.to(self.device))

        if interpolate:
            clipped_trajectories = torch.vstack(clipped_trajectories)

        if update_manifold_index:
            self.update_manifold(clipped_trajectories)
        print(f"{(clip_index[:,-1]==0).cpu().numpy().sum()} trajectories clipped.")
        return clipped_trajectories

    def prune(self, threshold: float = 0.1, update_manifold_index: bool = True):
        """
        Prune trajectories (remove trajectories that are less than threshold in-manifold)

        Args:
            threshold (float, optional): Mean in-manifold membership threshold. Default is 0.1.
            update_manifold_index (bool, optional): Whether to update the manifold index. Default is True.

        Returns:
            Tuple[Any, Any]: Pruned trajectories and corresponding cell identifiers.
        """
        n_start = self.trajectories.shape[0]
        # mean in-manifold membership above threshold
        index = (1.0 * self.manifold_index).mean(1) > (1 - threshold)
        trajectories = self.trajectories[index]
        cell_ids = self.cell_ids[index]
        n_new = trajectories.shape[0]
        if update_manifold_index:
            self.cell_ids = cell_ids
            self.trajectories = trajectories
            self.trajectories_rep = self.trajectories_rep[index]
            self.manifold_index = self.manifold_index[index]
        print(f"{n_start - n_new} trajectories removed from {n_start} initial total.")
        return trajectories, cell_ids

    def update_manifold(self, trajectories):
        """
        Update manifold to reflect new clipped/pruned data.

        Args:
            trajectories (Any): The new trajectories data.
        """
        self.manifold_index_accurate = True
        self.trajectories = trajectories
        self.trajectories_rep = torch.tensor(
            np.array([self.pca.transform(t.detach().cpu().numpy()) for t in trajectories]), device=trajectories.device
        )
        self.manifold_index = self.estimator(self.trajectories_rep)


def train_obsnet(
    model,
    obs_key: str,
    latent_key: str = "X_z",
    batch_size: int = 256,
    train_proportion: float = 0.9,
    epochs: int = 500,
):
    """
    Build and train a predictive model for a categorical feature based on latent space position.

    Args:
        model (Any): The model to be trained.
        obs_key (str): The key for the observation in the model's adata.
        latent_key (str, optional): Key for the latent data in the model's adata. Default is 'X_z'.
        batch_size (int, optional): The number of samples per batch. Default is 256.
        train_proportion (float, optional): The proportion of data to use for training. Default is 0.9.
        epochs (int, optional): The number of training epochs. Default is 500.

    Returns:
        Any: The trained predictive model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.tensor(model.adata.obsm[latent_key], device=device)
    y = torch.tensor(model.adata.obs[obs_key], device=device).reshape(-1, 1)

    net = ObsNet(n_latent=model.module.n_latent)
    net.to(device)
    X = X.to(device)
    y = y.to(device)

    train_data, val_data = _get_obsnet_dataloaders(X, y, batch_size, train_proportion)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    for epoch in (pbar := trange(epochs)):
        running_loss = 0
        n_steps = 0
        for i, data in enumerate(train_data, 0):
            X, y = data

            optimizer.zero_grad()
            y_hat = net(X)
            loss = criterion(y_hat, y)

            running_loss += loss
            n_steps += 1

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            running_loss = 0
            n_steps = 0
            for data in val_data:
                X, y = data

                optimizer.zero_grad()
                y_hat = net(X)
                running_loss += criterion(y_hat, y)
                n_steps += 1
        epoch_loss = running_loss / n_steps
        pbar.set_description(f"Validation Loss: {epoch_loss:.4f}")
    return net


def _get_obsnet_dataloaders(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 10000,
    train_proportion: float = 0.8,
) -> Tuple[data_utils.DataLoader, data_utils.DataLoader]:
    """
    Get the DataLoader objects for training and validation data.

    Args:
        model (Any): The model for which the data loaders are being prepared.
        X (torch.Tensor): The input features.
        y (torch.Tensor): The target labels.
        batch_size (int, optional): The number of samples per batch. Default is 10000.
        train_proportion (float, optional): The proportion of data to use for training. Default is 0.8.

    Returns:
        Tuple[data_utils.DataLoader, data_utils.DataLoader]: The DataLoader objects for the training and validation datasets.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X.to(device)
    y.to(device)

    dataset = data_utils.TensorDataset(X, y)
    train_size = int(train_proportion * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = data_utils.random_split(dataset, [train_size, test_size])

    train_dataloader = data_utils.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    val_dataloader = data_utils.DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
    return train_dataloader, val_dataloader
