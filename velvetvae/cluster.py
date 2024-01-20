"""
clustering methods, KMeansLayer implemented from:
https://github.com/birkhoffkiki/clustering-pytorch/blob/main/tools.py
"""
import random
import torch
from torch import nn

from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans


class Similarity:
    supported_distance = ["euclidean", "cosine"]

    def __init__(self, distance: str, save_memory=True):
        """
        :param distance: the method of measuring two vectors, support one of these `euclidean`, `cosine`
        :param save_memory: use for loop method to save memory, but it may be slow in some situations.
        """
        if distance not in self.supported_distance:
            raise NotImplementedError(
                "The distance `{}` is not supported yet, please implement it manually ...".format(distance)
            )
        func = {"euclidean": self.__euclidean, "cosine": self.__cosine}
        self.distance = func[distance]
        self.save_memory = save_memory

    def __call__(self, x, y, mode):
        """
        compute the distance between x and y. if set mode = 0, return len(y) indexes of x that are most similar to y.
        if set mode =1, return len(x) indexes of y that are most similar to x.
        :param x: whole dataset, the shape likes [samples, dimensions]
        :param y: center vectors, the shape likes [center_samples, dimensions]
        :param mode: set 0 to get x's index, set 1 to get y's index
        :return:
        """
        if self.save_memory:
            return self.__cal(x, y, self.distance, mode)
        else:
            x = x[:, None]
            y = y[None]
            return self.distance(x, y, dim=2).argmin(mode)

    @staticmethod
    def __cal(x, y, distance_fn, mode):
        if mode == 1:
            r = torch.randperm(len(x))
            for index, d in enumerate(x):
                m = distance_fn(d, y, dim=1).argmin(0)
                r[index] = m
        elif mode == 0:
            r = torch.randperm(len(y))
            for index, d in enumerate(y):
                m = distance_fn(d, x, dim=1).argmin(0)
                r[index] = m
        return r

    @staticmethod
    def __euclidean(x, y, dim):
        x = x.sub(y)
        return x.square_().sum(dim)

    @staticmethod
    def __cosine(x, y, dim):
        return torch.cosine_similarity(x, y, dim=dim)


class Distance:
    supported_distance = ["euclidean", "cosine"]

    def __init__(self, distance):
        if distance not in self.supported_distance:
            raise NotImplementedError(
                "The distance `{}` is not supported yet, please implement it manually ...".format(distance)
            )
        self.distance = {"euclidean": self.__euclidean, "cosine": self.__cosine}[distance]

    def __call__(self, x, y, dim):
        return self.distance(x, y, dim)

    @staticmethod
    def __euclidean(x, y, dim):
        x = x.sub(y)
        return torch.sqrt(x.square_().sum(dim))

    @staticmethod
    def __cosine(x, y, dim):
        return torch.cosine_similarity(x, y, dim=dim)


class KMeansLayer(nn.Module):
    def __init__(self, centers, iteration, distance="euclidean", save_memory=True):
        super(KMeansLayer, self).__init__()
        self.distance = distance
        self.similarity = Similarity(distance, save_memory)
        self.clusters = centers
        self.iteration = iteration

    def forward(self, data):
        avg_center, classes, centers, index = self.kmeans(data, self.clusters, self.iteration)
        return avg_center, classes, centers, index

    def kmeans(self, data: torch.Tensor, clusters: int, iteration: int):
        """
        :param data: [samples, dimension]
        :param clusters: the number of centers
        :param iteration: total iteration time
        :return: [average_center, class_map, center, index]
        """
        with torch.no_grad():
            N, D = data.shape
            c = data[torch.randperm(N)[:clusters]]
            for i in range(iteration):
                a = self.similarity(data, c, mode=1)
                c = torch.stack([data[a == k].mean(0) for k in range(clusters)])
                nanix = torch.any(torch.isnan(c), dim=1)
                ndead = nanix.sum().item()
                c[nanix] = data[torch.randperm(N)[:ndead]]
            # get centers (not average centers) and index
            index = self.similarity(data, c, mode=0)
            center = data[index]
            avg_center = c
        return avg_center, a, center, index


def cluster_trajectories(trajectories, n_clusters, n_iterations=100, final_steps=None, return_all=True):
    """
    Cluster trajectories using KMeansLayer.

    Args:
        trajectories: Input trajectories to be clustered.
        n_clusters: Number of clusters to form.
        n_iterations (int, optional): Number of iterations for KMeansLayer. Defaults to 100.
        final_steps: If provided, only the specified final steps of the trajectories are considered. Defaults to None.
        return_all (bool, optional): If True, returns average center, classes, centers, and index. If False, only returns classes. Defaults to True.

    Returns:
        If return_all is True, returns a tuple of (average center, classes, centers, index).
        If return_all is False, only returns classes.
    """
    if final_steps is not None:
        trajectories = trajectories[:, final_steps:, :]
    kmc = KMeansLayer(centers=n_clusters, iteration=n_iterations).to(trajectories.device)
    t_reshaped = trajectories.reshape(trajectories.shape[0], -1)
    avg_center, classes, centers, index = kmc(t_reshaped)
    if return_all:
        return avg_center, classes, centers, index
    else:
        return classes


def dtw_cluster_trajectories(
    trajectories,
    n_clusters,
    metric="dtw",
    n_iterations=10,
    n_jobs=1,
    verbose=False,
    final_steps=None,
    return_model=False,
):
    """
    Cluster trajectories using TimeSeriesKMeans.

    Args:
        trajectories: Input trajectories to be clustered.
        n_clusters: Number of clusters to form.
        metric (str, optional): Metric to use for the KMeans clustering. Defaults to 'dtw'.
        n_iterations (int, optional): Number of iterations for TimeSeriesKMeans. Defaults to 10.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to 1.
        verbose (bool, optional): Whether or not to print progress messages to stdout. Defaults to False.
        final_steps: If provided, only the specified final steps of the trajectories are considered. Defaults to None.
        return_model (bool, optional): If True, returns the prediction and the model. If False, only returns predictions. Defaults to False.

    Returns:
        If return_model is True, returns a tuple of (predictions, model).
        If return_model is False, only returns predictions.
    """
    if final_steps is not None:
        trajectories = trajectories[:, -final_steps:, :]
    traj_data = to_time_series_dataset(trajectories.detach().cpu().numpy())
    ts_kmc = TimeSeriesKMeans(
        n_clusters=n_clusters, metric=metric, max_iter=n_iterations, n_jobs=n_jobs, verbose=verbose
    )
    ts_kmc.fit(traj_data)
    predictions = torch.tensor(ts_kmc.labels_, device=trajectories.device)
    if return_model:
        return predictions, ts_kmc
    else:
        return predictions
