from typing import Optional, Union, List, Tuple
import sys
import numpy as np

import scanpy as sc
import scvelo as scv

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, issparse
from scvelo.preprocessing.neighbors import set_diagonal, get_csr_from_indices

from anndata import AnnData
import scanpy as sc


def select_genes(
    data: AnnData,
    n_variable_genes: Optional[int] = 2000,
    curated_list: Optional[List[str]] = [],
    unwanted_list: Optional[List[str]] = [],
    stratify_obs: Optional[str] = None,
    normalize: Optional[bool] = False,
) -> List[str]:
    """
        Select HV genes from an AnnData object.
        uses scanpy's HVG selection at its core.
        Stratifying by replicate can remedy batch effects without manipulating the data.
        takes a copy of data and returns the selected genes, to allow for flexibility
        with normalisation etc.
    asser
        Parameters
        ----------
        data : anndata.AnnData
            Annotated data matrix.
        n_variable_genes : int, optional
            Number of variable genes to select, by default 2000.
        curated_list : list of str, optional
            List of genes to always include in the selection, by default [].
        unwanted_list : list of str, optional
            List of genes to always exclude from the selection, by default [].
        stratify_obs : str, optional
            Observation variable to stratify by, by default None.
        normalise : bool, optional
            Whether to calculate highly variable genes using normalized, logged data
            or raw data, by default False.

        Returns
        -------
        list of str
            List of selected genes.
    """
    if normalize:

        def func(data, n):
            copy = data.copy()
            sc.pp.normalize_total(copy, target_sum=1e4)
            sc.pp.log1p(copy)
            sc.pp.highly_variable_genes(copy, n_top_genes=n, subset=True)
            return copy.var_names

    else:

        def func(data, n):
            copy = data.copy()
            sc.pp.highly_variable_genes(copy, n_top_genes=n, flavor="seurat_v3", subset=True)
            return copy.var_names

    if stratify_obs:
        gene_sets = []
        for obs in data.obs[stratify_obs].unique():
            sub = data[data.obs[stratify_obs] == obs]
            tp_genes = func(sub, n_variable_genes)
            gene_sets.append(tp_genes)
        genes = set(gene_sets[0]).intersection(*gene_sets[1:])
    else:
        genes = set(func(data, n_variable_genes))

    if curated_list:
        genes = genes.union(curated_list)
    if unwanted_list:
        genes = genes.difference(unwanted_list)
    genes = genes.intersection(data.var_names)

    return list(genes)


def size_normalize(
    adata0: AnnData,
    genes: Optional[List[str]] = None,
    total_layer: Optional[str] = "total",
    new_layer: Optional[str] = "new",
    unsparsify: Optional[bool] = True,
) -> AnnData:
    """
    Size normalize layers of an AnnData object.
    To handle two data layers, these are normalised independently,
    and the 'total' is recalculated from normalised constituents
    Normalising total & new creates issues as they are not independent.

    Parameters
    ----------
    adata0 : anndata.AnnData
        Annotated data matrix.
    genes : list of str, optional
        A list of genes to be selected from the dataset, by default None.
    total_layer : str, optional
        Specifies the 'total' layer in the adata.layers attribute, by default 'total'.
    new_layer : str, optional
        Specifies the 'new' layer in the adata.layers attribute, by default 'new'.
    unsparsify : bool, optional
        Whether to convert the 'total' and 'new' layers to dense format if they are in
        sparse format, by default True.

    Returns
    -------
    anndata.AnnData
        A new AnnData object with size normalized layers.
    """
    adata = adata0.copy()

    adata.layers["old"] = adata.layers[total_layer] - adata.layers[new_layer]
    sc.pp.normalize_total(adata, layer=new_layer, target_sum=None)
    sc.pp.normalize_total(adata, layer="old", target_sum=None)
    adata.layers[total_layer] = adata.layers["old"] + adata.layers[new_layer]

    if genes is not None:
        adata = adata[:, genes]

    if unsparsify:
        adata.layers[total_layer] = (
            adata.layers[total_layer].A if issparse(adata.layers[total_layer]) else adata.layers[total_layer]
        )
        adata.layers[new_layer] = (
            adata.layers[new_layer].A if issparse(adata.layers[new_layer]) else adata.layers[new_layer]
        )
        adata.layers["old"] = (
            adata.layers["old"].A if issparse(adata.layers["old"]) else adata.layers["old"]
        )
    return adata


def read(file):
    """
    simple wrapper for scanpys read function
    """
    return sc.read(file)


def moments(
    X: Union[np.ndarray, csr_matrix],
    connectivities: Union[np.ndarray, csr_matrix],
    rescale: Optional[bool] = True,
    n_neighbors: Optional[int] = 30,
) -> np.ndarray:
    """
    Create smoothened data using scVelo.

    Parameters:
    - X (np.ndarray or csr_matrix): raw data to be processed.
    - connectivities (np.ndarray or csr_matrix): representing the connectivity of the data.
    - rescale (bool, optional): whether to rescale the data or not, defaults to True.
    - n_neighbors (int, optional): the number of neighbors to consider for the smoothing operation, defaults to 30.

    Returns:
    - Xs (np.ndarray): the smoothened data.
    """
    Xs = csr_matrix.dot(connectivities, csr_matrix(X)).astype(np.float32).A
    if rescale:
        Xs = csr_matrix(Xs.A / n_neighbors)
    return Xs


def neighbors(
    adata: AnnData,
    total_layer: Optional[str] = "total",
    n_neighbors: Optional[int] = 30,
    include_self: Optional[bool] = False,
) -> None:
    """
    A lightweight wrapper for scvelo-style neighbour calculations, avoiding the umap/fuzzy
    simplicial set calculations and going for the simplest implementation.
    Includes formatting to be compatible with downstream scVelo tools.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    total_layer : str, optional
        Specifies the 'total' layer in the adata.layers attribute, by default 'total'.
    n_neighbors : int, optional
        The size of local neighborhood (in terms of number of neighboring data points) used
        for manifold approximation, by default 30.
    include_self : bool, optional
        Whether or not to include an edge to the data point itself, by default False.

    Returns
    -------
    None
        This function operates in place on `adata` and does not return a value.
    """
    adata.uns["neighbors"] = dict(
        distances=get_distances_csr(
            adata,
            total_layer=total_layer,
            n_neighbors=n_neighbors,
            include_self=include_self,
        ),
        connectivities=connectivities(
            total=adata.layers[total_layer],
            n_neighbors=n_neighbors,
            zero_diagonal=(not include_self),
        ),
        params={"n_neighbors": n_neighbors, "n_pcs": None},
    )


def connectivities(
    adata: Optional[AnnData] = None,
    total: Optional[Union[np.ndarray, csr_matrix]] = None,
    n_neighbors: Optional[int] = 30,
    zero_diagonal: Optional[bool] = True,
) -> csr_matrix:
    """
    Compute binarised connectivities for data smoothening.

    Parameters
    ----------
    adata : anndata.AnnData, optional
        Annotated data matrix.
    total : np.ndarray or csr_matrix, optional
        A 'total' matrix. If not provided, 'total' layer of adata will be used.
    n_neighbors : int, optional
        Number of neighbors for each sample. By default 30.
    zero_diagonal : bool, optional
        Whether to zero out the diagonal of the connectivity matrix. By default True.

    Returns
    -------
    csr_matrix
        A compressed sparse row matrix representing connectivities.
    """
    if adata is None and total is None:
        print("Supply one of adata or total")
        return
    if total is None:
        total = adata.layers["total"]
        total = total.A if issparse(total) else total

    c, g = total.shape

    neighbors = NearestNeighbors(n_neighbors=n_neighbors - 1, metric="euclidean")
    neighbors.fit(total)

    knn_distances, knn_indices = neighbors.kneighbors()
    knn_distances, knn_indices = set_diagonal(knn_distances, knn_indices)

    connectivities = np.zeros((c, c))
    for i in range(c):
        connectivities[i][knn_indices[i]] = 1

    if zero_diagonal:
        np.fill_diagonal(connectivities, 0)

    connectivities = csr_matrix(connectivities)

    return connectivities.tocsr().astype(np.float32)


def get_knn_distances_and_indices(
    adata: AnnData,
    total_layer: Optional[str] = "total",
    n_neighbors: Optional[int] = 30,
    include_self: Optional[bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the k-nearest neighbors for the data in the specified layer of the AnnData object,
    and return the corresponding distances and indices.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    total_layer : str, optional
        Specifies the 'total' layer in the adata.layers attribute, by default 'total'.
    n_neighbors : int, optional
        The size of local neighborhood (in terms of number of neighboring data points) used
        for manifold approximation, by default 30.
    include_self : bool, optional
        Whether or not to include an edge to the data point itself, by default False.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two 2D arrays:
        - The first array contains the distances to the nearest neighbors for each data point.
        - The second array contains the indices of the nearest neighbors for each data point.
    """
    X = adata.layers[total_layer]
    neighbors = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
    )
    neighbors.fit(X)

    knn_distances, knn_indices = neighbors.kneighbors()
    knn_distances, knn_indices = set_diagonal(knn_distances, knn_indices)

    if include_self:
        return knn_distances, knn_indices
    else:
        return knn_distances[:, 1:], knn_indices[:, 1:]


def get_distances_csr(
    adata: AnnData,
    total_layer: Optional[str] = "total",
    n_neighbors: Optional[int] = 30,
    include_self: Optional[bool] = False,
) -> csr_matrix:
    """
    Calculate the Compressed Sparse Row (CSR) distances matrix for the given data.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    total_layer : str, optional
        Specifies the 'total transcriptome' layer in the adata.layers attribute, by default 'total'.
    n_neighbors : int, optional
        The size of local neighborhood (in terms of number of neighboring data points) used
        for manifold approximation, by default 30.
    include_self : bool, optional
        Whether or not to include an edge to the data point itself, by default False.

    Returns
    -------
    csr_matrix
        The CSR distances matrix for the given data.
    """
    distances, indices = get_knn_distances_and_indices(
        adata, total_layer=total_layer, n_neighbors=n_neighbors, include_self=include_self
    )
    dist = get_csr_from_indices(indices, distances, adata.shape[0], n_neighbors)
    return dist
