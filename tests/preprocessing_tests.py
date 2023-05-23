import pytest
import pandas as pd
import numpy as np

from anndata import Anndata
import anndata as ad
from scipy.sparse import csr_matrix, issparse
from scanpy import preprocessing as sc_pp
from sklearn.datasets import make_spd_matrix
from sklearn.neighbors import NearestNeighbors


from velvet.preprocessing import (
    moments, 
    size_normalize, read, select_genes, neighbors, connectivities, 
    get_knn_distances_and_indices, get_distances_csr
)

class test_read:
    def test_read_output_type(self):
        output = read(self.file_path)
        self.assertIsInstance(output, anndata.AnnData)

class test_moments:
    def test_moments_rescale_true(self):
        Xs = moments(self.X, self.connectivities, rescale=True, n_neighbors=5)
        self.assertIsInstance(Xs, np.ndarray)

    def test_moments_rescale_false(self):
        Xs = moments(self.X, self.connectivities, rescale=False, n_neighbors=5)
        self.assertIsInstance(Xs, np.ndarray)

    def test_moments_wrong_X_type(self):
        with self.assertRaises(TypeError):
            moments("invalid type", self.connectivities)

    def test_moments_wrong_connectivities_type(self):
        with self.assertRaises(TypeError):
            moments(self.X, "invalid type")

    def test_moments_wrong_rescale_type(self):
        with self.assertRaises(TypeError):
            moments(self.X, self.connectivities, rescale="invalid type")

    def test_moments_wrong_n_neighbors_type(self):
        with self.assertRaises(TypeError):
            moments(self.X, self.connectivities, n_neighbors="invalid type")

    def test_moments_rescale_consistency(self):
        Xs_rescale_true = moments(self.X, self.connectivities, rescale=True, n_neighbors=30)
        Xs_rescale_false = moments(self.X, self.connectivities, rescale=False, n_neighbors=30)
        np.testing.assert_allclose(Xs_rescale_true, Xs_rescale_false * 30, rtol=1e-5)

    def test_moments_output_shape(self):
        Xs = moments(self.X, self.connectivities)
        self.assertEqual(self.X.shape, Xs.shape)


class test_gene_selection:
    self.curated_list = np.random.choice(self.data.var_names, size=100, replace=False).tolist()
    self.unwanted_list = np.random.choice(self.data.var_names, size=100, replace=False).tolist()

    def test_genes_in_output_exist_in_input(self):
        selected_genes = select_genes(self.data)
        assert all(gene in self.data.var_names for gene in selected_genes), "Not all genes in the output list exist in the input data's data.var_names."

    def test_curated_genes_in_output(self):
        selected_genes = select_genes(self.data, curated_list=self.curated_list)
        assert all(gene in selected_genes for gene in self.curated_list), "Not all genes from curated list are present in the output."

    def test_unwanted_genes_not_in_output(self):
        selected_genes = select_genes(self.data, unwanted_list=self.unwanted_list)
        assert all(gene not in selected_genes for gene in self.unwanted_list), "Some unwanted genes are present in the output."

    def test_output_length_with_no_stratification(self):
        for normalize in [True, False]:
            selected_genes = select_genes(self.data, n_variable_genes=2000, stratify_obs=None, normalize=normalize)
            assert abs(len(selected_genes) - 2000) <= 1, f"The number of genes selected is more than one away from the specified number when normalise={normalise}."


class test_size_normalize:
    def test_old_new_sum_to_total(self):
        adata_normalized = size_normalize(self.adata)
        np.testing.assert_allclose(
            adata_normalized.layers["old"] + adata_normalized.layers["new"],
            adata_normalized.layers["total"],
            rtol=1e-5,
        )

    def test_old_and_new_less_than_total(self):
        adata_normalized = size_normalize(self.adata)
        assert np.all(adata_normalized.layers["old"] <= adata_normalized.layers["total"])
        assert np.all(adata_normalized.layers["new"] <= adata_normalized.layers["total"])

    def test_gene_subsetting(self):
        genes_subset = np.random.choice(self.genes, size=5, replace=False)
        adata_normalized = size_normalize(self.adata, genes=genes_subset)
        assert set(adata_normalized.var_names) == set(genes_subset)

    def test_row_sums_equal(self):
        adata_normalized = size_normalize(self.adata)
        np.testing.assert_allclose(
            np.sum(adata_normalized.layers["old"], axis=1),
            np.sum(adata_normalized.layers["new"], axis=1),
            rtol=1e-5,
        )

    def test_unsparsify(self):
        adata_normalized = size_normalize(self.adata, unsparsify=True)
        assert not issparse(adata_normalized.layers["total"])
        assert not issparse(adata_normalized.layers["new"])
        assert not issparse(adata_normalized.layers["old"])

class test_neighbors:
    def test_neighbors_distances_shape(self):
        neighbors(self.adata, total_layer="total", n_neighbors=5, include_self=False)
        assert "neighbors" in self.adata.uns
        assert "distances" in self.adata.uns["neighbors"]
        assert self.adata.uns["neighbors"]["distances"].shape == (self.adata.shape[0], self.adata.shape[0])

    def test_neighbors_connectivities_shape(self):
        neighbors(self.adata, total_layer="total", n_neighbors=5, include_self=False)
        assert "neighbors" in self.adata.uns
        assert "connectivities" in self.adata.uns["neighbors"]
        assert self.adata.uns["neighbors"]["connectivities"].shape == (self.adata.shape[0], self.adata.shape[0])

    def test_connectivities_row_sum(self):
        conn = connectivities(adata=self.adata, n_neighbors=5, zero_diagonal=True)
        assert np.allclose(conn.sum(axis=1), np.array([4]*self.adata.shape[0]))

    def test_connectivities_diagonal(self):
        conn = connectivities(adata=self.adata, n_neighbors=5, zero_diagonal=True)
        assert not np.any(np.diag(conn.toarray()))

    def test_connectivities_shape(self):
        conn = connectivities(adata=self.adata, n_neighbors=5, zero_diagonal=True)
        assert conn.shape == (self.adata.shape[0], self.adata.shape[0])

    def test_knn_shapes_include_self(self):
        knn_distances, knn_indices = get_knn_distances_and_indices(adata=self.adata, n_neighbors=5, include_self=True)
        assert knn_distances.shape == (self.adata.shape[0], 5)
        assert knn_indices.shape == (self.adata.shape[0], 5)

    def test_knn_shapes_no_include_self(self):
        knn_distances, knn_indices = get_knn_distances_and_indices(adata=self.adata, n_neighbors=5, include_self=False)
        assert knn_distances.shape == (self.adata.shape[0], 4)
        assert knn_indices.shape == (self.adata.shape[0], 4)

    def test_csr_output_type(self):
        result = get_distances_csr(self.adata, total_layer='total', n_neighbors=5, include_self=True)
        assert isinstance(result, csr_matrix)

    def test_csr_output_shape(self):
        result = get_distances_csr(self.adata, total_layer='total', n_neighbors=5, include_self=True)
        assert result.shape == (self.adata.shape[0], self.adata.shape[0])