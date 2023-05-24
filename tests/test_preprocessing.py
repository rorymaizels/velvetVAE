import unittest
import numpy as np
import scanpy as sc
from scipy.sparse import issparse


from velvet.preprocessing import (
    moments,
    size_normalize,
    read,
    select_genes,
    neighbors,
    connectivities,
    get_knn_distances_and_indices,
    get_distances_csr,
)

class BaseClass(unittest.TestCase):
    def setUp(self): 
        self.file_path = '/workspaces/Velvet/tests/data/test_preprocessing.h5ad'
        self.adata = sc.read('/workspaces/Velvet/tests/data/test_preprocessing.h5ad')
        self.X = sc.read('/workspaces/Velvet/tests/data/test_downstream.h5ad').layers['total']
        self.connectivities = connectivities(total = self.X)
        self.curated_list = np.random.choice(self.adata.var_names, size=10, replace=False).tolist()
        self.unwanted_list = np.random.choice(self.adata.var_names, size=10, replace=False).tolist()

    def tearDown(self):
        pass

class TestRead(BaseClass):
    def test_read_output_type(self):
        output = read(self.file_path)
        self.assertIsInstance(output, anndata.AnnData)


class TestMoments(BaseClass):
    def test_moments_rescale_true(self):
        Xs = moments(self.X, self.connectivities, rescale=True, n_neighbors=5)
        self.assertIsInstance(Xs, np.ndarray)

    def test_moments_rescale_false(self):
        Xs = moments(self.X, self.connectivities, rescale=False, n_neighbors=5)
        self.assertIsInstance(Xs, np.ndarray)

    def test_moments_wrong_X_type(self):
        with self.assertRaises(TypeError):
            moments("invalid type", self.connectivities)

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

class TestGeneSelection(BaseClass):
    def test_genes_in_output_exist_in_input(self):
        selected_genes = select_genes(self.adata)
        assert all(
            gene in self.adata.var_names for gene in selected_genes
        ), "Not all genes in the output list exist in the input data's data.var_names."

    def test_curated_genes_in_output(self):
        selected_genes = select_genes(self.adata, curated_list=self.curated_list)
        assert all(
            gene in selected_genes for gene in self.curated_list
        ), "Not all genes from curated list are present in the output."

    def test_unwanted_genes_not_in_output(self):
        selected_genes = select_genes(self.adata, unwanted_list=self.unwanted_list)
        assert all(
            gene not in selected_genes for gene in self.unwanted_list
        ), "Some unwanted genes are present in the output."

class TestSizeNormalize(BaseClass):
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
        genes_subset = np.random.choice(self.adata.var_names, size=5, replace=False)
        adata_normalized = size_normalize(self.adata, genes=genes_subset)
        assert set(adata_normalized.var_names) == set(genes_subset)

    def test_row_sums_equal(self):
        adata_normalized = size_normalize(self.adata)
        np.testing.assert_allclose(
            np.array(np.sum(adata_normalized.layers["old"], axis=1)).flatten(),
            rtol=1e-1,
        )
        np.testing.assert_allclose(
            np.array(np.sum(adata_normalized.layers["new"], axis=1)).flatten(),
            rtol=1e-1,
        )

    def test_unsparsify(self):
        adata_normalized = size_normalize(self.adata, unsparsify=True)
        assert not issparse(adata_normalized.layers["total"])
        assert not issparse(adata_normalized.layers["new"])
        assert not issparse(adata_normalized.layers["old"])