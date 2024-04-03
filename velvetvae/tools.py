"""tools"""
import numpy as np
import torch
from sklearn.decomposition import PCA
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler

def latent_space_pca(model, device='cuda', n_components=10, embedding_key='latent_pca'):
    torch_device = 'cuda' if (torch.cuda.is_available() and device=='cuda') else 'cpu'
    
    X = model.adata_manager.get_from_registry("X")
    X = X.A if issparse(X) else X

    x = torch.tensor(X, device=torch_device)
    b = torch.zeros(X.shape[0], device=torch_device)

    z = model.module.inference(x, b)['z']
    v = model.module.vf(z)

    pca = PCA(n_components=n_components)
    zs = z.detach().cpu().numpy()
    zfs = (z+v).detach().cpu().numpy()
    z_pca = pca.fit_transform(zs)
    zf_pca = pca.transform(zfs)
    v_pca = zf_pca - z_pca

    model.adata.obsm[f'X_{embedding_key}'] = z_pca
    model.adata.obsm[f'velocity_{embedding_key}'] = v_pca
    model.adata.uns["velocity_params"] = {'embeddings':embedding_key}
    
def gene_space_pca(model, n_components=10, embedding_key='latent_pca'):
    scaler = StandardScaler(with_mean=True, with_std=False)
    pca = PCA(n_components=n_components)
    
    X = model.adata_manager.get_from_registry("X")
    X = X.A if issparse(X) else X    
    V = model.predict_velocity()
    X = np.array(X.A if issparse(X) else X)
    V = np.array(V.A if issparse(V) else V)

    Y = np.clip(X + V, 0, 1000)
            
    Xlogscale = scaler.fit_transform(np.log1p(X))      
    Ylogscale = scaler.transform(np.log1p(Y))
    Xpca = pca.fit_transform(Xlogscale)
    Ypca = pca.transform(Ylogscale)

    model.adata.obsm[f'X_{embedding_key}'] = Xpca
    model.adata.varm[f'PCs_{embedding_key}'] = pca.components_
    model.adata.obsm[f'velocity_{embedding_key}'] = Ypca - Xpca
    model.adata.uns["velocity_params"] = {'embeddings':embedding_key}

def get_plot_data(
    model,
    gene_expression,
    gene,
    mean_normalize=True
):
    gene_index = np.where(model.adata.var_names==gene)[0][0]
    gene_data = gene_expression[:,:,gene_index].cpu().numpy()
    
    mu = gene_data.mean(0)
    if mean_normalize:    
        mumax = mu.max()
        gene_data = gene_data / mumax
        mu = mu / mumax
    return gene_data, mu


