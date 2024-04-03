"""plotting functions"""
from typing import List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from scipy.sparse import issparse
import pandas as pd
from adjustText import adjust_text
import numpy as np
import scvelo as scv
import scanpy as sc
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch
import torchsde

from velvetvae.submodule import SDE, MarkovProcess

def scatter_3d(adata, color="cell_annotation", basis="vae", c=[0, 1, 2], size=False):
    """
    Plots data points in 3D using plotly.

    Args:
        adata: The annotated data matrix with data points.
        color (str, optional): The key of the color values in adata.obs. Defaults to 'cell_annotation'.
        basis (str, optional): The basis for the scatter plot. Defaults to 'vae'.
        c (list, optional): A list of the indices of the coordinates to use for the scatter plot. Defaults to [0,1,2].
        size (bool, optional): If True, the size of the points reflects the 'timepoint' value in adata.obs. Defaults to False.

    Displays:
        A 3D scatter plot.
    """
    X = "X_" + basis

    if color:
        col = adata.obs[color]
    else:
        col = [""] * adata.shape[0]

    if size:
        df = pd.DataFrame(
            data={
                "x": adata.obsm[X][:, c[0]],
                "y": adata.obsm[X][:, c[1]],
                "z": adata.obsm[X][:, c[2]],
                "c": col,
                "s": (adata.obs["timepoint"]),
            }
        )
        fig = px.scatter_3d(df, x="x", y="y", z="z", color="c", size="s", hover_name="c")
    else:
        df = pd.DataFrame(
            data={"x": adata.obsm[X][:, c[0]], "y": adata.obsm[X][:, c[1]], "z": adata.obsm[X][:, c[2]], "c": col}
        )
        fig = px.scatter_3d(df, x="x", y="y", z="z", color="c", hover_name="c")
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()


def velocity_3d(adata, basis="vae", c=[0, 1, 2], scale=3):
    """
    Plots velocity vectors in 3D using plotly.

    Args:
        adata: The annotated data matrix with data points.
        basis (str, optional): The basis for the velocity vectors. Defaults to 'vae'.
        c (list, optional): A list of the indices of the coordinates to use for the velocity vectors. Defaults to [0,1,2].
        scale (int, optional): A scaling factor for the velocity vectors. Defaults to 3.

    Displays:
        A 3D plot with velocity vectors.
    """
    X = adata.obsm[f"X_{basis}"].copy()
    V = adata.obsm[f"velocity_{basis}"].copy()

    df = pd.DataFrame(
        data={"x": X[:, c[0]], "y": X[:, c[1]], "z": X[:, c[2]], "u": V[:, c[0]], "v": V[:, c[1]], "w": V[:, c[2]]}
    )
    fig = go.Figure(data=go.Cone(x=df["x"], y=df["y"], z=df["z"], u=df["u"], v=df["v"], w=df["w"], sizeref=scale))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()

def velocity_stream(
    model, 
    basis, 
    figsize=(8,6), 
    dpi=200,
    title="",
    color=None, 
    show=True,
    palette=None,
    arrow_size=2,
    legend_fontoutline=10,
    size=200,
    fontsize=16, 
    legend_fontsize=16, 
    components='1,2'
):
    """super simple wrapper for scvelo's plotting function."""
    model.adata.uns["velocity_params"] = {'embeddings':basis}

    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.subplots()

    scv.pl.velocity_embedding_stream(
        model.adata, 
        basis=basis, 
        title=title,
        color=color, 
        show=show,
        ax=ax,
        palette=palette,
        arrow_size=arrow_size,
        legend_fontoutline=legend_fontoutline,
        size=size,
        fontsize=fontsize, 
        legend_fontsize=legend_fontsize, 
        components=components
    )
    
def plot_trajectories(
    model, 
    trajectories: np.ndarray, 
    cluster_labels: Optional[List[str]] = None, 
    cell_color: Optional[str] = None,
    trajectory_alpha: float = 0.2,
    cmap: Optional[List[str]] = None,
    components: List[int] = [0, 1],
    figsize=(8,8),
    dpi=200
):
    """
    Plots trajectories of data points in PCA space with different colors for different clusters.

    Args:
        model: A fitted model object.
        trajectories: The trajectories of data points.
        cluster_labels: The cluster labels corresponding to trajectories.
        cmap: The colormap used for coloring different clusters.
        components (list, optional): A list of the indices of the PCA components to use for the scatter plot. Defaults to [0,1].

    Displays:
        A scatter plot with trajectories of data points in PCA space.
    """
    try:
        z = model.adata.obsm["X_z"]
    except:
        X = model.adata_manager.get_from_registry("X")
        X = X.A if issparse(X) else X
        x = torch.tensor(X, device=model.device)
        b = torch.arange(X.shape[0], device=model.device)
        model.module.to(model.device)
        with torch.no_grad():
            z = model.module.inference(x, b)["z"].detach().cpu().numpy()
            
    pca = PCA()
    z_pca = pca.fit_transform(z)
    t_pca = []
    for traj in trajectories:
        t_pca.append(pca.transform(traj))

    copy = model.adata.copy()
    copy.obsm["X_vae"] = z_pca
    copy.uns["velocity_params"] = {"embeddings": "vae"}
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    sc.pl.scatter(
        copy,
        basis="vae",
        color=cell_color,
        ax=ax,
        size=40,
        show=False,
        components=f"{components[0]+1},{components[1]+1}",
    )

    if cluster_labels is None:
        for t in t_pca:
            plt.plot(t[:, components[0]], t[:, components[1]], alpha=trajectory_alpha)
    else:
        if cmap is None:
            unique_labels = np.unique(cluster_labels)
            colors = plt.get_cmap('tab10')
            cmap = {label: colors(i % colors.N) for i, label in enumerate(unique_labels)}
        for t, cl in zip(t_pca, cluster_labels):
            color = cmap[cl]
            plt.plot(t[:, components[0]], t[:, components[1]], color=color, alpha=trajectory_alpha)
    plt.show()

def plot_gene_expression(
    model, 
    trajectories, 
    genes, 
    colors=None,
    colormap='inferno',
    alpha=0.3,
    mean_normalize=False,
    figsize=(6,5),
    dpi=200,
    title="",
    fontsize=18,
    mu_linewidth=5,
):
    
    gex = model.get_trajectory_gene_expression(trajectories)

    plt.figure(figsize=figsize, dpi=dpi)
    
    cmap = plt.get_cmap(colormap)
    colors = [cmap(i) for i in np.linspace(0.2,0.9,len(genes))]

    for gene, col in zip(genes,colors):        
        gene_index = np.where(model.adata.var_names==gene)[0][0]
        gene_data = gex[:,:,gene_index].cpu().numpy()
        mu = gene_data.mean(0)
        if mean_normalize:    
            mumax = mu.max()
            gene_data = gene_data / mumax
            mu = mu / mumax
            
        for t in gene_data:
            plt.plot(t, color=col, alpha=alpha)
        plt.plot(mu, linewidth=mu_linewidth, color=col, label=gene, zorder=10)
        plt.plot(mu, linewidth=mu_linewidth+2, color='black', zorder=9)
        
    plt.title(title, fontsize=fontsize)
    plt.ylabel("Predicted expression", fontsize=fontsize)
    plt.xlabel("Simulation time", fontsize=fontsize)
    plt.show()


def pca_contributions(
    adata,
    genes: Optional[List[str]] = None,
    n_genes: Optional[int] = None,
    ax=None,
    dpi: int = 150,
    components: List[int] = [0, 1],
    show: bool = True,
    plot_velocity: bool = False,
    color: Optional[str] = None,
    palette: Optional[str] = None,
    legend_loc: str = "right margin",
    size: int = 500,
    fontsize: int = 18,
    legend_fontsize: int = 16,
    alpha: float = 0.8,
    figsize: Tuple[int, int] = (8, 8),
    arrow_scale: int = 100,
    arrow_width: float = 0.1,
    arrow_head_width: float = 0.3,
    arrow_color: str = "r",
    norm_arrows: bool = True,
    title: str = "PCA Contributions",
):
    """
    Plots the contributions of genes to the first few principal components and optionally adds velocity streams.

    Args:
        adata: The annotated data matrix.
        genes (list, optional): The list of genes for which to plot PCA contributions. Defaults to None.
        n_genes (int, optional): The number of top contributing genes to plot. Defaults to None.
        ax (matplotlib.axes.Axes, optional): A matplotlib axes object to plot on. Defaults to None.
        dpi (int, optional): The resolution of the plot. Defaults to 150.
        components (list, optional): A list of the indices of the PCA components to use for the plot. Defaults to [0,1].
        show (bool, optional): Whether to show the plot immediately. Defaults to True.
        plot_velocity (bool, optional): Whether to add velocity streams to the scatter plot. Defaults to True.
        color (str, optional): The color of the scatter points. Defaults to None.
        palette (str, optional): The color palette to use for coloring the scatter points. Defaults to None.
        legend_loc (str, optional): The location of the legend. Defaults to 'right margin'.
        size (int, optional): The size of the scatter points. Defaults to 500.
        fontsize (int, optional): The font size for labels and legends. Defaults to 18.
        legend_fontsize (int, optional): The font size for the legend. Defaults to 16.
        alpha (float, optional): The transparency level of the scatter points. Defaults to 0.8.
        figsize (tuple, optional): The size of the figure in inches. Defaults to (8,8).
        arrow_scale (int, optional): A scaling factor for the arrows representing gene contributions. Defaults to 100.
        arrow_width (float, optional): The width of the arrows. Defaults to .1.
        arrow_head_width (float, optional): The width of the arrow heads. Defaults to .3.
        arrow_color (str, optional): The color of the arrows. Defaults to 'r'.
        norm_arrows (bool, optional): Whether to normalize the arrows. Defaults to True.
        title (str, optional): The title of the plot. Defaults to 'PCA Contributions'.

    Displays:
        A scatter plot with the contributions of genes to the first few principal components.
    """
    if genes is None and n_genes is None:
        print("Please supply either list of genes to plot (genes) or number of top genes to plot (n_genes)")

    coeff = adata.varm["PCs"][:, components]
    labels = adata.var_names
    if n_genes is not None:
        selected_vars = np.linalg.norm(coeff, axis=1).argsort()[-n_genes:][::-1]
    elif genes is not None:
        selected_vars = [np.where(adata.var_names == g)[0][0] for g in genes]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if plot_velocity:
        scv.pl.velocity_embedding_stream(
            adata,
            basis="pca",
            color=color,
            show=False,
            ax=ax,
            alpha=alpha,
            title=title,
            palette=palette,
            legend_loc=legend_loc,
            size=size,
            fontsize=fontsize,
            legend_fontsize=legend_fontsize,
            components=f"{components[0]+1},{components[1]+1}",
        )
    else:
        scv.pl.scatter(
            adata,
            basis="pca",
            color=color,
            show=False,
            ax=ax,
            alpha=alpha,
            title=title,
            palette=palette,
            legend_loc=legend_loc,
            size=size,
            fontsize=fontsize,
            legend_fontsize=legend_fontsize,
            components=f"{components[0]+1},{components[1]+1}",
        )

    texts = []
    for i in selected_vars:
        norm = np.linalg.norm(coeff[i, :]) if norm_arrows else 1
        scale = arrow_scale * norm
        ax.arrow(
            0,
            0,
            scale * coeff[i, 0],
            scale * coeff[i, 1],
            width=arrow_width,
            head_width=arrow_head_width,
            color=arrow_color,
            alpha=1,
        )
        texts.append(
            ax.text(
                scale * coeff[i, 0] * 1.5,
                scale * coeff[i, 1] * 1.5,
                labels[i],
                fontsize=fontsize,
                color=arrow_color,
                ha="center",
                va="center",
            )
        )
    adjust_text(texts)

    if show == False:
        return ax
    else:
        plt.show()


def quick_visualisation(
    model,
    n_components: int = 10,
    components: str = "1,2",
    palette: Optional[str] = None,
    color: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 6),
    title: str = "Velvet",
    size: int = 200,
    fontsize: int = 16,
    show: bool = True,
    alpha: float = 0.5,
    return_pca: bool = False,
    show_quiver: bool = True,
):
    X = model.adata_manager.get_from_registry("X")
    X = X.A if issparse(X) else X

    x = torch.tensor(X, device="cuda")
    b = torch.zeros(X.shape[0], device="cuda")

    z = model.module.inference(x, b)["z"]
    v = model.module.vf(z)

    pca = PCA(n_components=n_components)
    zs = z.detach().cpu().numpy()
    zfs = (z + v).detach().cpu().numpy()
    z_pca = pca.fit_transform(zs)
    zf_pca = pca.transform(zfs)
    v_pca = zf_pca - z_pca

    copy = model.adata.copy()
    copy.obsm["X_latent_space_pca"] = z_pca
    copy.obsm["velocity_latent_space_pca"] = v_pca
    copy.uns["velocity_params"] = {"embeddings": "latent_space_pca"}
    fig = plt.figure(figsize=figsize)

    ax = fig.subplots()
    if show_quiver:
        scv.pl.velocity_embedding_stream(
            copy,
            basis="latent_space_pca",
            title=title,
            color=color,
            show=False,
            ax=ax,
            alpha=alpha,
            palette=palette,
            size=size,
            legend_loc="right margin",
            fontsize=fontsize,
            legend_fontsize=fontsize,
            components=components,
        )
    else:
        scv.pl.scatter(
            copy,
            basis="latent_space_pca",
            title=title,
            color=color,
            show=False,
            ax=ax,
            alpha=alpha,
            palette=palette,
            size=size,
            legend_loc="right margin",
            fontsize=fontsize,
            legend_fontsize=fontsize,
            components=components,
        )
    if show:
        plt.show()
        return copy
    else:
        if return_pca:
            return ax, pca
        else:
            return ax


def create_subplots(n_subplots: int, dpi: int = 150):
    """
    Create a matplotlib figure object with subplots arranged in a grid
    that is as close to 1.5 times more rows than columns as possible.

    Args:
        n_subplots (int): Number of subplots to create.
        dpi (int, optional): The resolution of the plot. Defaults to 150.

    Returns:
        fig (matplotlib.figure.Figure): Matplotlib figure object.
        axes (list): List of subplot axis objects.
    """
    fig_rows = int(np.ceil(np.sqrt(n_subplots * 1.5)))
    fig_cols = int(np.ceil(n_subplots / fig_rows))

    fig_width = fig_cols * 6
    fig_height = fig_rows * 4

    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_width, fig_height), dpi=dpi)

    # Flatten the 2D array of axes into a 1D list
    axes = axes.flatten()

    # Hide any unused subplots
    for i in range(len(axes)):
        if i >= n_subplots:
            axes[i].axis("off")

    return fig, axes[:n_subplots]


def compare_simulation_noise(
    model,
    mp: Optional["MarkovProcess"],
    n_repeats: int,
    n_steps: int,
    t_max: float,
    noise_scalar: float,
    dt: float,
    n_markov_steps: int,
    n_neighbors: int,
    n_tests: int = 10,
    components: List[int] = [0, 1],
    cell_color: Optional[str] = None,
    cell_size: int = 200,
    cell_alpha: float = 0.2,
    alpha: float = 0.5,
    title: str = "",
    det: bool = False,
) -> None:
    """
    Compare the results of the Markov and nSDE simulation of a model.

    Args:
        model: A fitted model object.
        mp: An object of the MarkovProcess class.
        n_repeats: Number of repetitions for each simulation.
        n_steps: Number of steps in each simulation.
        t_max: The maximum time for each simulation.
        noise_scalar: The scalar to adjust the noise level in the stochastic simulation.
        dt: The time step size for the stochastic simulation.
        n_markov_steps: Number of jumps in the Markov simulation.
        n_neighbors: Number of neighbors to consider in the Markov simulation.
        n_tests: Number of times the whole comparison process is repeated.
        components: The PCA components to consider for the plot.
        cell_color: The color of the scatter points.
        cell_size: The size of the scatter points.
        cell_alpha: The transparency level of the scatter points.
        alpha: The transparency level of the trajectory lines.
        title: The title of the plot.
        det: If True, the Markov simulation is deterministic.
        figsize: The size of the figure.

    Returns:
        None
    """
    z = model.adata.obsm["X_z"]
    z = torch.tensor(z, device=model.device)

    if mp is None:
        mp = MarkovProcess(
            model.adata,
            n_neighbors=n_neighbors,
            use_space="latent_space",
            use_spline=True,
        )
    sde = SDE(
        latent_dim=model.module.n_latent,
        prior_vectorfield=model.module.vf,
        noise_scalar=noise_scalar,
    )

    fig, axes = create_subplots(n_tests)

    for j in range(n_tests):
        cell_index = torch.randperm(z.shape[0])[:1]

        mtrajectories = torch.zeros(n_steps, n_repeats, z.shape[1])
        strajectories = torch.zeros(n_steps, n_repeats, z.shape[1])

        for i in range(n_repeats):
            mtrajectories[:, i, :] = mp.random_walk(
                z=z, initial_states=cell_index, n_jumps=n_markov_steps, n_steps=n_steps, deterministic=det
            ).squeeze(1)

        initial_cells = z[cell_index]
        initial_cells = initial_cells.repeat(n_repeats, 1)

        timespan = torch.linspace(0, t_max, n_steps, device=z.device)
        strajectories = torchsde.sdeint_adjoint(sde, initial_cells, timespan, method="midpoint", dt=dt)

        pca = PCA()
        z_pca = pca.fit_transform(z.detach().cpu().numpy())
        tm_pca = []
        ts_pca = []

        for i in range(n_repeats):
            tm = mtrajectories[:, i, :].squeeze().detach().cpu().numpy()
            tm_pca.append(pca.transform(tm))

            ts = strajectories[:, i, :].squeeze().detach().cpu().numpy()
            ts_pca.append(pca.transform(ts))

        copy = model.adata.copy()
        copy.obsm["X_vae"] = z_pca
        copy.uns["velocity_params"] = {"embeddings": "vae"}

        sc.pl.scatter(
            copy,
            basis="vae",
            color=cell_color,
            title=title,
            ax=axes[j],
            size=cell_size,
            alpha=cell_alpha,
            show=False,
            components=f"{components[0]+1},{components[1]+1}",
        )

        for i, t in enumerate(tm_pca):
            if i == 0:
                axes[j].plot(t[:, components[0]], t[:, components[1]], color="tab:blue", alpha=alpha, label="Markov")
            else:
                axes[j].plot(t[:, components[0]], t[:, components[1]], color="tab:blue", alpha=alpha)
        for i, t in enumerate(ts_pca):
            if i == 0:
                axes[j].plot(t[:, components[0]], t[:, components[1]], color="tab:orange", alpha=alpha, label="SDE")
            else:
                axes[j].plot(t[:, components[0]], t[:, components[1]], color="tab:orange", alpha=alpha)
        axes[j].legend()
    plt.tight_layout()
    plt.show()


def trajectories_2d(
    model,
    trajectories,
    labels: Optional[List] = None,
    colors: List[str] = list(mcolors.TABLEAU_COLORS.keys()),
    components: List[int] = [0, 1],
    cell_color: Optional[str] = None,
    cell_alpha: float = 0.4,
    cell_size: int = 1000,
    cell_palette: Optional[str] = None,
    line_alpha: float = 0.2,
    ax=None,
    title: str = "",
    show: bool = True,
):
    """
    Plot 2D trajectories.

    Args:
        model: A fitted model object.
        trajectories: A list of tensors representing cell trajectories.
        labels: Cluster labels for different trajectories.
        colors: Colors for different clusters.
        components: The PCA components to consider for the plot.
        cell_color: The color of the scatter points.
        cell_alpha: The transparency level of the scatter points.
        cell_size: The size of the scatter points.
        cell_palette: The color palette to use for coloring the scatter points.
        line_alpha: The transparency level of the trajectory lines.
        ax: A matplotlib axes object on which to draw the plot.
        title: The title of the plot.
        show: Whether to show the plot immediately.

    Returns:
        If show is False, returns the matplotlib figure object; otherwise, returns None.
    """
    z = model.adata.obsm["X_z"]
    z = torch.tensor(z, device=model.device)

    pca = PCA()
    z_pca = pca.fit_transform(z.detach().cpu().numpy())

    t_pca = []
    for traj in trajectories:
        t_pca.append(pca.transform(traj.detach().cpu().numpy()))

    copy = model.adata.copy()
    copy.obsm["X_vae"] = z_pca
    copy.uns["velocity_params"] = {"embeddings": "vae"}

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.subplots()

    scv_components = f"{components[0]+1},{components[1]+1}"
    scv.pl.scatter(
        copy,
        basis="vae",
        color=cell_color,
        alpha=cell_alpha,
        ax=ax,
        size=cell_size,
        show=False,
        components=scv_components,
        legend_loc=False,
        title=title,
        fontsize=16,
        palette=cell_palette,
    )

    if labels is not None:
        use_labels = True
        n_labels = len(np.unique(labels))
        if n_labels > len(colors):
            print("Not enough colors provided for number of clusters. Update colors argument.")
            return
        colors = colors[:n_labels]
    else:
        use_labels = False
        labels = [0] * len(t_pca)

    groups = []
    for t, cl in zip(t_pca, labels):
        if use_labels:
            color = colors[cl]
            if cl in groups:
                label = ""
            else:
                label = cl
                groups.append(cl)
        else:
            color = None
            label = None
        ax.scatter(t[0, components[0]], t[0, components[1]], color="red", marker="x")
        ax.plot(t[:, components[0]], t[:, components[1]], color=color, alpha=line_alpha, linewidth=2, label=label)
    if use_labels:
        plt.legend(loc=(0.8, 0.55), fontsize=12)
    plt.tight_layout()
    if show:
        plt.show()
