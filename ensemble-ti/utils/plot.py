import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phate
 
from matplotlib import cm
from sklearn.manifold import TSNE


def plot_embeddings(X, **kwargs):
    assert X.shape[-1] == 2
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], **kwargs)
    plt.show()


def generate_plot_embeddings(X, method='tsne', **kwargs):
    if method == 'phate':
        phate_op = phate.PHATE(**kwargs)
        X_phate = phate_op.fit_transform(X)
        return X_phate
    elif method == 'tsne':
        tsne = TSNE(n_components=2, **kwargs)
        X_tsne = tsne.fit_transform(X)
        return X_tsne
    else:
        raise ValueError(f'Unsupported embedding method type: {method}')


def plot_gene_expression(adata, genes, nrows=1, cmap=None, figsize=None, marker_size=5, obsm_key='X_embedded'):
    # BUG: Currently displays the colormap for
    # each gene individually. Update to get a common vmin and
    # vmax for all the genes that need to be plotted
    assert type(genes).__name__ in ['list', 'tuple']
    try:
        X_embedded = adata.obsm[obsm_key]
    except KeyError:
        raise Exception(f'Key {obsm_key} not found in {adata}')

    try:
        X_imputed = adata.obsm['X_magic']
    except KeyError:
        print('MAGIC imputed data not found. Using raw counts instead')
        X_imputed = adata.X
    
    assert X_embedded.shape[-1] == 2
    cmap = cm.Spectral_r if cmap is None else cmap
    raw_data_df = adata.to_df()
    imputed_data_df = pd.DataFrame(X_imputed, columns=raw_data_df.columns, index=raw_data_df.index)

    # Remove genes excluded from the analysis
    excluded_genes = set(genes) - set(raw_data_df.columns)
    if len(excluded_genes) != 0:
        print(f'The following genes were not plotted: {excluded_genes}')

    net_genes = list(set(genes) - set(excluded_genes))
    ncols = math.ceil(len(net_genes) / nrows)
    gs = plt.GridSpec(nrows=nrows, ncols=ncols)
    fig = plt.figure(figsize=figsize)
    gene_index = 0
    for row_idx in range(nrows):
        for col_idx in range(ncols):
            gene_name = net_genes[gene_index]
            gene_expression = imputed_data_df[gene_name].to_numpy()
            axes = plt.subplot(gs[row_idx, col_idx]) 
            axes.scatter(
                X_embedded[:, 0], X_embedded[:, 1], s=marker_size,
                c=gene_expression, cmap=cmap
            )
            axes.set_title(gene_name)
            axes.set_axis_off()
            gene_index = gene_index + 1

            # Display the Colorbar
            vmin = np.min(gene_expression)
            vmax = np.max(gene_expression)
            normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            cax, _ = matplotlib.colorbar.make_axes(axes)
            matplotlib.colorbar.ColorbarBase(cax, norm=normalize, cmap=plt.get_cmap(cmap))
    plt.show()


def plot_clusters(adata, cluster_key='communities', embedding_key='X_embedding', cmap=None, figsize=(12, 8)):
    communities = adata.obs[cluster_key]
    embeddings = adata.obsm[embedding_key]
    # Only 2d embeddings can be visualized :)
    assert embeddings.shape[-1] == 2

    plt.figure(figsize=figsize)
    axes = plt.subplot(111)
    scatter = axes.scatter(embeddings[:, 0], embeddings[:, 1], c=communities, s=8, cmap=cmap)
    legend1 = axes.legend(*scatter.legend_elements(num=len(np.unique(communities))), loc="center left", title="Cluster Id", bbox_to_anchor=(1, 0.5))
    axes.add_artist(legend1)
    axes.set_axis_off()
    plt.show()


def plot_pseudotime(adata, cmap=None, figsize=None, marker_size=5):
    pseudotime = adata.obsm['X_pseudotime']
    X_embedded = adata.obsm['X_embedded']

    # Plot
    axes = plt.subplot(111)
    axes.scatter(
        X_embedded[:, 0], X_embedded[:, 1], s=marker_size,
        c=pseudotime, cmap=cmap
    )
    axes.set_axis_off()

    # Display the Colorbar
    vmin = np.min(pseudotime)
    vmax = np.max(pseudotime)
    normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cax, _ = matplotlib.colorbar.make_axes(axes)
    matplotlib.colorbar.ColorbarBase(cax, norm=normalize, cmap=plt.get_cmap(cmap))
    plt.show()
