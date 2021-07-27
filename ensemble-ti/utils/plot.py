import math
import matplotlib as mp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import phate
import pygam as pg
import scanpy as sc
import scipy
import umap

from matplotlib import cm
from matplotlib.text import Annotation
from matplotlib.font_manager import FontProperties
from sklearn.manifold import TSNE

from models.ti.graph import (
    compute_connectivity_graph,
    compute_trajectory_graph,
    compute_trajectory_graph_v2,
)
from utils.util import compute_runtime


# TODO: In the plotting module, create a decorator to save the plots


@compute_runtime
def generate_plot_embeddings(X, method="tsne", **kwargs):
    if method == "phate":
        phate_op = phate.PHATE(**kwargs)
        X_phate = phate_op.fit_transform(X)
        return X_phate
    elif method == "tsne":
        tsne = TSNE(n_components=2, **kwargs)
        X_tsne = tsne.fit_transform(X)
        return X_tsne
    elif method == "umap":
        u = umap.UMAP(n_components=2, **kwargs)
        X_umap = u.fit_transform(X)
        return X_umap
    else:
        raise ValueError(f"Unsupported embedding method type: {method}")


def plot_annotated_heatmap(
    mat,
    col_labels,
    row_labels,
    figsize=None,
    ax=None,
    cmap="YlGn",
    fontsize=16,
    fontcolor="black",
    save_path=None,
    col_font=None,
    row_font=None,
    save_kwargs={},
    annotate_text=True,
    show_colorbar=True,
    cb_axes_pos=None,
    cb_kwargs={},
    **kwargs,
):
    # Code inspired from: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    # Create figure
    fig = plt.figure(figsize=figsize)

    if ax is None:
        ax = plt.gca()

    # Show plot
    im = ax.imshow(mat, cmap=cmap, **kwargs)

    # Configure axis labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))

    if col_font is not None:
        ax.set_xticklabels(col_labels, fontproperties=col_font, fontsize=fontsize)
    else:
        ax.set_xticklabels(col_labels, fontsize=fontsize)

    if row_font is not None:
        ax.set_yticklabels(row_labels, fontproperties=row_font, fontsize=fontsize)
    else:
        ax.set_yticklabels(row_labels, fontsize=fontsize)

    # Horizontal axes labels appear on top!
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    ax.set_xticks(np.arange(mat.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(mat.shape[0] + 1) - 0.5, minor=True)

    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor")

    # Annotate the heatmap
    if annotate_text:
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                ax.text(
                    j,
                    i,
                    np.round(mat[i, j], 3),
                    ha="center",
                    va="center",
                    color=fontcolor,
                    fontsize=fontsize,
                )

    # Colorbar
    if show_colorbar:
        cax = None
        if cb_axes_pos is not None:
            cax = fig.add_axes(cb_axes_pos)
        plt.colorbar(im, cax=cax, **cb_kwargs)

    # Save figure
    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)

    return im, ax


def plot_embeddings(
    X,
    figsize=(12, 8),
    save_path=None,
    title=None,
    show_legend=False,
    show_colorbar=False,
    axis_off=True,
    hover_labels=None,
    labels=None,
    legend_kwargs={},
    cb_axes_pos=None,
    cb_kwargs={},
    save_kwargs={},
    picker=False,
    **kwargs,
):
    def annotate(axis, text, x, y):
        text_annotation = Annotation(text, xy=(x, y), xycoords="data")
        axis.add_artist(text_annotation)

    def onpick(event):
        ind = event.ind

        label_pos_x = event.mouseevent.xdata
        label_pos_y = event.mouseevent.ydata

        # Take only the first of many indices returned
        label = X[ind[0], :]
        if hover_labels is not None:
            label = hover_labels[ind[0]]

        # Create Text annotation
        annotate(ax, label, label_pos_x, label_pos_y)

        # Redraw the figure
        ax.figure.canvas.draw_idle()

    assert X.shape[-1] == 2

    # Set figsize
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    # Set title (if set)
    if title is not None:
        plt.title(title)

    # Plot
    scatter = ax.scatter(X[:, 0], X[:, 1], picker=picker, **kwargs)

    if show_legend:
        if labels is None:
            raise ValueError("labels must be provided when plotting legend")

        # Create legend
        legend = ax.legend(*scatter.legend_elements(num=len(labels)), **legend_kwargs)

        # Replace default labels with the provided labels
        text = legend.get_texts()
        assert len(text) == len(labels)

        for t, label in zip(text, labels):
            t.set_text(label)
        ax.add_artist(legend)

    if axis_off:
        ax.set_axis_off()

    if show_colorbar:
        cax = None
        if cb_axes_pos is not None:
            cax = fig.add_axes(cb_axes_pos)
        plt.colorbar(scatter, cax=cax, **cb_kwargs)

    # Pick Event handling (useful for selecting start cells)
    if picker is True:
        fig.canvas.mpl_connect("pick_event", onpick)

    # Save
    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)
    plt.show()


def plot_boxplot_expression(
    ad,
    groups,
    order=None,
    cluster_key="metric_clusters",
    imputation_key=None,
    colors=None,
    figsize=None,
    show_labels=False,
    save_path=None,
    save_kwargs={},
    **kwargs,
):
    communities = ad.obs[cluster_key]

    data_ = ad.X
    if imputation_key is not None:
        data_ = ad.obsm[imputation_key]

    if not isinstance(data_, pd.DataFrame):
        if isinstance(data_, scipy.sparse.csr_matrix):
            data_ = data_.todense()
        data_ = pd.DataFrame(data_, index=ad.obs_names, columns=ad.var_names)

    if order is not None:
        for cluster_id in np.unique(order):
            assert cluster_id in np.unique(communities)

    # Set figsize
    plt.figure(figsize=figsize)
    ax = plt.gca()

    for id, (g_id, genes) in enumerate(groups.items()):
        data = []
        for cluster_id in order:
            ids = communities == cluster_id

            # Create the boxplot
            expr_ = []
            for gene in genes:
                if gene not in ad.var_names:
                    print(f"Gene {gene} not found. Skipping")
                    continue
                gene_expr = list(data_.loc[ids, gene])
                expr_.extend(gene_expr)
            data.append(expr_)

        box = plt.boxplot(
            data,
            positions=np.arange(len(order)),
            labels=order,
            patch_artist=True,
            **kwargs,
        )

        # Facecolor for a gene will be same
        if colors is not None:
            for patch in box["boxes"]:
                patch.set(facecolor=colors[id])

        if show_labels:
            ax.set_ylabel("Gene expression")
            ax.set_xlabel("Cluster Ids")

    # Remove the right and top axes
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)
    plt.show()


def plot_gene_expression(
    adata,
    genes,
    nrows=1,
    cmap=None,
    figsize=None,
    marker_size=1,
    obsm_key="X_embedded",
    save_kwargs={},
    save_path=None,
    cb_kwargs={},
    norm=False,
    show_title=False,
    **kwargs,
):
    assert type(genes).__name__ in ["list", "tuple"]
    try:
        X_embedded = adata.obsm[obsm_key]
    except KeyError:
        raise Exception(f"Key {obsm_key} not found in {adata}")

    try:
        X_imputed = adata.obsm["X_magic"]
    except KeyError:
        print("MAGIC imputed data not found. Using raw counts instead")
        X_imputed = adata.X

    assert X_embedded.shape[-1] == 2
    cmap = cm.Spectral_r if cmap is None else cmap
    raw_data_df = adata.to_df()
    imputed_data_df = pd.DataFrame(
        X_imputed, columns=raw_data_df.columns, index=raw_data_df.index
    )

    # Remove genes excluded from the analysis
    excluded_genes = set(genes) - set(raw_data_df.columns)
    if len(excluded_genes) != 0:
        print(f"The following genes were not plotted: {excluded_genes}")

    net_genes = list(set(genes) - set(excluded_genes))
    ncols = math.ceil(len(net_genes) / nrows)
    gs = plt.GridSpec(nrows=nrows, ncols=ncols)
    fig = plt.figure(figsize=figsize)
    gene_index = 0
    for row_idx in range(nrows):
        for col_idx in range(ncols):
            gene_name = genes[gene_index]
            if gene_name not in net_genes:
                gene_index = gene_index + 1
                continue

            gene_expression = imputed_data_df[gene_name].to_numpy()

            if norm:
                gene_expression = (gene_expression - np.min(gene_expression)) / (
                    np.max(gene_expression) - np.min(gene_expression)
                )

            axes = plt.subplot(gs[row_idx, col_idx])
            sc = axes.scatter(
                X_embedded[:, 0],
                X_embedded[:, 1],
                s=marker_size,
                c=gene_expression,
                cmap=cmap,
                **kwargs,
            )
            if show_title:
                axes.set_title(gene_name)
            axes.set_axis_off()
            gene_index = gene_index + 1

            # Colorbar
            # TODO: Adjust the position of the colorbar
            plt.colorbar(sc, ax=axes, **cb_kwargs)

    fig.tight_layout()

    # Save plot
    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)
    plt.show()


def plot_clusters(
    adata,
    cluster_key="communities",
    embedding_key="X_embedding",
    figsize=(8, 8),
    title=None,
    save_path=None,
    color_map=None,
    show_legend=True,
    leg_marker_size=None,
    legend_kwargs={},
    save_kwargs={},
    **kwargs,
):
    communities = adata.obs[cluster_key]
    embeddings = adata.obsm[embedding_key]
    # Only 2d embeddings can be visualized :)
    assert embeddings.shape[-1] == 2

    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    axes = plt.gca()

    for cluster_id in np.unique(communities):
        ids = communities == cluster_id
        c = None if color_map is None else color_map[cluster_id]
        axes.scatter(
            embeddings[ids, 0],
            embeddings[ids, 1],
            c=c,
            label=cluster_id,
            **kwargs,
        )

    axes.set_axis_off()

    if show_legend:
        legend = plt.legend(**legend_kwargs)

        # Hack to change the size of the markers in the legend
        if leg_marker_size is not None:
            for h in legend.legendHandles:
                h.set_sizes([leg_marker_size])

    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)
    plt.show()


def plot_clusters_with_cell_overlay(
    adata,
    cell_ids,
    cluster_key="metric_clusters",
    embedding_key="X_met_embedding",
    overlay_marker_size=5,
    overlay_color="black",
    figsize=(8, 8),
    title=None,
    save_path=None,
    color_map=None,
    show_legend=True,
    leg_marker_size=None,
    legend_kwargs={},
    save_kwargs={},
    **kwargs,
):
    communities = adata.obs[cluster_key]
    embeddings = pd.DataFrame(adata.obsm[embedding_key], index=adata.obs_names)
    # Only 2d embeddings can be visualized :)
    assert embeddings.shape[-1] == 2

    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    axes = plt.gca()

    for cluster_id in np.unique(communities):
        ids = communities == cluster_id
        c = None if color_map is None else color_map[cluster_id]
        axes.scatter(
            embeddings.loc[ids, 0],
            embeddings.loc[ids, 1],
            c=c,
            label=cluster_id,
            **kwargs,
        )

    # Turn-off axes
    axes.set_axis_off()

    # Overlay cells
    overlay_embed = embeddings.loc[cell_ids, :].to_numpy()
    axes.scatter(
        overlay_embed[:, 0],
        overlay_embed[:, 1],
        s=overlay_marker_size,
        c=overlay_color,
        zorder=10,
    )

    if show_legend:
        legend = plt.legend(**legend_kwargs)

        # Hack to change the size of the markers in the legend
        if leg_marker_size is not None:
            for h in legend.legendHandles:
                h.set_sizes([leg_marker_size])

    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)
    plt.show()


def plot_pseudotime(
    adata,
    embedding_key="X_met_embedding",
    pseudotime_key="metric_pseudotime_v2",
    cmap=None,
    figsize=None,
    cb_axes_pos=None,
    save_path=None,
    save_kwargs={},
    cb_kwargs={},
    **kwargs,
):
    # An alias to plotting embeddings with pseudotime projected on it
    pseudotime = adata.obs[pseudotime_key]
    X_embedded = adata.obsm[embedding_key]

    # Plot
    plot_embeddings(
        X_embedded,
        c=pseudotime,
        cmap=cmap,
        figsize=figsize,
        show_colorbar=True,
        cb_axes_pos=cb_axes_pos,
        save_path=save_path,
        save_kwargs=save_kwargs,
        cb_kwargs=cb_kwargs,
        **kwargs,
    )


def plot_graph(
    G,
    node_positions=None,
    figsize=(16, 12),
    title=None,
    save_path=None,
    save_kwargs={},
    offset=0,
    **kwargs,
):
    # Draw the graph
    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    edge_weights = [offset + w for _, _, w in G.edges.data("weight")]
    nx.draw_networkx(G, pos=node_positions, width=edge_weights, **kwargs)
    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)


def plot_trajectory_graph(
    embeddings,
    communities,
    cluster_connectivities,
    start_cell_ids,
    cmap="YlGn",
    figsize=(16, 12),
    node_size=400,
    font_color="black",
    title=None,
    save_path=None,
    save_kwargs={},
    offset=0,
    **kwargs,
):
    g, node_positions = compute_trajectory_graph(
        embeddings, communities, cluster_connectivities, start_cell_ids
    )
    # Draw the graph
    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    edge_weights = [offset + w for _, _, w in g.edges.data("weight")]
    nx.draw_networkx(
        g,
        pos=node_positions,
        cmap=cmap,
        node_color=np.unique(communities),
        font_color=font_color,
        node_size=node_size,
        width=edge_weights,
        **kwargs,
    )
    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)


def plot_trajectory_graph_v2(
    pseudotime,
    adj_cluster,
    communities,
    d_connectivity,
    node_positions,
    start_cell_ids=None,
    cmap="YlGn",
    figsize=(16, 12),
    node_size=400,
    font_color="black",
    title=None,
    start_node_color=None,
    node_color=None,
    save_path=None,
    save_kwargs={},
    offset=0,
    **kwargs,
):
    adj_g = compute_trajectory_graph_v2(
        pseudotime, adj_cluster, communities, d_connectivity
    )
    g = nx.from_pandas_adjacency(adj_g, create_using=nx.DiGraph)

    if start_cell_ids is not None:
        start_cell_ids = (
            start_cell_ids if isinstance(start_cell_ids, list) else [start_cell_ids]
        )
    else:
        start_cell_ids = []

    start_cluster_ids = set([communities.loc[id] for id in start_cell_ids])

    colors = np.unique(communities)
    if node_color is not None:
        colors = []
        for c_id in np.unique(communities):
            if c_id in start_cluster_ids and start_node_color is not None:
                colors.append(start_node_color)
            else:
                colors.append(node_color)

    # Draw the graph
    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    edge_weights = [offset + w for _, _, w in g.edges.data("weight")]
    nx.draw_networkx(
        g,
        pos=node_positions,
        cmap=cmap,
        node_color=colors,
        font_color=font_color,
        node_size=node_size,
        width=edge_weights,
        **kwargs,
    )
    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)


def plot_connectivity_graph(
    embeddings,
    communities,
    cluster_connectivities,
    start_cell_ids=None,
    mode="undirected",
    cmap="YlGn",
    figsize=(12, 12),
    node_size=800,
    font_color="black",
    start_node_color=None,
    node_color=None,
    title=None,
    save_path=None,
    save_kwargs={},
    offset=0,
    **kwargs,
):
    g, node_positions = compute_connectivity_graph(
        embeddings, communities, cluster_connectivities, mode=mode
    )

    if start_cell_ids is not None:
        start_cell_ids = (
            start_cell_ids if isinstance(start_cell_ids, list) else [start_cell_ids]
        )
    else:
        start_cell_ids = []

    start_cluster_ids = set([communities.loc[id] for id in start_cell_ids])

    colors = np.unique(communities)
    if node_color is not None:
        colors = []
        for c_id in np.unique(communities):
            if c_id in start_cluster_ids and start_node_color is not None:
                colors.append(start_node_color)
            else:
                colors.append(node_color)

    # Draw the graph
    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    edge_weights = [offset + w for _, _, w in g.edges.data("weight")]
    nx.draw_networkx(
        g,
        pos=node_positions,
        cmap=cmap,
        node_color=colors,
        font_color=font_color,
        node_size=node_size,
        width=edge_weights,
        **kwargs,
    )
    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)


def plot_gt_milestone_network(
    ad,
    uns_mn_key="milestone_network",
    start_node_color="red",
    node_color="yellow",
    figsize=(12, 12),
    node_size=800,
    font_size=9,
    save_path=None,
    save_kwargs={},
    **kwargs,
):
    # NOTE: Since the dyntoy tool does not provide the spatial position
    # of the milestones, this function uses spring_layout to plot the
    # node positions. Hence the displayed graph is not guaranteed to produce
    # accurate spatial embeddings of milestones in the 2d plane.
    if uns_mn_key not in ad.uns_keys():
        raise Exception(f"Milestone network not found in uns.{uns_mn_key}")

    # Get a set of unique milestones
    mn = ad.uns[uns_mn_key]
    from_milestones = set(mn["from"].unique())
    to_milestones = set(mn["to"].unique())
    milestones = from_milestones.union(to_milestones)

    # Construct the milestone network
    milestone_network = nx.DiGraph()
    start_milestones = (
        [ad.uns["start_milestones"]]
        if isinstance(ad.uns["start_milestones"], str)
        else list(ad.uns["start_milestones"])
    )
    color_map = []
    for milestone in milestones:
        milestone_network.add_node(milestone)
        if milestone in start_milestones:
            color_map.append(start_node_color)
        else:
            color_map.append(node_color)

    for idx, (f, t) in enumerate(zip(mn["from"], mn["to"])):
        milestone_network.add_edge(f, t, weight=mn["length"][idx])

    # Draw graph
    plt.figure(figsize=figsize)
    plt.axis("off")
    edge_weights = [1 + w for _, _, w in milestone_network.edges.data("weight")]
    nx.draw_networkx(
        milestone_network,
        pos=nx.spring_layout(milestone_network),
        node_size=node_size,
        width=edge_weights,
        node_color=color_map,
        font_size=font_size,
        **kwargs,
    )

    # Save
    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)


def plot_lineage_trends(
    ad,
    cell_branch_probs,
    genes,
    pseudotime_key="metric_pseudotime",
    imputed_key=None,
    nrows=1,
    figsize=None,
    norm=True,
    threshold=0.95,
    show_title=False,
    save_path=None,
    ts_map=None,
    color_map=None,
    save_kwargs={},
    gam_kwargs={},
    **kwargs,
):
    t_states = cell_branch_probs.columns
    pt = ad.obs[pseudotime_key]

    data_ = ad.X
    if imputed_key is not None:
        data_ = ad.obsm[imputed_key]

    if isinstance(data_, scipy.sparse.csr_matrix):
        data_ = data_.todense()

    if norm:
        # Min-max normalization
        data_ = (data_ - np.min(data_, axis=0)) / (
            np.max(data_, axis=0) - np.min(data_, axis=0)
        )

    data_df = pd.DataFrame(data_, columns=ad.var_names, index=ad.obs_names)
    ncols = math.ceil(len(genes) / nrows)
    fig, ax = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=figsize
    )
    gene_idx = 0

    # Compute lineage expression trends for each gene
    for row_idx in range(nrows):
        for col_idx in range(ncols):
            if nrows == 1 and ncols == 1:
                axes = ax
            elif nrows == 1:
                axes = ax[col_idx]
            elif ncols == 1:
                axes = ax[row_idx]
            else:
                axes = ax[row_idx, col_idx]

            gene_exp = data_df.loc[:, genes[gene_idx]]
            for i in t_states:
                # Get the val set
                loc_i = np.where(cell_branch_probs.loc[:, i] > threshold)[0]
                if loc_i.shape[0] == 0:
                    continue
                val_pt = pt[loc_i]
                max_val_pt = max(val_pt)

                # Fit GAM
                # NOTE: GAM Code inspired from https://github.com/ShobiStassen/VIA
                loc_i_bp = np.where(cell_branch_probs.loc[:, i] > 0)[0]
                x = np.asarray(pt)[loc_i_bp].reshape(-1, 1)
                y = np.asarray(gene_exp)[loc_i_bp].reshape(-1, 1)
                weights = np.asarray(cell_branch_probs.loc[:, i])[loc_i_bp].reshape(
                    -1, 1
                )
                geneGAM = pg.LinearGAM(
                    n_splines=10, spline_order=4, lam=10, **gam_kwargs
                ).fit(x, y, weights=weights)

                # Eval GAM
                nx_spacing = 100
                xval = np.linspace(0, max_val_pt, nx_spacing * 2)
                yg = geneGAM.predict(X=xval)

                # Plot
                ts_label = ts_map[i] if ts_map is not None else i
                if color_map is not None:
                    kwargs["color"] = color_map[i]

                # Remove the right and top axes
                axes.spines["right"].set_visible(False)
                axes.spines["top"].set_visible(False)
                # axes.set_ylim([0, 1])
                axes.plot(xval, yg, linewidth=3.5, zorder=3, label=ts_label, **kwargs)
            axes.set_ylabel("Normalized expression")
            axes.legend()

            if show_title:
                plt.title(genes[gene_idx])
            gene_idx += 1

    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)
    plt.show()


def plot_connectivity_graph_with_gene_expressions(
    ad,
    cluster_connectivities,
    gene,
    embedding_key="X_met_embedding",
    comm_key="metric_clusters",
    magic_key="X_magic",
    mode="undirected",
    cmap="YlGn",
    figsize=(16, 12),
    node_size=400,
    font_color="black",
    title=None,
    save_path=None,
    save_kwargs={},
    offset=0,
    **kwargs,
):
    try:
        X_embedded = ad.obsm[embedding_key]
    except KeyError:
        raise Exception(f"Key {embedding_key} not found in {ad}")

    try:
        communities = ad.obs[comm_key]
    except KeyError:
        raise Exception(f"Key {comm_key} not found in {ad}")

    try:
        X_imputed = pd.DataFrame(
            ad.obsm[magic_key], index=ad.obs_names, columns=ad.var_names
        )
    except KeyError:
        print("MAGIC imputed data not found. Using raw counts instead")
        X_imputed = ad.to_df()

    if gene not in ad.var_names:
        raise ValueError(f"Gene: {gene} was not found.")

    g, node_positions = compute_connectivity_graph(
        X_embedded, communities, cluster_connectivities, mode=mode
    )

    # Compute cluster wise mean expression of the gene
    X_gene = X_imputed[gene]
    gene_exprs = []
    for cluster_id in np.unique(communities):
        ids = communities == cluster_id
        mean_gene_expr = X_gene.loc[ids].mean()
        gene_exprs.append(mean_gene_expr)

    # Draw the graph
    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    edge_weights = [offset + w for _, _, w in g.edges.data("weight")]
    nx.draw_networkx(
        g,
        pos=node_positions,
        cmap=cmap,
        node_color=gene_exprs,
        font_color=font_color,
        node_size=node_size,
        width=edge_weights,
        **kwargs,
    )
    # Setup color bar
    vmin = np.min(gene_exprs)
    vmax = np.max(gene_exprs)
    normalize = mp.colors.Normalize(vmin=vmin, vmax=vmax)
    cax, _ = mp.colorbar.make_axes(plt.gca())
    mp.colorbar.ColorbarBase(cax, norm=normalize, cmap=plt.get_cmap(cmap))

    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)
    plt.show()


def plot_cell_branch_probs(
    ad,
    cell_ids,
    nrows=1,
    bp_key="metric_branch_probs",
    save_path=None,
    figsize=None,
    save_kwargs={},
    color_map=None,
    tick_map=None,
    **bp_kwargs,
):
    ncols = math.ceil(len(cell_ids) / nrows)
    fig, ax = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=figsize
    )
    branch_probs = ad.obsm[bp_key]
    cell_idx = 0

    for row_idx in range(nrows):
        for col_idx in range(ncols):
            if nrows == 1 and ncols == 1:
                axes = ax
            elif nrows == 1:
                axes = ax[col_idx]
            elif ncols == 1:
                axes = ax[row_idx]
            else:
                axes = ax[row_idx, col_idx]

            cell_id = cell_ids[cell_idx]
            cell_bp = branch_probs.loc[cell_id, :]

            color = None
            if color_map is not None:
                color = [color_map[ts] for ts in cell_bp.index]

            # Barplot of probs
            x = np.arange(cell_bp.shape[-1])
            axes.bar(x, cell_bp, color=color, **bp_kwargs)

            # Remove the right and top axes
            axes.spines["right"].set_visible(False)
            axes.spines["top"].set_visible(False)

            # Set Ticks
            axes.set_xticks(x)
            ticks = []

            if tick_map is not None:
                ticks = [tick_map[t_cell_id] for t_cell_id in cell_bp.index]
            axes.set_xticklabels(ticks, rotation=45)

            cell_idx += 1

    # Save
    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)
    plt.show()


def plot_dp_vs_pseudotime(
    ad,
    lineage,
    comms_key="metric_clusters",
    pt_key="metric_pseudotime_v2",
    dp_key="metric_dp",
    lineage_color_map=None,
    show_label=True,
    figsize=None,
    save_path=None,
    save_kwargs={},
    **kwargs,
):
    lineage_cell_ids = []
    colors = []
    comms = ad.obs[comms_key]

    for cluster_id in lineage:
        assert cluster_id in np.unique(comms)
        cell_ids = list(comms.index[comms == cluster_id])
        lineage_cell_ids.extend(cell_ids)

        # Add cell colors
        if lineage_color_map is not None:
            cluster_color = lineage_color_map[cluster_id]
            colors.extend([cluster_color] * len(cell_ids))

    dp = ad.obs[dp_key]
    pt = ad.obs[pt_key]

    lineage_pt = list(pt.loc[lineage_cell_ids])
    lineage_dp = list(dp.loc[lineage_cell_ids])

    # Display
    plt.figure(figsize=figsize)
    plt.scatter(lineage_pt, lineage_dp, s=1, c=colors, **kwargs)

    # Remove the right and top axes
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Set labels
    if show_label:
        ax.set_xlabel("Pseudotime")
        ax.set_ylabel("Differentiation Potential")

    # Save
    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)
    plt.show()


def plot_de_comparison(
    ad,
    id1,
    id2,
    comm_key="metric_clusters",
    groupby_key="clusters",
    n_genes=25,
    rank_kwargs={},
    **kwargs,
):
    comms = ad.obs[comm_key]
    clusters = np.unique(comms)

    # Checks!
    if id1 not in clusters:
        raise ValueError(f"Cluster {id1} not found in the parent anndata object")

    if id2 not in clusters:
        raise ValueError(f"Cluster {id2} not found in the parent anndata object")

    # Aggregate cells from the 2 clusters
    cells_id1 = list(comms.index[comms == id1])
    cells_id2 = list(comms.index[comms == id2])

    cell_ids = cells_id1 + cells_id2
    cells_gene_expr = ad.to_df().loc[cell_ids, :]

    # New anndata object
    ad2 = sc.AnnData(cells_gene_expr)
    ad2.obs_names = cell_ids
    ad2.var_names = ad.var_names

    clusters = pd.Series(index=ad2.obs_names)
    clusters.loc[cells_id1] = str(id1)
    clusters.loc[cells_id2] = str(id2)

    ad2.obs[groupby_key] = clusters.astype("category")

    # DE analysis
    key1 = f"{id1}v{id2}"
    key2 = f"{id2}v{id1}"
    sc.tl.rank_genes_groups(
        ad2,
        groupby_key,
        method="wilcoxon",
        key_added=key1,
        reference=str(id2),
        **rank_kwargs,
    )
    sc.tl.rank_genes_groups(
        ad2,
        groupby_key,
        method="wilcoxon",
        key_added=key2,
        reference=str(id1),
        **rank_kwargs,
    )

    vars1 = [gene_id[0] for gene_id in ad2.uns[key1]["names"]][:n_genes]
    vars2 = [gene_id[0] for gene_id in ad2.uns[key2]["names"]][:n_genes]

    vars = {str(id1): vars1, str(id2): vars2}

    # Heatmap
    sc.pl.heatmap(
        ad2,
        var_names=vars,
        groupby=groupby_key,
        standard_scale="var",
        show_gene_labels=True,
        **kwargs,
    )
