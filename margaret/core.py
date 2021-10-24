import networkx as nx
import numpy as np
import os
import palantir
import random
import scanpy as sc
import torch
import warnings

from sklearn.neighbors import NearestNeighbors

from models.ti.connectivity import (
    compute_directed_cluster_connectivity,
    compute_undirected_cluster_connectivity,
)
from models.ti.graph import (
    compute_gt_milestone_network,
    compute_connectivity_graph,
    compute_trajectory_graph,
    compute_trajectory_graph_v2,
)
from models.ti.pseudotime import compute_pseudotime as cp
from models.ti.pseudotime_v2 import compute_pseudotime as cp2
from train_metric import train_metric_learner
from utils.plot import generate_plot_embeddings
from utils.util import get_start_cell_cluster_id, determine_cell_clusters


def seed_everything(seed=0):
    # NOTE: Uses the implementation from Pytorch Lightning
    """Function that sets seed for pseudo-random number generators  in:
    pytorch, numpy, python.random
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if (seed > max_seed_value) or (seed < min_seed_value):
        raise ValueError(
            f"{seed} is not in bounds, \
            numpy accepts from {min_seed_value} to {max_seed_value}"
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def run_metti(
    ad,
    n_episodes=10,
    n_metric_epochs=10,
    use_rep="X_pca",
    code_size=10,
    c_backend="louvain",
    chkpt_save_path=os.getcwd(),
    random_state=0,
    cluster_kwargs={},
    neighbor_kwargs={},
    trainer_kwargs={},
    viz_method="umap",
    viz_kwargs={},
    n_neighbors_ti=30,
    threshold=0.5,
    device="cuda",
):
    # Seed setting
    seed_everything(seed=random_state)

    # In case we need to reference self data using keys
    ad.obsm["X"] = ad.X

    # Dimensionality reduction and clustering
    print("Computing embeddings and clusters...")
    with warnings.catch_warnings():
        # Filter out user warnings from PyTorch about saving scheduler state
        warnings.simplefilter("ignore")
        train_metric_learner(
            ad,
            n_episodes=n_episodes,
            n_metric_epochs=n_metric_epochs,
            obsm_data_key=use_rep,
            code_size=code_size,
            backend=c_backend,
            save_path=chkpt_save_path,
            cluster_kwargs=cluster_kwargs,
            nn_kwargs=neighbor_kwargs,
            trainer_kwargs=trainer_kwargs,
            device=device,
        )

    # 2d embedding visualization generation
    print("\nComputing 2d embedding visualizations...")
    ad.obsm["metric_viz_embedding"] = generate_plot_embeddings(
        ad.obsm["metric_embedding"],
        method=viz_method,
        random_state=random_state,
        **viz_kwargs,
    )

    # TI
    print("\nComputing trajectory graphs...")
    communities = ad.obs["metric_clusters"].to_numpy().astype(np.int)
    X = ad.obsm["metric_embedding"]

    n_neighbors = n_neighbors_ti
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(X)
    adj_dist = nbrs.kneighbors_graph(X, mode="distance")
    adj_conn = nbrs.kneighbors_graph(X)

    connectivity, scores = compute_directed_cluster_connectivity(
        communities, adj_conn, threshold=threshold
    )
    un_connectivity, un_scores = compute_undirected_cluster_connectivity(
        communities, adj_conn, threshold=threshold
    )

    ad.uns["metric_directed_connectivities"] = connectivity
    ad.uns["metric_undirected_connectivities"] = un_connectivity
    ad.uns["metric_directed_scores"] = scores
    ad.uns["metric_undirected_scores"] = un_scores

    start_cell_ids = ad.uns["start_id"]
    start_cell_ids = (
        [start_cell_ids] if isinstance(start_cell_ids, str) else list(start_cell_ids)
    )
    start_cluster_ids = get_start_cell_cluster_id(ad, start_cell_ids, communities)
    g, node_positions = compute_trajectory_graph(
        ad.obsm["metric_viz_embedding"], communities, connectivity, start_cluster_ids
    )
    ad.uns["metric_trajectory"] = g
    ad.uns["metric_trajectory_node_positions"] = node_positions

    g, node_positions = compute_connectivity_graph(
        ad.obsm["metric_viz_embedding"], communities, un_connectivity, mode="undirected"
    )
    ad.uns["metric_undirected_graph"] = g
    ad.uns["metric_undirected_node_positions"] = node_positions

    # Pseudotime computation
    print("\nComputing Pseudotime...")
    trajectory_graph = nx.to_numpy_array(ad.uns["metric_trajectory"])
    pseudotime = cp(
        ad,
        start_cell_ids,
        adj_conn,
        adj_dist,
        trajectory_graph,
        comm_key="metric_clusters",
        data_key="metric_embedding",
    )


def run_metti_v2(
    ad,
    n_episodes=10,
    n_metric_epochs=10,
    use_rep="X_pca",
    code_size=10,
    c_backend="louvain",
    chkpt_save_path=os.getcwd(),
    random_state=0,
    cluster_kwargs={},
    neighbor_kwargs={},
    trainer_kwargs={},
    viz_method="umap",
    viz_kwargs={},
    n_neighbors_ti=30,
    threshold=0.5,
    device="cuda",
):
    # Seed setting
    seed_everything(seed=random_state)

    # In case we need to reference self data using keys
    ad.obsm["X"] = ad.X

    # Dimensionality reduction and clustering
    print("Computing embeddings and clusters...")
    with warnings.catch_warnings():
        # Filter out user warnings from PyTorch about saving scheduler state
        warnings.simplefilter("ignore")
        train_metric_learner(
            ad,
            n_episodes=n_episodes,
            n_metric_epochs=n_metric_epochs,
            obsm_data_key=use_rep,
            code_size=code_size,
            backend=c_backend,
            save_path=chkpt_save_path,
            cluster_kwargs=cluster_kwargs,
            nn_kwargs=neighbor_kwargs,
            trainer_kwargs=trainer_kwargs,
            device=device,
        )

    # 2d embedding visualization generation
    print("\nComputing 2d embedding visualizations...")
    ad.obsm["metric_viz_embedding"] = generate_plot_embeddings(
        ad.obsm["metric_embedding"],
        method=viz_method,
        random_state=random_state,
        **viz_kwargs,
    )

    # TI
    print("\nComputing Connectivity graphs...")
    communities = ad.obs["metric_clusters"].to_numpy().astype(np.int)
    X = ad.obsm["metric_embedding"]

    n_neighbors = n_neighbors_ti
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(X)
    adj_dist = nbrs.kneighbors_graph(X, mode="distance")
    adj_conn = nbrs.kneighbors_graph(X)

    un_connectivity, un_scores = compute_undirected_cluster_connectivity(
        communities, adj_conn, threshold=threshold
    )

    ad.uns["metric_undirected_connectivities"] = un_connectivity
    ad.uns["metric_undirected_scores"] = un_scores

    start_cell_ids = ad.uns["start_id"]
    start_cell_ids = (
        [start_cell_ids] if isinstance(start_cell_ids, str) else list(start_cell_ids)
    )
    start_cluster_ids = get_start_cell_cluster_id(ad, start_cell_ids, communities)

    g_undirected, node_positions = compute_connectivity_graph(
        ad.obsm["metric_viz_embedding"], communities, un_connectivity, mode="undirected"
    )
    ad.uns["metric_undirected_graph"] = g_undirected
    ad.uns["metric_undirected_node_positions"] = node_positions

    # Pseudotime computation
    print("\nComputing Pseudotime...")
    adj_cluster = nx.to_pandas_adjacency(g_undirected)
    pseudotime = cp2(ad, start_cell_ids, adj_dist, adj_cluster)

    # Directed graph computation
    g_directed = compute_trajectory_graph_v2(
        pseudotime, adj_cluster, ad.obs["metric_clusters"]
    )
    ad.uns["metric_trajectory"] = g_directed
    ad.uns["metric_trajectory_node_positions"] = node_positions


def run_paga(
    ad,
    start_cell,
    n_neighbors=15,
    use_rep="X_pca",
    c_backend="louvain",
    random_state=0,
    neighbor_kwargs={},
    cluster_kwargs={},
    paga_kwargs={},
):
    # Nearest neighbors
    sc.pp.neighbors(ad, **neighbor_kwargs)

    # Cluster generation
    if c_backend == "louvain":
        sc.tl.louvain(ad, key_added="paga_clusters", **cluster_kwargs)
    else:
        sc.tl.leiden(ad, key_added="paga_clusters", **cluster_kwargs)

    # PAGA
    sc.tl.paga(ad, groups="paga_clusters", **paga_kwargs)

    # Pseudotime computation using PAGA
    obs_ = ad.obs_names
    start_cell_id = np.where(obs_ == start_cell)[0][0]
    ad.uns["iroot"] = start_cell_id
    sc.tl.dpt(ad)


def run_palantir(ad, early_cell, knn=30):
    # PCA and diffusion Map computation with scaled eigenvectors
    pca_projections, _ = palantir.utils.run_pca(ad, use_hvg=False)
    dm_res = palantir.utils.run_diffusion_maps(pca_projections, n_components=10)
    ms_data = palantir.utils.determine_multiscale_space(dm_res)

    # Pseudotime and DP computation
    presults = palantir.core.run_palantir(ms_data, early_cell, knn=knn)
    return presults, ms_data
