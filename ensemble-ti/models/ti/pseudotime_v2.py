import numpy as np
import pandas as pd

from scipy.sparse.csgraph import dijkstra
from utils.util import (
    compute_runtime,
    prune_network_edges,
    connect_graph,
)


@compute_runtime
def compute_pseudotime(
    ad,
    start_cell_ids,
    adj_dist,
    adj_cluster,
    comm_key="metric_clusters",
    data_key="metric_embedding",
):
    communities = ad.obs[comm_key]
    cluster_ids = np.unique(communities)
    data = pd.DataFrame(ad.obsm[data_key], index=ad.obs_names)

    # Create cluster index
    clusters = {}
    for idx in cluster_ids:
        cluster_idx = communities == idx
        clusters[idx] = cluster_idx

    # Prune the initial adjacency matrix
    adj_dist = pd.DataFrame(
        adj_dist.todense(), index=ad.obs_names, columns=ad.obs_names
    )
    adj_dist_pruned = prune_network_edges(communities, adj_dist, adj_cluster)

    # Pseudotime computation on the pruned graph
    start_indices = [np.where(ad.obs_names == s)[0][0] for s in start_cell_ids]
    p = dijkstra(adj_dist_pruned.to_numpy(), indices=start_indices, min_only=True)
    pseudotime = pd.Series(p, index=ad.obs_names)

    for _, cluster in clusters.items():
        p_cluster = pseudotime.loc[cluster]
        cluster_start_cell = p_cluster.idxmin()
        adj_sc = adj_dist_pruned.loc[cluster, cluster]
        adj_sc = connect_graph(
            adj_sc,
            data.loc[cluster, :],
            np.where(adj_sc.index == cluster_start_cell)[0][0],
        )

        # Update the cluster graph with
        adj_dist_pruned.loc[cluster, cluster] = adj_sc

    # Recompute the pseudotime with the updated graph
    p = dijkstra(adj_dist_pruned, indices=start_indices, min_only=True)
    pseudotime = pd.Series(p, index=ad.obs_names)

    # Set the pseudotime for unreachable cells to 0
    pseudotime[pseudotime == np.inf] = 0

    # Add pseudotime to annotated data object
    ad.obs["metric_pseudotime_v2"] = pseudotime
    return pseudotime
