import numpy as np
import pandas as pd
import scipy.stats as stats

from scipy.sparse.csgraph import dijkstra
from sklearn.metrics import pairwise_distances
from utils.util import get_start_cell_cluster_id, compute_runtime


@compute_runtime
def compute_pseudotime(ad, start_cell_ids, adj_conn, adj_dist, connectivities, comm_key='communities', data_key='X_metric', max_iter=10):
    communities = ad.obs[comm_key]
    n_communities = np.unique(communities).shape[0]
    base_inds = np.arange(n_communities)
    N = communities.shape[0]
    data = pd.DataFrame(ad.obsm[data_key], index=communities.index)

    adj_conn = pd.DataFrame(adj_conn.todense(), index=communities.index, columns=communities.index)
    adj_dist = pd.DataFrame(adj_dist.todense(), index=communities.index, columns=communities.index)

    # Create cluster index
    clusters = []
    for idx in range(n_communities):
        cluster_idx = communities == idx
        clusters.append(cluster_idx)

    pseudotime = pd.Series(np.zeros((N,)), index=communities.index)
    
    # Convergence params
    prev_pseudotime = pseudotime
    is_converged = False
    iter = 0

    while not is_converged and iter < max_iter:
        computed = []
        for s in start_cell_ids:
            # Get the start cluster id
            sc_id = list(get_start_cell_cluster_id(ad, [s], communities))[0]

            # Compute shortest path distances within the start cluster
            adj_sc = adj_dist.loc[clusters[sc_id], clusters[sc_id]]
            adj_sc = _connect_graph(adj_sc, data.loc[clusters[sc_id], :], np.where(adj_sc.index == s)[0][0])
            dists_sc = dijkstra(adj_sc, indices=np.where(adj_sc.index == s)[0][0])
            pseudotime[clusters[sc_id]] = dists_sc
            computed.append(sc_id)

            inds = base_inds[connectivities[sc_id, :] > 0]
            remaining_clusters = []
            remaining_clusters.extend(inds)

            while remaining_clusters != []:
                c = remaining_clusters.pop(0)
                # Compute pseudotime for c
                in_neighbors = base_inds[connectivities[:, c] > 0]
                c_pseudotimes = []
                for n in in_neighbors:
                    if n not in computed:
                        continue
                    # Find the early cell for this incoming cluster
                    adj_conn_nc = adj_conn.loc[clusters[n], clusters[c]]
                    adj_dist_nc = adj_dist.loc[clusters[n], clusters[c]]
                    max_neigh_cell_id = adj_conn_nc.index[np.argmax(np.sum(adj_conn_nc, axis=1))]
                    candidates = adj_dist_nc.loc[max_neigh_cell_id, :]
                    candidates = candidates[candidates > 0]
                    early_cell_id = candidates.index[np.argmin(candidates)]

                    adj_cc = adj_dist.loc[clusters[c], clusters[c]]
                    adj_cc = _connect_graph(adj_cc, data.loc[clusters[c], :], np.where(adj_cc.index == early_cell_id)[0][0])
                    pseudotime_n = dijkstra(adj_cc, indices=np.where(adj_cc.index == early_cell_id)[0][0])
                    pseudotime_n += (pseudotime.loc[max_neigh_cell_id] + adj_dist_nc.loc[max_neigh_cell_id, early_cell_id])
                    c_pseudotimes.append(pseudotime_n)
                pseudotime[clusters[c]] = np.min(np.stack(c_pseudotimes), axis=0)

                # Book-keeping stuff!
                computed.append(c)

                # Find outgoing neighbors of c and add to stack
                inds = base_inds[connectivities[c, :] > 0]
                remaining_clusters.extend(inds)
        corr = stats.pearsonr(pseudotime, prev_pseudotime)[0]
        if corr > 0.999:
            is_converged = True
        prev_pseudotime = pseudotime
        iter += 1
    return pseudotime


def _connect_graph(adj, data, start_cell_id):
    index = adj.index
    dists = dijkstra(adj, indices=start_cell_id)
    unreachable_nodes = index[dists == np.inf]
    if len(unreachable_nodes) == 0:
        return adj

    # Connect unreachable nodes
    while len(unreachable_nodes) > 0:
        farthest_reachable_id = index[np.argmax(dists[dists != np.inf])]

        # Compute distances to unreachable nodes
        unreachable_dists = pairwise_distances(
            data.loc[farthest_reachable_id, :].values.reshape(1, -1),
            data.loc[unreachable_nodes, :],
        )
        unreachable_dists = pd.Series(
            np.ravel(unreachable_dists), index=unreachable_nodes
        )

        # Add edge between farthest reacheable and its nearest unreachable
        adj.loc[farthest_reachable_id, unreachable_dists.idxmin()] = unreachable_dists.min()

        # Recompute distances to early cell
        dists = dijkstra(adj, indices=start_cell_id)

        # Idenfity unreachable nodes
        unreachable_nodes = index[dists == np.inf]
    return adj


def prune_network_edges(communities, adj, connectivities):
    n_communities = np.unique(communities).shape[0]
    pruned_edges = 0

    # Create cluster index
    clusters = []
    for idx in range(n_communities):
        cluster_idx = communities == idx
        clusters.append(cluster_idx)

    n_row, n_col = connectivities.shape
    col_ids = np.arange(n_col)
    for cluster_idx in range(n_row):
        cluster_i = clusters[cluster_idx]
        non_connected_clusters = col_ids[connectivities[cluster_idx] == 0]
        for non_cluster_idx in non_connected_clusters:
            pruned_edges += np.sum(adj[cluster_i, non_cluster_idx])
            adj[cluster_i, non_cluster_idx] = np.zeros((adj[cluster_i, non_cluster_idx].shape[0], ))
    return pruned_edges
