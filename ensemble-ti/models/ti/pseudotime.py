import numpy as np
import pandas as pd
import scipy.stats as stats

from scipy.sparse.csgraph import dijkstra
from utils.util import get_start_cell_cluster_id, compute_runtime, connect_graph


@compute_runtime
def compute_pseudotime(
    ad,
    start_cell_ids,
    adj_conn,
    adj_dist,
    connectivities,
    comm_key="metric_clusters",
    data_key="metric_embnedding",
    max_iter=10,
):
    communities = ad.obs[comm_key]
    n_communities = np.unique(communities).shape[0]
    base_inds = np.arange(n_communities)
    N = communities.shape[0]
    data = pd.DataFrame(ad.obsm[data_key], index=communities.index)

    adj_conn = pd.DataFrame(
        adj_conn.todense(), index=communities.index, columns=communities.index
    )
    adj_dist = pd.DataFrame(
        adj_dist.todense(), index=communities.index, columns=communities.index
    )

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
            adj_sc = connect_graph(
                adj_sc, data.loc[clusters[sc_id], :], np.where(adj_sc.index == s)[0][0]
            )
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
                    max_neigh_cell_id = adj_conn_nc.index[
                        np.argmax(np.sum(adj_conn_nc, axis=1))
                    ]
                    candidates = adj_dist_nc.loc[max_neigh_cell_id, :]
                    candidates = candidates[candidates > 0]
                    early_cell_id = candidates.index[np.argmin(candidates)]

                    adj_cc = adj_dist.loc[clusters[c], clusters[c]]
                    adj_cc = connect_graph(
                        adj_cc,
                        data.loc[clusters[c], :],
                        np.where(adj_cc.index == early_cell_id)[0][0],
                    )
                    pseudotime_n = dijkstra(
                        adj_cc, indices=np.where(adj_cc.index == early_cell_id)[0][0]
                    )
                    pseudotime_n += (
                        pseudotime.loc[max_neigh_cell_id]
                        + adj_dist_nc.loc[max_neigh_cell_id, early_cell_id]
                    )
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

    # Add pseudotime to annotated data object
    ad.obs["metric_pseudotime"] = pseudotime
    return pseudotime
