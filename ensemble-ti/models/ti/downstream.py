import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as ss

from utils.util import get_start_cell_cluster_id


def get_terminal_states(ad, start_cell_ids, use_rep='metric_embedding', cluster_key='metric_clusters', graph_key='metric_trajectory'):
    # Check 1: Input must be in AnnData format
    assert isinstance(ad, sc.AnnData)
 
    # Check 2: All keys must be present
    if use_rep not in ad.obsm_keys():
        raise ValueError(f'Representation `{use_rep}` not present in ad.obsm.')
    if cluster_key not in ad.obs_keys():
        raise ValueError(f'Cluster key `{cluster_key}` not present in ad.obs.')
    if graph_key not in ad.uns_keys():
        raise ValueError(f'Graph Key `{graph_key}` not present in ad.uns.')

    communities = ad.obs[cluster_key]
    X = pd.DataFrame(ad.obsm[use_rep], index=communities.index)
    g = ad.uns[graph_key]
    start_cluster_ids = set(get_start_cell_cluster_id(ad, start_cell_ids, communities))

    # Find clusters with no outgoing edges
    adj_g = nx.convert_matrix.to_numpy_array(g)
    nodes_g = np.array(nx.nodes(g))
    terminal_candidates_1 = set(nodes_g[np.sum(adj_g, axis=1) == 0])

    # Find clusters with maximal embedding components (as done in Palantir)
    max_ids = X.idxmax(axis=0)
    terminal_candidates_2 = set(communities.loc[max_ids])
    
    # Compute betweeness of second set of candidates and exclude
    # clusters with low betweenness (based on MAD)
    betweenness = nx.betweenness_centrality(g)
    mad_betweenness = ss.median_absolute_deviation(list(betweenness.values()))
    median_betweenness = np.median(list(betweenness.values()))
    threshold = median_betweenness - 3 * mad_betweenness
    threshold = threshold if threshold > 0 else 0

    terminal_candidates_2 = set(c for c in terminal_candidates_2 if betweenness[c] < threshold)
    terminal_candidates = terminal_candidates_1.union(terminal_candidates_2)

    # Remove starting clusters
    terminal_candidates = terminal_candidates - start_cluster_ids
    
    # Remove clusters with no incoming edges which are not start_clusters
    islands = set(nodes_g[np.sum(adj_g, axis=0) == 0]) - start_cluster_ids
    terminal_candidates = terminal_candidates - islands

    ad.uns['metric_terminal_clusters'] = terminal_candidates
    return terminal_candidates


def compute_cluster_lineage_likelihoods(ad, cluster_key='metric_clusters', terminal_key='metric_terminal_clusters', graph_key='metric_trajectory'):
    communities = ad.obs[cluster_key]
    cluster_ids = np.unique(communities)
    terminal_ids = ad.uns[terminal_key]
    g = ad.uns[graph_key]

    # TODO: Should this adjacency matrix be normalized?
    adj_g = nx.convert_matrix.to_numpy_array(g)

    cluster_lineage_likelihoods = pd.DataFrame(np.zeros((len(cluster_ids), len(terminal_ids))), columns=terminal_ids, index=cluster_ids)
    for t_id in terminal_ids:
        for c_id in cluster_ids:
            # All terminal states end up in that state
            if c_id == t_id:
                cluster_lineage_likelihoods.loc[c_id, t_id] = 1.0
                continue

            # Compute total likelihood along all possible paths
            paths = nx.all_simple_paths(g, c_id, t_id)
            likelihood = 0
            for path in paths:
                next_state = path[0]
                _l = 1
                for idx in range(1, len(path)):
                    _l *= adj_g[next_state, path[idx]]
                    next_state = path[idx]
                likelihood += _l
            cluster_lineage_likelihoods.loc[c_id, t_id] = likelihood

    # Row-Normalize the lineage likelihoods
    nz_inds = cluster_lineage_likelihoods.sum(axis=1) > 0
    cluster_lineage_likelihoods[nz_inds] = cluster_lineage_likelihoods[nz_inds].div(cluster_lineage_likelihoods[nz_inds].sum(axis=1), axis=0)

    return cluster_lineage_likelihoods


def compute_cell_branch_probs(ad, adj_dist, cluster_lineages, cluster_key='metric_clusters', graph_key='metric_trajectory'):
    communities = ad.obs[cluster_key]
    cluster_ids = np.unique(communities)
    n_clusters = len(cluster_ids)
    N = communities.shape[0]

    # Prune the distance graph
    g = ad.uns[graph_key]
    adj_g = nx.convert_matrix.to_numpy_array(g)
    adj_dist_pruned = _prune_network_edges(communities, adj_dist, adj_g)
    adj_dist_pruned = pd.DataFrame(adj_dist_pruned, index=communities.index, columns=communities.index)

    # Compute the cell to cluster connectivity
    cell_branch_probs = pd.DataFrame(np.zeros((N, n_clusters)), index=communities.index, columns=cluster_ids)
    for idx in communities.index:
        row = adj_dist_pruned.loc[idx, :]
        neighboring_clus = communities[np.where(row > 0)[0]]

        for clus_i in set(neighboring_clus):
            num_clus_i = np.sum(row.loc[neighboring_clus.index[np.where(neighboring_clus == clus_i)[0]]])
            w_i = num_clus_i / np.sum(row)
            cell_branch_probs.loc[idx, clus_i] = w_i

    # Project onto cluster lineage probabilities
    cell_branch_probs = cell_branch_probs.dot(cluster_lineages)
    return cell_branch_probs


def _prune_network_edges(communities, adj_sc, adj_cluster):
    n_communities = np.unique(communities).shape[0]
    n_pruned = 0

    # Create cluster index
    clusters = []
    for idx in range(n_communities):
        cluster_idx = communities == idx
        clusters.append(cluster_idx)

    n_row, n_col = adj_cluster.shape
    col_ids = np.arange(n_col)
    for c_idx in range(n_row):
        cluster_i = clusters[c_idx]
        non_connected_clusters = col_ids[adj_cluster[c_idx] == 0]
        for nc_idx in non_connected_clusters:
            if nc_idx == c_idx:
                continue
            cluster_nc = clusters[nc_idx]
            n_pruned += np.sum(adj_sc[cluster_i, :][:, cluster_nc] > 0)
            adj_sc[cluster_i, :][:, cluster_nc] = np.zeros_like(adj_sc[cluster_i, :][:, cluster_nc]).squeeze()

    print(f'Successfully pruned {n_pruned} edges')
    return adj_sc
