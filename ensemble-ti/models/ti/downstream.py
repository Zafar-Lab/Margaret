import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as ss

from utils.util import get_start_cell_cluster_id, prune_network_edges


def get_terminal_states(
    ad,
    adj_g,
    start_cell_ids,
    use_rep="metric_embedding",
    cluster_key="metric_clusters",
    mad_multiplier=1.0,
):
    # Check 1: Input must be in AnnData format
    assert isinstance(ad, sc.AnnData)

    # Check 2: All keys must be present
    if use_rep not in ad.obsm_keys():
        raise ValueError(f"Representation `{use_rep}` not present in ad.obsm.")
    if cluster_key not in ad.obs_keys():
        raise ValueError(f"Cluster key `{cluster_key}` not present in ad.obs.")

    communities = ad.obs[cluster_key]
    X = pd.DataFrame(ad.obsm[use_rep], index=communities.index)

    # adj_g will represent connectivities. For computing betweenness we
    # need to account for distances, so invert. (Assuming a directed graph)
    adj_g = 1 / adj_g
    adj_g[adj_g == np.inf] = 0
    g = nx.from_pandas_adjacency(adj_g, create_using=nx.DiGraph)
    start_cluster_ids = set(get_start_cell_cluster_id(ad, start_cell_ids, communities))

    # Find clusters with no outgoing edges (Candidate 1)
    nodes_g = np.array(nx.nodes(g))
    terminal_candidates_1 = set(nodes_g[np.sum(adj_g, axis=1) == 0])
    print(f"Terminal cluster candidate 1: {terminal_candidates_1}")

    # Compute betweeness of second set of candidates and exclude
    # clusters with low betweenness (based on MAD)
    terminal_candidates_2 = set(np.unique(communities))
    betweenness = pd.DataFrame(
        nx.betweenness_centrality(g).values(), index=np.unique(communities)
    )
    betweenness = betweenness / betweenness.sum()
    mad_betweenness = ss.median_absolute_deviation(betweenness.to_numpy())
    median_betweenness = betweenness.median()
    threshold = (median_betweenness - mad_multiplier * mad_betweenness).to_numpy()
    threshold = threshold if threshold > 0 else 0

    terminal_candidates_2 = set(
        c for c in terminal_candidates_2 if betweenness.loc[c, 0] < threshold
    )
    print(f"Terminal cluster candidate 2: {terminal_candidates_2}")

    terminal_candidates = terminal_candidates_1.union(terminal_candidates_2)

    # Remove starting clusters
    terminal_candidates = terminal_candidates - start_cluster_ids

    # Remove clusters with no incoming edges which are not start_clusters
    islands = set(nodes_g[np.sum(adj_g, axis=0) == 0]) - start_cluster_ids
    terminal_candidates = terminal_candidates - islands

    # convert candidate set to list as sets cant be serialized in anndata objects
    ad.uns["metric_terminal_clusters"] = list(terminal_candidates)
    print(f"Terminal clusters: {terminal_candidates}")
    return terminal_candidates


def compute_cluster_lineage_likelihoods(
    ad, adj_g, cluster_key="metric_clusters", terminal_key="metric_terminal_clusters"
):
    communities = ad.obs[cluster_key]
    cluster_ids = np.unique(communities)
    terminal_ids = ad.uns[terminal_key]
    g = nx.from_pandas_adjacency(adj_g, create_using=nx.DiGraph)
    cluster_lineage_likelihoods = pd.DataFrame(
        np.zeros((len(cluster_ids), len(terminal_ids))),
        columns=terminal_ids,
        index=cluster_ids,
    )

    for t_id in terminal_ids:
        for c_id in cluster_ids:
            # All terminal states end up in that state with prob 1.0
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
                    _l *= adj_g.loc[next_state, path[idx]]
                    next_state = path[idx]
                likelihood += _l
            cluster_lineage_likelihoods.loc[c_id, t_id] = likelihood

    # Row-Normalize the lineage likelihoods
    nz_inds = cluster_lineage_likelihoods.sum(axis=1) > 0
    cluster_lineage_likelihoods[nz_inds] = cluster_lineage_likelihoods[nz_inds].div(
        cluster_lineage_likelihoods[nz_inds].sum(axis=1), axis=0
    )

    return cluster_lineage_likelihoods


def compute_cell_branch_probs(
    ad, adj_g, adj_dist, cluster_lineages, cluster_key="metric_clusters"
):
    communities = ad.obs[cluster_key]
    cluster_ids = np.unique(communities)
    n_clusters = len(cluster_ids)
    N = communities.shape[0]

    # Prune the distance graph
    adj_dist = pd.DataFrame(adj_dist, index=ad.obs_names, columns=ad.obs_names)
    adj_dist_pruned = prune_network_edges(communities, adj_dist, adj_g)

    # Compute the cell to cluster connectivity
    cell_branch_probs = pd.DataFrame(
        np.zeros((N, n_clusters)), index=communities.index, columns=cluster_ids
    )
    for idx in communities.index:
        row = adj_dist_pruned.loc[idx, :]
        neighboring_clus = communities[np.where(row > 0)[0]]

        for clus_i in set(neighboring_clus):
            num_clus_i = np.sum(
                row.loc[neighboring_clus.index[np.where(neighboring_clus == clus_i)[0]]]
            )
            w_i = num_clus_i / np.sum(row)
            cell_branch_probs.loc[idx, clus_i] = w_i

    # Project onto cluster lineage probabilities
    cell_branch_probs = cell_branch_probs.dot(cluster_lineages)
    return cell_branch_probs
