import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as ss


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