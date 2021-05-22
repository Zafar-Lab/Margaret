import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as ss

from scipy.sparse.csgraph import dijkstra
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
    ad,
    adj_g,
    cluster_key="metric_clusters",
    terminal_key="metric_terminal_clusters",
    norm=False,
):
    communities = ad.obs[cluster_key]
    cluster_ids = np.unique(communities)
    terminal_ids = ad.uns[terminal_key]
    cll = pd.DataFrame(
        np.zeros((len(cluster_ids), len(terminal_ids))),
        columns=terminal_ids,
        index=cluster_ids,
    )

    # Create Directed Graph from adj matrix
    g = nx.from_pandas_adjacency(adj_g, create_using=nx.DiGraph)

    for t_id in terminal_ids:
        for c_id in cluster_ids:
            # All terminal states end up in that state with prob 1.0
            if c_id == t_id:
                cll.loc[c_id, t_id] = 1.0
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
            cll.loc[c_id, t_id] = likelihood

    # Row-Normalize the lineage likelihoods
    if norm:
        nz_inds = cll.sum(axis=1) > 0
        cll[nz_inds] = cll[nz_inds].div(cll[nz_inds].sum(axis=1), axis=0)

    return cll


def _sample_cluster_waypoints(X, g, cell_ids, n_waypoints=10, scheme="kmpp"):
    X_cluster = X.loc[cell_ids, :]
    cluster_index = X.index[cell_ids]
    N = X_cluster.shape[0]

    wps = []
    cached_dists = {}

    if scheme == "kmpp":
        if n_waypoints > N:
            return None

        # Sample the first waypoint randomly
        id = np.random.randint(0, N)
        wps.append(cluster_index[id])
        n_sampled = 0

        # Sample the remaining waypoints
        while True:
            dist = pd.DataFrame(
                np.zeros((N, len(wps))), index=cluster_index, columns=wps
            )
            for wp in wps:
                # If the dist with a wp is precomputed use it
                if wp in cached_dists.keys():
                    dist.loc[:, wp] = cached_dists[wp]
                    continue

                # Else Compute the shortest path distance and cache
                wp_id = np.where(cluster_index == wp)[0][0]
                d = dijkstra(g.to_numpy(), directed=True, indices=wp_id)
                dist.loc[:, wp] = d
                cached_dists[wp] = d

            # Exit if desired n_waypoints have been sampled
            if n_sampled == n_waypoints - 1:
                break

            # Otherwise find the next waypoint
            # Find the min_dist of the datapoints with existing centroids
            min_dist = dist.min(axis=1)

            # New waypoint will be the max of min distances
            new_wp_id = min_dist.idxmax()
            wps.append(new_wp_id)
            n_sampled += 1

        return wps, cached_dists

    if scheme == "random":
        raise NotImplementedError("The option `random` has not been implemented yet")


def sample_waypoints(
    ad,
    adj_dist,
    cluster_key="metric_clusters",
    embedding_key="metric_embedding",
    n_waypoints=10,
    scheme="kmpp",
):
    X = pd.DataFrame(ad.obsm[embedding_key], index=ad.obs_names)
    clusters = ad.obs[cluster_key]

    adj_dist = pd.DataFrame(adj_dist, index=ad.obs_names, columns=ad.obs_names)
    labels = np.unique(clusters)
    wps = list()
    dists = pd.DataFrame(index=ad.obs_names)

    for cluster_id in labels:
        # Sample waypoints for a cluster at a time
        cell_ids = clusters == cluster_id
        g = adj_dist.loc[cell_ids, cell_ids]
        res = _sample_cluster_waypoints(
            X, g, cell_ids, n_waypoints=n_waypoints, scheme=scheme
        )
        # res will be None when a cluster has less points than the n_waypoints
        # This behavior is subject to change in the future.
        if res is None:
            continue

        w_, d_ = res
        wps.extend(w_)

        for k, v in d_.items():
            dists.loc[cell_ids, k] = v

    return dists.fillna(0), wps


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
