import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as ss

from numpy.linalg import inv, pinv
from numpy.linalg.linalg import LinAlgError
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import find, csr_matrix
from scipy.stats import entropy
from scipy.sparse.csgraph import dijkstra

from models.ti.sim import compute_lpi, compute_lrw
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
    terminal_candidates_1 = set(adj_g.index[np.sum(adj_g, axis=1) == 0])
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
    islands = set(adj_g.index[np.sum(adj_g, axis=0) == 0]) - start_cluster_ids
    terminal_candidates = terminal_candidates - islands

    # convert candidate set to list as sets cant be serialized in anndata objects
    ad.uns["metric_terminal_clusters"] = list(terminal_candidates)
    print(f"Terminal clusters: {terminal_candidates}")
    return terminal_candidates


def get_terminal_cells(
    ad,
    terminal_keys="metric_terminal_clusters",
    cluster_key="metric_clusters",
    pt_key="metric_pseudotime_v2",
):
    t_cell_ids = []
    comms = ad.obs[cluster_key]
    pt = ad.obs[pt_key]
    for ts in ad.uns[terminal_keys]:
        # Find the terminal cell within that cluster
        t_ids = comms == ts
        t_pt = pt.loc[t_ids]

        # The terminal cell is the cell with the max pseudotime within a terminal cluster
        t_cell_ids.append(t_pt.idxmax())

    ad.uns["metric_terminal_cells"] = t_cell_ids

    return t_cell_ids


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
    exclude_clusters=None,
):
    X = pd.DataFrame(ad.obsm[embedding_key], index=ad.obs_names)
    clusters = ad.obs[cluster_key]

    adj_dist = pd.DataFrame(adj_dist, index=ad.obs_names, columns=ad.obs_names)
    labels = np.unique(clusters)
    wps = list()
    dists = pd.DataFrame(index=ad.obs_names)

    for cluster_id in labels:
        # Skip if the cluster id is excluded
        if exclude_clusters is not None and cluster_id in exclude_clusters:
            print(f"Excluding waypoint computation for cluster: {cluster_id}")
            continue
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

    # Add the waypoint to the annotated data object
    ad.uns["metric_waypoints"] = wps

    return dists.fillna(0), wps


# NOTE: This method for computing cell branch probs is now obsolete
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


# NOTE: Code credits: https://github.com/dpeerlab/Palantir/
def _construct_markov_chain(
    wp_data,
    knn,
    pseudotime,
    comms,
    adj_cluster,
    std_factor=1.0,
    prune_wp_graph=True,
    n_jobs=1,
):

    # Markov chain construction
    print("Markov chain construction...")
    waypoints = wp_data.index

    # kNN graph
    n_neighbors = knn
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, metric="euclidean", n_jobs=n_jobs
    ).fit(wp_data)
    kNN = nbrs.kneighbors_graph(wp_data, mode="distance")
    dist, ind = nbrs.kneighbors(wp_data)

    # Prune the kNN graph wrt to the undirected graph
    if prune_wp_graph:
        wp = wp_data.index
        kNN_pruned = pd.DataFrame(kNN.todense(), index=wp, columns=wp)
        kNN_pruned = prune_network_edges(comms.loc[wp], kNN_pruned, adj_cluster)
        kNN = csr_matrix(kNN_pruned)

    # Standard deviation allowing for "back" edges
    adpative_k = np.min([int(np.floor(n_neighbors / 3)) - 1, 30])
    adaptive_std = np.ravel(dist[:, adpative_k])

    # Directed graph construction
    # pseudotime position of all the neighbors
    traj_nbrs = pd.DataFrame(
        pseudotime[np.ravel(waypoints.values[ind])].values.reshape(
            [len(waypoints), n_neighbors]
        ),
        index=waypoints,
    )

    # Remove edges that move backwards in pseudotime except for edges that are within
    # the computed standard deviation
    rem_edges = traj_nbrs.apply(
        lambda x: x < pseudotime[traj_nbrs.index] - std_factor * adaptive_std
    )
    rem_edges = rem_edges.stack()[rem_edges.stack()]

    # Determine the indices and update adjacency matrix
    cell_mapping = pd.Series(range(len(waypoints)), index=waypoints)
    x = list(cell_mapping[rem_edges.index.get_level_values(0)])
    y = list(rem_edges.index.get_level_values(1))

    # Update adjacecy matrix
    kNN[x, ind[x, y]] = 0

    # Affinity matrix and markov chain
    x, y, z = find(kNN)
    aff = np.exp(
        -(z ** 2) / (adaptive_std[x] ** 2) * 0.5
        - (z ** 2) / (adaptive_std[y] ** 2) * 0.5
    )
    W = csr_matrix((aff, (x, y)), [len(waypoints), len(waypoints)])

    # Transition matrix
    D = np.ravel(W.sum(axis=1))
    x, y, z = find(W)
    T = csr_matrix((z / D[x], (x, y)), [len(waypoints), len(waypoints)])

    return T


def _differentiation_entropy(
    wp_data,
    terminal_states,
    knn,
    pseudotime,
    comms,
    adj_cluster,
    std_factor=1.0,
    n_jobs=1,
    prune_wp_graph=True,
):

    T = _construct_markov_chain(
        wp_data,
        knn,
        pseudotime,
        comms,
        adj_cluster,
        std_factor=std_factor,
        n_jobs=n_jobs,
        prune_wp_graph=prune_wp_graph,
    )

    # Absorption states should not have outgoing edges
    waypoints = wp_data.index
    abs_states = np.where(waypoints.isin(terminal_states))[0]
    # Reset absorption state affinities by Removing neigbors
    T[abs_states, :] = 0
    # Diagnoals as 1s
    T[abs_states, abs_states] = 1

    # Fundamental matrix and absorption probabilities
    print("Computing fundamental matrix and absorption probabilities...")
    # Transition states
    trans_states = list(set(range(len(waypoints))).difference(abs_states))

    # Q matrix
    Q = T[trans_states, :][:, trans_states]
    # Fundamental matrix
    mat = np.eye(Q.shape[0]) - Q.todense()
    try:
        N = inv(mat)
    except LinAlgError:
        # Compute the pseudoinverse if the main inv cannot be computed
        N = pinv(mat)

    # Absorption probabilities
    branch_probs = np.dot(N, T[trans_states, :][:, abs_states].todense())
    branch_probs = pd.DataFrame(
        branch_probs, index=waypoints[trans_states], columns=waypoints[abs_states]
    )
    branch_probs[branch_probs < 0] = 0

    # Entropy
    ent = branch_probs.apply(entropy, axis=1)

    # Add terminal states
    ent = ent.append(pd.Series(0, index=terminal_states))
    bp = pd.DataFrame(0, index=terminal_states, columns=terminal_states)
    bp.values[range(len(terminal_states)), range(len(terminal_states))] = 1
    branch_probs = branch_probs.append(bp.loc[:, branch_probs.columns])

    return ent, branch_probs


def compute_diff_potential(
    ad,
    adj_dist,
    adj_cluster,
    comms_key="metric_clusters",
    embed_key="metric_embedding",
    pt_key="metric_pseudotime_v2",
    wp_key="metric_waypoints",
    tc_key="metric_terminal_cells",
    sim_scheme="lrw",
    sim_kwargs={},
    std_factor=1.0,
    knn=30,
    prune_wp_graph=True,
    exclude_clusters=None,
    n_jobs=1,
):
    wps = ad.uns[wp_key]

    # Add the terminal cells to the wp list
    t_cell_ids = ad.uns[tc_key]
    wps.extend(t_cell_ids)
    wp_ = set(wps)

    # Cell to Waypoint Connectivity
    print("Cell to Waypoint connectivity")
    communities = ad.obs[comms_key]
    adj_dist_pruned = prune_network_edges(
        communities,
        pd.DataFrame(adj_dist, index=ad.obs_names, columns=ad.obs_names),
        adj_cluster,
    )
    adj_dist_pruned[adj_dist_pruned == 0] = np.inf

    if exclude_clusters is not None:
        for cid in exclude_clusters:
            idx = ad.obs_names[communities == cid]
            adj_dist_pruned = adj_dist_pruned.drop(index=idx, columns=idx)
            adj_cluster = adj_cluster.drop(index=cid, columns=cid)

    # This represents the final connectivity Adj mat. between cells
    adj_conn = csr_matrix(np.exp(-np.power(adj_dist_pruned, 2)))

    if sim_scheme == "lpi":
        S = compute_lpi(adj_conn, **sim_kwargs)
    elif sim_scheme == "lrw":
        S = compute_lrw(adj_conn, **sim_kwargs)
    else:
        raise NotImplementedError(
            f"The scheme {sim_scheme} has not been implemented yet!"
        )

    S = pd.DataFrame(
        S.todense(), index=adj_dist_pruned.index, columns=adj_dist_pruned.index
    )

    # Waypoint to Terminal State connectivity
    print("Waypoint to Terminal State connectivity")
    X = pd.DataFrame(ad.obsm[embed_key], index=ad.obs_names)
    X_wp = X.loc[wp_, :]
    pt = ad.obs[pt_key]
    wp_comms = communities.loc[wp_]
    _, bp = _differentiation_entropy(
        X_wp,
        t_cell_ids,
        knn,
        pt,
        wp_comms,
        adj_cluster,
        std_factor=std_factor,
        prune_wp_graph=prune_wp_graph,
        n_jobs=n_jobs,
    )

    # Project branch probs on the cells
    bps_ = S.loc[:, wp_].loc[:, bp.index].dot(bp)
    bps_ = bps_.div(bps_.sum(axis=1), axis=0)

    # We perform the assignment in this way to account for
    # clusters that might have been excluded.
    bps = pd.DataFrame(index=ad.obs_names, columns=bps_.columns)
    bps.loc[bps_.index, :] = bps_

    # Compute row-wise entropy to compute DP
    ent_ = ss.entropy(bps_, base=2, axis=1)
    ent = pd.Series(index=ad.obs_names)
    ent.loc[adj_dist_pruned.index] = ent_

    # Add to the anndata object
    ad.obsm["metric_branch_probs"] = bps
    ad.obs["metric_dp"] = ent

    return ent, bps
