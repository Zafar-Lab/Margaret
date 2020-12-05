import pandas as pd
import numpy as np
import scanpy as sc

from copy import deepcopy
from scipy.sparse.csgraph import dijkstra
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors


# Module that implements the psuedotime computation
# using the waypoint approach suggested in Palantir
def compute_trajectory(
    adata, early_cell_label,
    n_neighbors=30, n_waypoints=500, max_iterations=10,
    obsm_key='X_diffusion'
):
    """Computes Pseudotime for a given dataset

    Args:
        adata ([sc.AnnData]): Annotated data
        early_cell_label ([str]): Label of the early cell
        n_neighbors (int, optional): Number of neighbors when computing shortest path graph. Defaults to 30.
        n_waypoints (int, optional): Total number of waypoints. Defaults to 500.
        max_iterations (int, optional): Max number of iterations. Defaults to 10.
        obsm_key (str, optional): Key which contains latent space embeddings. Defaults to 'X_diffusion'.
    """
    if not isinstance(adata, sc.AnnData):
        raise ValueError('adata must be of type sc.AnnData')
    try:
        X = adata.obsm[obsm_key]
    except KeyError:
        raise Exception(f'Key {obsm_key} not found in adata')

    X_df = pd.DataFrame(X, index=adata.to_df().index)

    # Get starting cell
    start_cell_idx = get_starting_cell(X_df, early_cell_label)

    # Compute waypoints
    waypoint_set = get_waypoints(X, n_waypoints=n_waypoints)

    # Compute pseudotime
    pseudotime, W = compute_pseudotime(
        X, start_cell_idx, waypoint_set,
        n_neighbors=n_neighbors, max_iterations=max_iterations
    )
    adata.obsm['X_pseudotime'] = pseudotime
    adata.uns['waypoint_weights'] = W


def get_starting_cell(data_df, cell_label):
    """Returns the id of the boundary cell closest to the user-specified starting cell"""
    if not isinstance(data_df, pd.DataFrame):
        raise ValueError('data_df must be of type pd.DataFrame')

    # Index of the early cell
    early_cell_idx = np.where(data_df.index == cell_label)[0][0]
    
    # Find the cell ids with max and min components
    X = data_df.to_numpy()
    min_ids = np.argmin(X, axis=0)
    max_ids = np.argmax(X, axis=0)
    boundary_ids = np.union1d(min_ids, max_ids)

    # Compute distances of the user-defined cell with boundary cells
    # and return the boundary cell closest to it.
    dists = np.linalg.norm(X[boundary_ids, :] - X[early_cell_idx, :], ord=2, axis=1)
    start_cell_idx = boundary_ids[np.argmin(dists)]
    return start_cell_idx


def get_waypoints(X, n_waypoints=50):
    """Returns a set of waypoint indices in embedding space
    for pseudo time computation. Uses the max-min sampling
    method as described in Palantir.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError('X must be of type np.ndarray')

    N = X.shape[0]
    n_components = X.shape[-1]
    n_waypoints_per_comp = n_waypoints / n_components
    waypoints = set()

    # Sampling over individual components
    for comp_idx in range(n_components):
        comp_data = np.ravel(X[:, comp_idx])

        # Sample a random starting waypoint along this component
        start_idx = np.random.randint(0, high=N)
        comp_wp_set = set([start_idx])

        dists = np.zeros((N, n_waypoints))
        dists[:, 0] = np.abs(comp_data - comp_data[start_idx])

        # Iteratively add waypoints
        for wp_idx in range(1, n_waypoints_per_comp):
            # Compute the min distances for all cells and add the max to WPs
            min_dists = np.min(dists[:, 0:wp_idx], axis=1)
            max_dist_idx = np.argmax(min_dists)
            comp_wp_set.add(max_dist_idx)

            # Update the distances
            dists[:, wp_idx] = np.abs(comp_data - comp_data[max_dist_idx])
        waypoints = waypoints.union(comp_wp_set)
    return waypoints


def compute_pseudotime(X, start_cell_idx, waypoints, n_neighbors=30, max_iterations=10):
    """Computes Pseudotime based on the method suggested in Palantir
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(X)
    adj = nbrs.kneighbors_graph(X, mode="distance")

    # Compute distances of waypoints from cells
    D = np.zeros((len(waypoints), N))
    start_cell_id = None
    for idx, wp_idx in enumerate(waypoints):
        if wp_idx == start_cell_idx:
            start_cell_id = idx
        D[idx, :] = dijkstra(adj, directed=False, indices=wp_idx)

    # Normalized Waypoint weights
    # NOTE: In the paper there is a typo in the supplementary methods
    # In the paper they have normalized over all data points
    sdv = np.std(np.ravel(D)) * 1.06 * len(np.ravel(D)) ** (-1 / 5)
    W = np.exp(-0.5 * np.power((D / sdv), 2))
    col_sum = np.sum(W, axis=0)
    W = W / col_sum

    # Computation Loop
    pseudotime = D[start_cell_id, :]
    is_converged = False
    iter_count = 0
    while not is_converged and iter_count < max_iterations:
        # The perspective matrix
        P = deepcopy(D)
        for idx, wp_idx in enumerate(waypoints):
            dist_wp = pseudotime[wp_idx]
            neg_inds = pseudotime < dist_wp
            # NOTE: This update step is also different
            # than what has been described in the paper
            P[idx, neg_inds] = - D[idx, neg_inds]
            P[idx, :] = P[idx, :] + dist_wp
        pseudotime_ = np.sum(np.multiply(P, W), axis=0)

        # Check for convergence
        corr = pearsonr(pseudotime, pseudotime_)[0]
        print(f"Correlation at iteration {iter_count}: {corr}")
        if corr > 0.9999:
            is_converged = True

        # If not converged, continue iteration
        pseudotime = pseudotime_
        iter_count += 1
    return pseudotime, W
