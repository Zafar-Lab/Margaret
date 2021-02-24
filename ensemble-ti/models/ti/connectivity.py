import numpy as np


def compute_undirected_cluster_connectivity(communities, adj, threshold=1.0):
    n_communities = np.unique(communities).shape[0]

    # Create cluster index
    clusters = []
    for idx in range(n_communities):
        cluster_idx = communities == idx
        clusters.append(cluster_idx)

    undirected_cluster_connectivity = np.zeros((n_communities, n_communities))
    undirected_z_score = np.zeros((n_communities, n_communities))
    cluster_outgoing_edges = np.zeros(n_communities)
    for i in range(n_communities):
        cluster_i = clusters[i]

        # Compute the outgoing edges from the ith cluster
        adj_i = adj[cluster_i, :]
        adj_ii = adj_i[:, cluster_i]
        e_i = np.sum(adj_i) - np.sum(adj_ii)
        n_i = np.sum(cluster_i)
        cluster_outgoing_edges[i] = e_i

        for j in range(n_communities):
            if i == j:
                continue
            # Compute the outgoing edges from the jth cluster
            cluster_j = clusters[j]
            adj_j = adj[cluster_j, :]
            adj_jj = adj_j[:, cluster_j]
            e_j = np.sum(adj_j) - np.sum(adj_jj)
            n_j = np.sum(cluster_j)

            # Compute the number of inter-edges from the ith to jth cluster
            adj_ij = adj_i[:, cluster_j]
            e_ij = np.sum(adj_ij)

            # Compute the number of inter-edges from the jth to ith cluster
            adj_ji = adj_j[:, cluster_i]
            e_ji = np.sum(adj_ji)
            e_sym = e_ij + e_ji

            # Compute the random assignment of edges from the ith to the jth
            # cluster under the PAGA binomial model
            e_sym_random = (e_i * n_j + e_j * n_i) / (N - 1)

            # Compute the cluster connectivity measure
            std_sym = (e_i * n_j * (N - n_j - 1) + e_j * n_i * (N - n_i - 1))/(N - 1)**2
            undirected_z_score[i][j] = (e_sym - e_sym_random) / std_sym

            # Only add non-spurious edges based on a threshold
            if undirected_z_score[i][j] >= threshold:
                undirected_cluster_connectivity[i][j] = (e_sym - e_sym_random) / (e_i + e_j - e_sym_random)
        return undirected_cluster_connectivity, undirected_z_score


def compute_directed_cluster_connectivity(communities, adj, threshold=1.0):
    N = communities.shape[0]
    n_communities = np.unique(communities).shape[0]

    # Create cluster index
    clusters = []
    for idx in range(n_communities):
        cluster_idx = communities == idx
        clusters.append(cluster_idx)

    directed_cluster_connectivity = np.zeros((n_communities, n_communities))
    directed_z_score = np.zeros((n_communities, n_communities))
    cluster_outgoing_edges = np.zeros(n_communities)
    for i in range(n_communities):
        cluster_i = clusters[i]

        # Compute the outgoing edges from the ith cluster
        adj_i = adj[cluster_i, :]
        adj_ii = adj_i[:, cluster_i]
        e_i = np.sum(adj_i) - np.sum(adj_ii)
        cluster_outgoing_edges[i] = e_i

        for j in range(n_communities):
            if i == j:
                continue
            # Compute the outgoing edges from the jth cluster
            cluster_j = clusters[j]
            n_j = np.sum(clusters[j])

            # Compute the number of inter-edges from the ith to jth cluster
            adj_ij = adj_i[:, cluster_j]
            e_ij = np.sum(adj_ij)

            # Compute the random assignment of edges from the ith to the jth
            # cluster under the PAGA binomial model
            e_ij_random = (e_i * n_j) / (N - 1)

            # Compute the cluster connectivity measure
            std_j = e_i * n_j * (N - n_j - 1)/(N - 1)**2
            directed_z_score[i][j] = (e_ij - e_ij_random)/std_j

            # Only add non-spurious edges with 95% CI
            if directed_z_score[i][j] >= threshold:
                directed_cluster_connectivity[i][j] = (e_ij - e_ij_random) / (e_i - e_ij_random)
    return directed_cluster_connectivity, directed_z_score
