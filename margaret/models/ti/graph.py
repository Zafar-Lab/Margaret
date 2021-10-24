import networkx as nx
import numpy as np
import pandas as pd

from utils.util import compute_runtime


@compute_runtime
def compute_gt_milestone_network(ad, uns_mn_key="milestone_network", mode="directed"):

    # NOTE: Since the dyntoy tool does not provide the spatial position
    # of the milestones, this function uses spring_layout to plot the
    # node positions. Hence the displayed graph is not guaranteed to produce
    # accurate spatial embeddings of milestones in the 2d plane.

    assert mode in ["directed", "undirected"]

    if uns_mn_key not in ad.uns_keys():
        raise Exception(f"Milestone network not found in uns.{uns_mn_key}")

    # Get a set of unique milestones
    mn = ad.uns[uns_mn_key]
    from_milestones = set(mn["from"].unique())
    to_milestones = set(mn["to"].unique())
    milestones = from_milestones.union(to_milestones)

    # Construct the milestone network
    milestone_network = nx.DiGraph() if mode == "directed" else nx.Graph()
    start_milestones = (
        [ad.uns["start_milestones"]]
        if isinstance(ad.uns["start_milestones"], str)
        else list(ad.uns["start_milestones"])
    )
    for milestone in milestones:
        milestone_network.add_node(milestone)

    for idx, (f, t) in enumerate(zip(mn["from"], mn["to"])):
        milestone_network.add_edge(f, t, weight=mn["length"][idx])
    return milestone_network


@compute_runtime
def compute_connectivity_graph(
    embeddings, communities, cluster_connectivities, mode="undirected"
):
    assert mode in ["directed", "undirected"]
    g = nx.Graph() if mode == "undirected" else nx.DiGraph()
    node_positions = {}
    cluster_ids = np.unique(communities)
    for i in cluster_ids:
        g.add_node(i)
        # determine the node pos for the cluster
        cluster_i = communities == i
        node_pos = np.mean(embeddings[cluster_i, :], axis=0)
        node_positions[i] = node_pos

    n_nodes = len(cluster_ids)
    for row_id, i in enumerate(cluster_ids):
        for col_id, j in enumerate(cluster_ids):
            if cluster_connectivities.loc[i, j] > 0:
                g.add_edge(
                    cluster_ids[row_id],
                    cluster_ids[col_id],
                    weight=cluster_connectivities.loc[i, j],
                )
    return g, node_positions


@compute_runtime
def compute_trajectory_graph(
    embeddings, communities, cluster_connectivities, start_cell_ids
):
    g = nx.DiGraph()
    node_positions = {}
    cluster_ids = np.unique(communities)
    for i in cluster_ids:
        g.add_node(i)
        # determine the node pos for the cluster
        cluster_i = communities == i
        node_pos = np.mean(embeddings[cluster_i, :], axis=0)
        node_positions[i] = node_pos

    n_nodes = len(cluster_ids)
    visited = [False] * n_nodes
    for start_cell_cluster_idx in start_cell_ids:
        current_node_id = start_cell_cluster_idx
        s = [current_node_id]
        while True:
            if s == []:
                break
            current_node_id = s.pop()
            if visited[current_node_id] is True:
                continue
            inds = np.argsort(cluster_connectivities[current_node_id, :])
            inds = inds[cluster_connectivities[current_node_id, inds] > 0]
            s.extend(cluster_ids[inds])
            visited[current_node_id] = True
            for id in cluster_ids[inds]:
                if visited[id] is True:
                    continue
                g.add_edge(
                    current_node_id,
                    id,
                    weight=cluster_connectivities[current_node_id][id],
                )
    return g, node_positions


@compute_runtime
def compute_trajectory_graph_v2(
    pseudotime, adj_cluster, communities, d_connectivity, norm=False
):
    n_communities = np.unique(communities).shape[0]
    cluster_ids = np.unique(communities)

    adj = pd.DataFrame(
        np.zeros((n_communities, n_communities)), index=cluster_ids, columns=cluster_ids
    )

    # Create cluster index
    cluster_pt = pd.DataFrame(index=cluster_ids)
    for idx in cluster_ids:
        cluster_idx = communities == idx
        cluster_pt.loc[idx, "t"] = np.mean(pseudotime.loc[cluster_idx])

    cols = adj_cluster.columns
    for idx in cluster_ids:
        connected_c_idx = cols[adj_cluster.loc[idx, :] != 0]
        for c_idx in connected_c_idx:
            if (cluster_pt.loc[c_idx, "t"] > cluster_pt.loc[idx, "t"]) and (
                adj_cluster.loc[c_idx, idx] != 0
            ):
                # The edge weight will be the contribution from the directed
                # connectivities and difference of the pseudotimes
                adj.loc[idx, c_idx] = d_connectivity.loc[idx, c_idx] + 1 / (
                    1 + np.exp(cluster_pt.loc[c_idx, "t"] - cluster_pt.loc[idx, "t"])
                )

    # Normalize the directed adjacency matrix
    if norm:
        adj = adj.div(adj.sum(axis=1), axis=0).fillna(0)
    return adj
