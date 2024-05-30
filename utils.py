from torch_scatter import scatter
import torch
import torch_geometric
import networkx as nx


def normalize_edge_attributes(edge_index, edge_attr):
    """
    edge_index: [2, E] tensor with source and target nodes of each edge
    edge_attr: [E] tensor with the attribute (weight) of each edge
    """
    src_nodes = edge_index[0, :]
    
    # Sum weights of outgoing edges for each node
    # Effectively sums up the weights of all outgoing edges for each node
    sum_outgoing_weights = scatter(edge_attr, src_nodes, dim=0, reduce='add')
    
    # For normalization, we divide each edge's weight by the sum of its source node's outgoing edge weights
    # We use src_nodes to index into sum_outgoing_weights to get the normalization factor for each edge
    safe_sum_feature_outgoing = sum_outgoing_weights.clone()
    safe_sum_feature_outgoing[safe_sum_feature_outgoing == 0] = 1

    # Normalize feature, temporarily using safe sum
    normalized_feature = edge_attr / safe_sum_feature_outgoing[src_nodes]

    # Correct cases where the sum was originally 0: set normalized feature to 0 in these cases
    correction_mask = sum_outgoing_weights[src_nodes] == 0  # Mask of positions where sum was 0
    normalized_feature[correction_mask] = 0  # Apply correction

    return normalized_feature


def wl_baseline(test_dataset, use_edge_attr=False):
    # compare with WL
    correct = 0
    total = 0

    for g1, g2, target in test_dataset:
        G1 = torch_geometric.utils.to_networkx(g1)
        G2 = torch_geometric.utils.to_networkx(g2)

        # set edge attributes
        for edge in G1.edges:
            g1_edge_idx = torch.where((g1.edge_index[0] == edge[0]) & (g1.edge_index[1] == edge[1]))[0]
            G1.edges[edge]['weight'] = g1.edge_attr[g1_edge_idx].item()
        for edge in G2.edges:
            g2_edge_idx = torch.where((g2.edge_index[0] == edge[0]) & (g2.edge_index[1] == edge[1]))[0]
            g2_edge_idx = g2_edge_idx[0]
            G2.edges[edge]['weight'] = g2.edge_attr[g2_edge_idx].item()

        if use_edge_attr:
            wl_hash_1 = nx.weisfeiler_lehman_graph_hash(G1, edge_attr='weight')
            wl_hash_2 = nx.weisfeiler_lehman_graph_hash(G2, edge_attr='weight')
        else:
            wl_hash_1 = nx.weisfeiler_lehman_graph_hash(G1)
            wl_hash_2 = nx.weisfeiler_lehman_graph_hash(G2)

        if wl_hash_1 == wl_hash_2 and target == 1:
            correct += 1
        elif wl_hash_1 != wl_hash_2 and target == 0:
            correct += 1
        total += 1

    return correct/total
