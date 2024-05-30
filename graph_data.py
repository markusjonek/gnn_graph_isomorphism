import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, Batch
import torch_geometric.utils
from torch_scatter import scatter

import random
import numpy as np
import networkx as nx
from tqdm import tqdm


def graph_generator(num_nodes, max_degree):
    """
    Generates two random graphs with the same degree sequence.
    """

    # generete a degree sequence
    def create_degree_sequence(n, max_deg):
        while True:
            sequence = [random.randint(2, max_deg) for _ in range(n)]
            if sum(sequence) % 2 == 0:  # Ensure the sum of degrees is even
                return sequence
            
    degree_sequence = create_degree_sequence(num_nodes, max_degree)

    # generate a random graph
    G1 = nx.random_degree_sequence_graph(degree_sequence)
    G2 = nx.random_degree_sequence_graph(degree_sequence)

    G1 = G1.to_undirected()
    G2 = G2.to_undirected()

    # to torch_geometric
    edge_index_1 = torch.tensor(list(G1.edges)).t()
    edge_index_2 = torch.tensor(list(G2.edges)).t()

    edge_index_1 = torch_geometric.utils.to_undirected(edge_index_1)
    edge_index_2 = torch_geometric.utils.to_undirected(edge_index_2)

    node_degrees_1 = torch_geometric.utils.degree(edge_index_1[0], num_nodes)
    node_degrees_2 = torch_geometric.utils.degree(edge_index_2[0], num_nodes)

    node_degree_diff_1 = torch.abs(node_degrees_1[edge_index_1[0]] - node_degrees_1[edge_index_1[1]])
    node_degree_diff_2 = torch.abs(node_degrees_2[edge_index_2[0]] - node_degrees_2[edge_index_2[1]])

    g1 = Data(
        edge_index=edge_index_1,
        x=node_degrees_1,
        edge_attr=node_degree_diff_1.view(-1, 1),
        node_degrees=node_degrees_1.to(torch.long)
    )

    g2 = Data(
        edge_index=edge_index_2,
        x=node_degrees_2,
        edge_attr=node_degree_diff_2.view(-1, 1),
        node_degrees=node_degrees_2.to(torch.long)
    )

    return g1, g2




def graph_isomorphism_generator(num_nodes, max_degree):
    g1, g2 = graph_generator(num_nodes, max_degree)

    G1 = torch_geometric.utils.to_networkx(g1)
    G2 = torch_geometric.utils.to_networkx(g2)

    assert G1.number_of_nodes() == G2.number_of_nodes()
    assert G1.number_of_edges() == G2.number_of_edges()

    while nx.algorithms.isomorphism.is_isomorphic(G1, G2):
        g1, g2 = graph_generator(num_nodes, max_degree)
        G1 = torch_geometric.utils.to_networkx(g1)
        G2 = torch_geometric.utils.to_networkx(g2)

    return g1, g2



def rerout_edge(g, max_tries=1000):
    edges = g.edge_index.t().tolist()

    # pick a random edge
    edge_idx = random.randint(0, len(edges)-1)
    edge = edges[edge_idx]

    n1 = edge[0]
    n2 = edge[1]

    # remove the edge
    edges.remove([n1, n2])
    edges.remove([n2, n1])

    nd2 = g.node_degrees[n2].item()

    # find a new node to connect to which has the same degree
    tries = 0
    while True:
        n3 = random.randint(0, g.x.shape[0]-1)
        tries += 1

        if tries > max_tries:
            return g, False

        if n3 == n1 or n3 == n2:
            continue
        if g.node_degrees[n3].item() == nd2 - 1:
            break

    edges.append([n1, n3])
    edges.append([n3, n1])
    
    edge_index = torch.tensor(edges).t()
    node_degrees = torch_geometric.utils.degree(edge_index[0], g.x.shape[0])

    assert g.node_degrees[n1] == node_degrees[n1]
    assert g.node_degrees[n2] == node_degrees[n3], f"{g.node_degrees[n2]} != {node_degrees[n3]}"

    node_degree_diff = torch.abs(node_degrees[edge_index[0]] - node_degrees[edge_index[1]])

    new_g = Data(
        edge_index=edge_index,
        x=node_degrees,
        edge_attr=node_degree_diff.view(-1, 1),
        node_degrees=node_degrees.to(torch.long)
    )

    # check if the new graph is isomorphic to the original one
    G1 = torch_geometric.utils.to_networkx(g)
    G2 = torch_geometric.utils.to_networkx(new_g)

    if nx.algorithms.isomorphism.is_isomorphic(G1, G2):
        return g, False
    
    return new_g, True
    


class GraphIsomorphismDataset(torch.utils.data.Dataset):
    def __init__(self, num_graphs, num_nodes, max_degree, device, mode):
        self.num_graphs = num_graphs
        self.graphs = []

        for i in tqdm(range(num_graphs)):
            g1, g2 = graph_isomorphism_generator(
                num_nodes=num_nodes, 
                max_degree=max_degree
            )
            target = torch.tensor([0.0], dtype=torch.float32)

            if mode == "test":
                g2, check = rerout_edge(g1)
                if not check:
                    continue

            if random.random() > 0.5:
                g1, g2 = g1, g1
                target = torch.tensor([1.0], dtype=torch.float32)
  

            g1.x = F.one_hot(g1.node_degrees, num_classes=max_degree+1).float()
            g2.x = F.one_hot(g2.node_degrees, num_classes=max_degree+1).float()

            g1 = g1.to(device)
            g2 = g2.to(device)

            target = target.to(device)
            self.graphs.append((g1, g2, target))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g1, g2, target = self.graphs[idx]
        return g1, g2, target
    
    def collate_fn(self, batch):
        graphs1, graphs2, targets = zip(*batch)
        batched_g1 = Batch.from_data_list(graphs1)
        batched_g2 = Batch.from_data_list(graphs2)
        targets = torch.stack(targets, dim=0)
        return batched_g1, batched_g2, targets
