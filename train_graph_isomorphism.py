import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import (
    GCNConv, 
    SAGEConv, 
    GATConv, 
    GATv2Conv, 
    GINConv,
    global_mean_pool, 
    global_max_pool, 
    global_add_pool,
) 
import numpy as np
import networkx as nx
import random

from torch_geometric.nn import MessagePassing

from torch_scatter import scatter

from tqdm import tqdm


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


class EAGNNLayer(MessagePassing):
    def __init__(self, input_dim, output_dim, channel_dim):
        super(EAGNNLayer, self).__init__(aggr='sum')  # Use sum aggregation.

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channel_dim = channel_dim
        
        self.weight0 = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.weight1 = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        
        self.bias = nn.Parameter(torch.FloatTensor(output_dim * channel_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight0)
        nn.init.xavier_uniform_(self.weight1)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # x: node features [N, node_features]
        # edge_index: edge list [2, E]
        # edge_attr: edge features [E, p], where p is the number of features

        support0 = torch.matmul(x, self.weight0)
        support1 = torch.matmul(x, self.weight1)

        out_tensor = torch.zeros((support0.size(0), support0.size(1), edge_attr.shape[1]), dtype=torch.float32, device=self.weight1.device)
        
        for channel in range(edge_attr.shape[1]):
            size = (support1.size(0), support1.size(0))
            edge_weight = edge_attr[:, channel]
            out_tensor[:, :, channel] = self.propagate(edge_index, size=size, x=support1, edge_attr=edge_weight) + support0
        
        # concatenate the output tensor along the channel dimension
        out_shape = (-1, support0.size(1) * edge_attr.shape[1])
        out_tensor = out_tensor.transpose(2, 1).reshape(out_shape)
        
        return out_tensor + self.bias

    def message(self, x_j, edge_attr):
        # x_j: Input features of source nodes [E, in_channels]
        # edge_attr: Edge weights [E, 1] for each edge in edge_index

        # Multiply each message by its corresponding edge weight
        weighted_message = x_j * edge_attr.unsqueeze(1)
        return weighted_message

    def update(self, aggr_out):
        # aggr_out: Output of the aggregation step [N, out_channels]
        return aggr_out




    def __repr__(self):
        return f"{self.__class__.__name__}(input_dim={self.input_dim}, output_dim={self.output_dim}, channel_dim={self.channel_dim})"


class GNN(nn.Module):
    def __init__(self, input_dim, channel_dim, layers, conv_type):
        super(GNN, self).__init__()

        self.layers = nn.ModuleList()
        
        if conv_type == "eagnn":
            self.layers.append(EAGNNLayer(input_dim, layers[0], channel_dim))
        elif conv_type == "gat":
            self.layers.append(GATv2Conv(input_dim, layers[0], edge_dim=channel_dim))
        elif conv_type == "graphsage":
            self.layers.append(SAGEConv(input_dim, layers[0]))
        elif conv_type == "gcn":
            self.layers.append(GCNConv(input_dim, layers[0]))
        elif conv_type == "gin":
            mlp = MLP(
                layers=[input_dim, input_dim, layers[0]]
            )
            self.layers.append(GINConv(mlp))
        elif conv_type == "0_hop":
            self.layers.append(nn.Linear(input_dim, layers[0]))

        for i in range(1, len(layers)):
            if conv_type == "eagnn":
                self.layers.append(EAGNNLayer(layers[i-1]*channel_dim, layers[i], channel_dim))
            elif conv_type == "gat":
                self.layers.append(GATv2Conv(layers[i-1], layers[i], edge_dim=channel_dim))
            elif conv_type == "graphsage":
                self.layers.append(SAGEConv(layers[i-1], layers[i]))
            elif conv_type == "gcn":
                self.layers.append(GCNConv(layers[i-1], layers[i]))
            elif conv_type == "gin":
                mlp = MLP(
                    layers=[layers[i-1], layers[i-1], layers[i]]
                )
                self.layers.append(GINConv(mlp))

        self.conv_type = conv_type
        if conv_type == "eagnn":
            self.out_dim = layers[-1]*channel_dim
        else:
            self.out_dim = layers[-1]

    def forward(self, node_features, edge_index, edge_attr):
        """
        node_features: torch.Tensor of shape [N, F]
            N is the number of nodes
            F is the number of node features
            
        edge_index: torch.Tensor of shape [2, E]
            E is the number of edges
            
        edge_attr: torch.Tensor of shape [E, P]
            P is the number of edge features
        
        Returns:
            torch.Tensor of shape [N, out_dim]
        """

        x = node_features
        for layer in self.layers:
            if self.conv_type == "eagnn" or self.conv_type == "gat":
                x = layer(x, edge_index, edge_attr)
            elif self.conv_type == "graphsage" or self.conv_type == "gcn" or self.conv_type == "gin":
                x = layer(x, edge_index)
            elif self.conv_type == "0_hop":
                x = layer(x)
            x = F.relu(x)
        
        return x  
        

class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        

    def forward(self, x):
        """
        edge_embs: torch.Tensor of shape [E, G, N]
            E is the number of edges
            G is the graphlet size
            N is the node embedding size

        Returns:
        torch.Tensor of shape [E, output_dim]
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        return x
    

class GraphIsomorphismModel(nn.Module):
    def __init__(self, gnn, classifier):
        super(GraphIsomorphismModel, self).__init__()
        self.gnn = gnn
        self.classifier = classifier

    def forward(self, g1, g2, sigmoid=False):
        gnn_out_1 = self.gnn(g1.x, g1.edge_index, g1.edge_attr)
        gnn_out_2 = self.gnn(g2.x, g2.edge_index, g2.edge_attr)

        graph_emb_1 = global_mean_pool(gnn_out_1, g1.batch)
        graph_emb_2 = global_mean_pool(gnn_out_2, g2.batch)

        graph_emb = torch.cat([graph_emb_1, graph_emb_2], dim=1)

        out = self.classifier(graph_emb)
        if sigmoid:
            out = torch.sigmoid(out)
        return out



def graph_isomorphism_generator(num_nodes=30, num_edges=60, train_mode=True):
    x_1 = torch.rand(num_nodes, 2)

    edge_list_1 = []
    while len(edge_list_1) < num_edges*2:
        n1 = random.randint(0, num_nodes-1)
        n2 = random.randint(0, num_nodes-1)

        if n1 == n2:
            continue

        if (n1, n2) in edge_list_1 or (n2, n1) in edge_list_1:
            continue

        edge_list_1.append((n1, n2))
        edge_list_1.append((n2, n1))

    edge_index_1 = torch.tensor(edge_list_1, dtype=torch.long).t()

    edge_dists_1 = torch.norm(x_1[edge_index_1[0]] - x_1[edge_index_1[1]], dim=1)
    edge_dists_1 = normalize_edge_attributes(edge_index_1, edge_dists_1)
    edge_attr_1 = edge_dists_1.view(-1, 1)

    # create a graph
    graph1 = Data(x=x_1, edge_index=edge_index_1, edge_attr=edge_attr_1)

    if train_mode:
        num_edges_to_change = random.randint(1, num_edges)
    else:
        num_edges_to_change = 10

    edge_index_1 = edge_index_1[:, num_edges_to_change*2:]

    new_edges = []
    while len(new_edges) < num_edges_to_change*2:
        n1 = random.randint(0, num_nodes-1)
        n2 = random.randint(0, num_nodes-1)

        if n1 == n2:
            continue

        # check if the edge already exists
        source_matches = (edge_index_1[0] == n1)
        target_matches = (edge_index_1[1] == n2)
        already_exists = (source_matches & target_matches).any()

        source_matches_2 = (edge_index_1[0] == n2)
        target_matches_2 = (edge_index_1[1] == n1)
        already_exists = already_exists or (source_matches_2 & target_matches_2).any()

        if len(new_edges) > 0:
            # check if the edge is already in the new_edges
            source_matches = (torch.tensor(new_edges, dtype=torch.long).t()[0] == n1)
            target_matches = (torch.tensor(new_edges, dtype=torch.long).t()[1] == n2)
            already_exists = already_exists or (source_matches & target_matches).any()

            source_matches_2 = (torch.tensor(new_edges, dtype=torch.long).t()[0] == n2)
            target_matches_2 = (torch.tensor(new_edges, dtype=torch.long).t()[1] == n1)
            already_exists = already_exists or (source_matches_2 & target_matches_2).any()

        if already_exists:
            continue

        new_edges.append([n1, n2])
        new_edges.append([n2, n1])

    new_edges = torch.tensor(new_edges).t()

    edges_index_2 = torch.cat([edge_index_1, new_edges], dim=1)

    edges_index_2 = torch_geometric.utils.to_undirected(edges_index_2)
    edges_index_2 = torch_geometric.utils.remove_self_loops(edges_index_2)[0]
    edge_dists_2 = torch.norm(x_1[edges_index_2[0]] - x_1[edges_index_2[1]], dim=1)




    edge_dists_2 = normalize_edge_attributes(edges_index_2, edge_dists_2)
    edge_attr_2 = edge_dists_2.view(-1, 1)

    graph2 = Data(x=x_1, edge_index=edges_index_2, edge_attr=edge_attr_2)

    return graph1, graph2



def graph_generator_2(num_nodes, num_edges):
    edge_list = []

    while len(edge_list) < num_edges*2:
        n1 = random.randint(0, num_nodes-1)
        n2 = random.randint(0, num_nodes-1)

        if n1 == n2:
            continue

        if (n1, n2) in edge_list or (n2, n1) in edge_list:
            continue

        edge_list.append((n1, n2))
        edge_list.append((n2, n1))

    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    node_degrees = torch_geometric.utils.degree(edge_index[0], num_nodes)


    # add edges until no node has a node degree 0
    while (node_degrees == 0).any():
        n1 = (node_degrees == 0).nonzero(as_tuple=False)[0].item()
        n2 = random.randint(0, num_nodes-1)

        if n1 == n2:
            continue

        if (n1, n2) in edge_list or (n2, n1) in edge_list:
            continue

        edge_list.append((n1, n2))
        edge_list.append((n2, n1))

        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        node_degrees = torch_geometric.utils.degree(edge_index[0], num_nodes)


    node_degree_diff = torch.abs(node_degrees[edge_index[0]] - node_degrees[edge_index[1]])

    return Data(
        edge_index=edge_index, 
        x=node_degrees, 
        edge_attr=node_degree_diff.view(-1, 1),
        node_degrees=node_degrees
    )

def graph_generator_3(num_nodes, max_degree):
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



def graph_isomorphism_generator_2(num_nodes, max_degree):
    g1, g2 = graph_generator_3(num_nodes, max_degree)

    G1 = torch_geometric.utils.to_networkx(g1)
    G2 = torch_geometric.utils.to_networkx(g2)

    assert G1.number_of_nodes() == G2.number_of_nodes()
    assert G1.number_of_edges() == G2.number_of_edges()

    while nx.algorithms.isomorphism.is_isomorphic(G1, G2):
        g1, g2 = graph_generator_3(num_nodes, max_degree)
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
    

def add_new_edge(g, max_tries=1000):
    edges = g.edge_index.t().tolist()

    tries = 0
    while True:
        n1 = random.randint(0, g.x.shape[0]-1)
        n2 = random.randint(0, g.x.shape[0]-1)

        tries += 1
        if tries > max_tries:
            return g, False

        if n1 == n2:
            continue

        if [n1, n2] in edges or [n2, n1] in edges:
            continue

        edges.append([n1, n2])
        edges.append([n2, n1])
        
        edge_index = torch.tensor(edges).t()
        node_degrees = torch_geometric.utils.degree(edge_index[0], g.x.shape[0])
        node_degree_diff = torch.abs(node_degrees[edge_index[0]] - node_degrees[edge_index[1]])

        g = Data(
            edge_index=edge_index,
            x=node_degrees,
            edge_attr=node_degree_diff.view(-1, 1),
            node_degrees=node_degrees.to(torch.long)
        )

        return g, True





class GraphIsomorphismDataset(torch.utils.data.Dataset):
    def __init__(self, num_graphs, num_nodes, max_degree, device, mode):
        self.num_graphs = num_graphs
        self.graphs = []

        for i in tqdm(range(num_graphs)):
            g1, g2 = graph_isomorphism_generator_2(
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


def find_max_degree(train_dataset):
    all_degrees = []
    for i in range(len(train_dataset)):
        g1, g2, target = train_dataset[i]
        all_degrees.extend(g1.node_degrees.tolist())
        all_degrees.extend(g2.node_degrees.tolist())

    avg_degree = np.mean(all_degrees)
    print(f"Average degree: {avg_degree}")
    max_degree = max(all_degrees)
    return int(max_degree)



@torch.no_grad()
def evaluate_model(gnn_iso_model, train_loader, test_loader, device):
    gnn_iso_model.eval()

    preds = []
    targets = []

    # find the best threshold on the training set
    for i, (g1s, g2s, target) in enumerate(train_loader):
        g1s = g1s.to(device)
        g2s = g2s.to(device)
        target = target.to(device)

        out = gnn_iso_model(g1s, g2s, sigmoid=True)

        preds.extend(out.cpu().detach().numpy().tolist())
        targets.extend(target.cpu().detach().numpy().tolist())

    thresholds = np.linspace(0, 1, 100)
    best_accuracy = 0
    best_threshold = 0

    for threshold in thresholds:
        preds_binary = np.array(preds) > threshold
        accuracy = np.mean(preds_binary == np.array(targets))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    # validate on the test set
    correct = 0
    total = 0
    for i, (g1s, g2s, target) in enumerate(test_loader):
        g1s = g1s.to(device)
        g2s = g2s.to(device)
        target = target.to(device)

        out = gnn_iso_model(g1s, g2s, sigmoid=True)

        preds = out.cpu().detach().numpy() > best_threshold
        correct += np.sum(preds == target.cpu().detach().numpy())
        total += len(preds)

    accuracy = correct / total

    return best_accuracy




def train(config, train_dataset, test_dataset, device):
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )

    gnn = GNN(
        input_dim=train_dataset[0][0].x.shape[1],
        channel_dim=train_dataset[0][0].edge_attr.shape[1],
        layers=config.gnn_layers,
        conv_type=config.conv_type
    )

    classifier = MLP(
        layers=config.classifier_layers
    )

    gnn_iso_model = GraphIsomorphismModel(
        gnn, 
        classifier, 
    )
    gnn_iso_model = gnn_iso_model.to(device)

    optimizer = torch.optim.Adam(
        gnn_iso_model.parameters(),
        lr=config.lr,
    )

    criterion = nn.BCEWithLogitsLoss()

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=1000, 
        gamma=0.85
    )

    losses = []

    for epoch in range(config.num_epochs):
        for i, (g1s, g2s, targets) in enumerate(train_loader):
            gnn_iso_model.train()
            optimizer.zero_grad()

            g1s = g1s.to(device)
            g2s = g2s.to(device)
            targets = targets.to(device)

            out = gnn_iso_model(g1s, g2s)

            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            cur_lr = optimizer.param_groups[0]['lr']

            losses.append(loss.item())
            print(f"\rEpoch: {epoch}, Loss: {np.mean(losses[-200:]):.4f}, Cur lr: {cur_lr:.6f} - {i/len(train_loader)*100:.2f}%    ", end="")

    # test the model
    accuracy = evaluate_model(gnn_iso_model, train_loader, test_loader, device)
    return accuracy



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



class Config:
    def __init__(self):
        self.gnn_layers = [64]*4
        self.classifier_layers = [128, 64, 32, 1]
        self.lr = 0.001
        self.num_train_graphs = 5000
        self.num_test_graphs = 2000
        self.num_epochs = 50
        self.batch_size = 128

        self.conv_type = "eagnn"

        self.max_degree = 20
        self.num_nodes = 1000

        self.base_random_seed = random.randint(0, 10000)

def run_experiemnts():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = Config()

    train_dataset = GraphIsomorphismDataset(
       num_graphs=config.num_train_graphs, 
       num_nodes=config.num_nodes,
       max_degree=config.max_degree,
       device=device,
       mode="train"
    )

    test_dataset = GraphIsomorphismDataset(
        num_graphs=config.num_test_graphs, 
        num_nodes=config.num_nodes,
        max_degree=config.max_degree,
        device=device,
        mode="test"
    )

    conv_types = ["eagnn", "graphsage", "gcn", "gin", "gat"]

    for num_layers in range(1, 6):
        print()
        print(f"Number of layers: {num_layers}")
        for conv_type in conv_types:
            config = Config()
            config.conv_type = conv_type
            config.gnn_layers = [64]*num_layers
            accuracy = train(config, train_dataset, test_dataset, device)
            print(f"\rAccuracy {conv_type}: {accuracy}                                        ")

    print()

    wl_accuracy_no_edges = wl_baseline(config, test_dataset, use_edge_attr=False)
    wl_accuracy_edges = wl_baseline(config, test_dataset, use_edge_attr=True)

    print(f"WL accuracy without edge_attr: {wl_accuracy_no_edges}")
    print(f"WL accuracy with edge_attr: {wl_accuracy_edges}")


def playground():
    config = Config()
    max_degree = find_max_degree()

    config.num_train_graphs = 10000
    accuracy = train(config, max_degree)
    print(f"Accuracy: {accuracy}")

    config.num_train_graphs = 40000
    accuracy = train(config, max_degree)
    print(f"Accuracy: {accuracy}")




if __name__ == "__main__":
    # playground()
    run_experiemnts()
