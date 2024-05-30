import torch
import torch.nn as nn
import torch.nn.functional as F
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

from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from layers import EAGNNLayer

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
