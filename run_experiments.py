from config import Config
from train import train
from graph_data import GraphIsomorphismDataset
import torch
import utils


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
            config.conv_type = conv_type
            config.gnn_layers = [config.gnn_layers[0]]*num_layers
            accuracy = train(config, train_dataset, test_dataset, device)
            print(f"\rAccuracy {conv_type}: {accuracy}                                        ")

    print()

    wl_accuracy_no_edges = utils.wl_baseline(test_dataset, use_edge_attr=False)
    wl_accuracy_edges = utils.wl_baseline(test_dataset, use_edge_attr=True)

    print(f"WL accuracy without edge_attr: {wl_accuracy_no_edges}")
    print(f"WL accuracy with edge_attr: {wl_accuracy_edges}")

if __name__ == "__main__":
    run_experiemnts()