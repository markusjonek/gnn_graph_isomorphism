import torch
import torch.nn as nn

from models import GNN, MLP, GraphIsomorphismModel

import utils
import numpy as np
import torch_geometric
import networkx as nx

from config import Config

import graph_data


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


def main():
    config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = graph_data.GraphIsomorphismDataset(
        num_graphs=config.num_train_graphs,
        num_nodes=config.num_nodes,
        max_degree=config.max_degree,
        device=device,
        mode="train"
    )

    test_dataset = graph_data.GraphIsomorphismDataset(
        num_graphs=config.num_test_graphs,
        num_nodes=config.num_nodes,
        max_degree=config.max_degree,
        device=device,
        mode="test"
    )

    accuracy = train(config, train_dataset, test_dataset, device)
    print(f"\nAccuracy: {accuracy:.4f}")

    wl_accuracy = wl_baseline(test_dataset)
    print(f"WL Accuracy: {wl_accuracy:.4f}")



if __name__ == "__main__":
    main()
