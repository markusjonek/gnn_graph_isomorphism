# gnn_graph_isomorphism
Benchmark of GNN models on graph isomorphism classification

## Usage
Set the desired config in `config.py`

The start training with `python3 train.py`

### Run all experiments
Run all experiemnts with `python3 run_experiments.py`


## Results
- **Nodes:** 1000
- **Max Node Degree:** 20
- **GNN Dimensions:** 64
- **Classifier Layers:** [128, 64, 32, 1]
- **Batch Size:** 128
- **Learning Rate (lr):** 0.001
- **Step Decay:** 0.85 every 1000 steps
- **Number of Training Graphs:** 5000
- **Number of Test Graphs:** 2000
- **Number of Epochs:** 50
- **Node Input:** One-hot encoded node degrees
- **Edge Input:** Absolute value of node degree difference


| Number of Layers | EAGNN   | GraphSAGE | GCN    | GIN    | GAT    |
|------------------|---------|-----------|--------|--------|--------|
| 1                | 50.04   | 50.04     | 50.04  | 50.04  | 50.04  |
| 2                | 58.00   | 50.04     | 50.04  | 50.04  | 50.04  |
| 3                | 96.50   | 50.04     | 50.04  | 50.04  | 50.04  |
| 4                | 79.64   | 50.04     | 50.04  | 50.04  | 50.04  |
| 5                | 63.26   | 50.04     | 50.04  | 50.04  | 50.04  |
