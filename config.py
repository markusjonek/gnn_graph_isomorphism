
class Config:
    gnn_layers = [64]*4
    classifier_layers = [128, 64, 32, 1] # first layer should be 2*gnn_layers[-1]
    lr = 0.001
    num_train_graphs = 5000
    num_test_graphs = 2000
    num_epochs = 100
    batch_size = 128
    conv_type = "eagnn" # choose between: "eagnn", "graphsage", "gcn", "gin", "gat"
    max_degree = 20
    num_nodes = 1000

