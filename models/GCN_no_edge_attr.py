import torch
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool

class GNN_no_edge_attr(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation_fn, pooling_fn, dropout_rate):
        super(GNN_no_edge_attr, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.batchnorm1 = torch.nn.BatchNorm1d(hidden_dim)
        self.batchnorm2 = torch.nn.BatchNorm1d(hidden_dim)
        self.batchnorm3 = torch.nn.BatchNorm1d(hidden_dim)
        self.pooling = pooling_fn
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = activation_fn
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
    
        x = self.conv1(x, edge_index)
        x = self.batchnorm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.batchnorm3(x)
        x = self.activation(x)

        x = self.pooling(x, data.batch)
        x = self.fc(x)
        return x