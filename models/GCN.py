import torch
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool, MessagePassing
from torch_geometric.utils import add_self_loops, softmax



class EdgeFeatGCNConv(MessagePassing):
    def __init__(self, input_dim, hidden_dim, edge_dim):
        super().__init__(aggr='add')
        self.node_fc = torch.nn.Linear(input_dim, hidden_dim)
        self.edge_fc = torch.nn.Linear(edge_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_fc(x)
        edge_attr = self.edge_fc(edge_attr)
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, fill_value=0)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr
    

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation_fn, pooling_fn, dropout_rate, edge_dim):
        super(GNN, self).__init__()
        self.conv1 = EdgeFeatGCNConv(input_dim, hidden_dim, edge_dim)
        self.conv2 = EdgeFeatGCNConv(hidden_dim, hidden_dim, edge_dim)
        self.conv3 = EdgeFeatGCNConv(hidden_dim, hidden_dim, edge_dim)

        self.batchnorm1 = torch.nn.BatchNorm1d(hidden_dim)
        self.batchnorm2 = torch.nn.BatchNorm1d(hidden_dim)
        self.batchnorm3 = torch.nn.BatchNorm1d(hidden_dim)
        self.pooling = pooling_fn
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = activation_fn
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    
        x = self.conv1(x, edge_index, edge_attr)
        x = self.batchnorm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.batchnorm3(x)
        x = self.activation(x)

        x = self.pooling(x, data.batch)
        x = self.fc(x)
        return x