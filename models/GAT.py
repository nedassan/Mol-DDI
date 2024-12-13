import torch
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation_fn, pooling_fn, dropout_rate, num_heads, edge_dim):
        super(GNN, self).__init__()
        
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads)
        self.conv3 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads)
        
        self.batchnorm1 = torch.nn.BatchNorm1d(hidden_dim * num_heads)
        self.batchnorm2 = torch.nn.BatchNorm1d(hidden_dim * num_heads)
        self.batchnorm3 = torch.nn.BatchNorm1d(hidden_dim * num_heads)
        
        self.pooling = pooling_fn
        self.fc = torch.nn.Linear(hidden_dim * num_heads, output_dim)
        
        self.activation = activation_fn
        self.dropout = torch.nn.Dropout(dropout_rate)

        self.edge_fc = torch.nn.Linear(edge_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        edge_attr = self.edge_fc(edge_attr)

        x = self.conv1(x, edge_index, edge_attr = edge_attr)
        x = self.batchnorm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr = edge_attr)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_attr = edge_attr)
        x = self.batchnorm3(x)
        x = self.activation(x)

        x = self.pooling(x, data.batch)
        x = self.fc(x)
        return x