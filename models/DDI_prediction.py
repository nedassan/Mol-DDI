import torch
import torch.nn as nn
import torch.nn.functional as F



class FFNN(torch.nn.Module):
    def __init__(self, gnn, embed_dim, hidden_dim, num_layers, activation_fn):
        super(FFNN, self).__init__()
        self.gnn = gnn
        seq_layers = [torch.nn.Linear(2*embed_dim, hidden_dim), activation_fn]
        for _ in range(num_layers - 2):
            seq_layers.extend([torch.nn.Linear(hidden_dim, hidden_dim), activation_fn])
        seq_layers.extend([torch.nn.Linear(hidden_dim, 1), torch.nn.Sigmoid()])
        self.fc = torch.nn.Sequential(*seq_layers)

    def forward(self, molecule1, molecule2, return_embed = False):
        embed_1 = self.gnn(molecule1)
        embed_2 = self.gnn(molecule2)
        concat_embed = torch.cat((embed_1, embed_2), dim = 1)
        if return_embed:
            return concat_embed
        
        pred_prob_ddi = self.fc(concat_embed)
        return pred_prob_ddi, embed_1, embed_2
    

class CrossAttention(torch.nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(CrossAttention, self).__init__()
        self.hidden_dim = torch.Tensor([hidden_dim])
        self.query_fc = torch.nn.Linear(embed_dim, hidden_dim)
        self.key_fc = torch.nn.Linear(embed_dim, hidden_dim)
        self.value_fc = torch.nn.Linear(embed_dim, hidden_dim)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, molecule1, molecule2):
        Q = self.query_fc(molecule1) 
        K = self.key_fc(molecule2)   
        V = self.value_fc(molecule2) 

        attn_scores = torch.matmul(Q, K.T)/torch.sqrt(self.hidden_dim) 
        attn_probs = self.softmax(attn_scores)
        attn_output = torch.matmul(attn_probs, V) 

        return attn_output

class FFNNAttn(torch.nn.Module):
    def __init__(self, gnn, embed_dim, hidden_dim, num_layers, activation_fn):
        super(FFNNAttn, self).__init__()
        self.gnn = gnn
        self.attn = CrossAttention(embed_dim, hidden_dim)
        seq_layers = []
        for _ in range(num_layers - 1):
            seq_layers.extend([torch.nn.Linear(hidden_dim, hidden_dim), activation_fn])
        seq_layers.extend([torch.nn.Linear(hidden_dim, 1), torch.nn.Sigmoid()])
        self.fc = torch.nn.Sequential(*seq_layers)

    def forward(self, molecule1, molecule2, return_embed = False):
        embed_1 = self.gnn(molecule1)
        embed_2 = self.gnn(molecule2)
        
        attn_embed = self.attn(embed_1, embed_2)
        if return_embed:
            return attn_embed
        
        pred_prob_ddi = self.fc(attn_embed)
        return pred_prob_ddi, embed_1, embed_2
    
class LSTM_DDI(torch.nn.Module):
    def __init__(self, gnn, embed_dim, hidden_dim, num_layers, activation_fn):
        super(LSTM_DDI, self).__init__()
        self.gnn = gnn
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first = True)
        
        self.fc = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), activation_fn, torch.nn.Linear(hidden_dim, 1), torch.nn.Sigmoid())

    def forward(self, molecule1, molecule2, return_embed = False):
        embed_1 = self.gnn(molecule1)
        embed_2 = self.gnn(molecule2)

        lstm_input = torch.stack([embed_1, embed_2], dim = 1)
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)

        lstm_output = h_n[-1]

        pred_prob_ddi = self.fc(lstm_output)
        if return_embed:
            return lstm_output
        
        return pred_prob_ddi, embed_1, embed_2
    
class GRU_DDI(torch.nn.Module):
    def __init__(self, gnn, embed_dim, hidden_dim, num_layers, activation_fn):
        super(GRU_DDI, self).__init__()
        self.gnn = gnn
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first = True)  
        self.fc = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), activation_fn, torch.nn.Linear(hidden_dim, 1), torch.nn.Sigmoid())

    def forward(self, molecule1, molecule2, return_embed = False):
        embed_1 = self.gnn(molecule1)
        embed_2 = self.gnn(molecule2)

        gru_input = torch.stack([embed_1, embed_2], dim = 1)
        gru_out, h_n = self.gru(gru_input)

        gru_output = h_n[-1]

        pred_prob_ddi = self.fc(gru_output)
        if return_embed:
            return gru_output
        
        return pred_prob_ddi, embed_1, embed_2