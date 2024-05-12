import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.utils import scatter


class BareboneGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout, readout_pooling, **kwargs):
        super().__init__()
        
        self.dropout = dropout
        self.pooling = readout_pooling
        self.layers = torch.nn.ModuleList()
        
        
        self.initial_layer = GCNConv(in_channels=-1, out_channels=hidden_channels)
        
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers-1):
            self.layers.append(GCNConv(in_channels=hidden_channels, out_channels=hidden_channels))
        
        self.classifcation_layer1 = Linear(hidden_channels, hidden_channels)
        self.classifcation_layer2 = Linear(hidden_channels, hidden_channels)
        self.classification_layer3 = Linear(hidden_channels, 1)
        
        
            

    def forward(self, X, edge_index, batch):
        
        X = self.initial_layer(X, edge_index)
        X = F.dropout(X, training=self.training, p=self.dropout) # droput before relu faster
        X = F.relu(X)
        for layer in self.layers:
            X = layer(x=X, edge_index=edge_index)
            X = F.dropout(X, training=self.training, p=self.dropout)
            X = F.relu(X)
            
        
        
        read_out = scatter(src=X, index=batch, dim=0, reduce=self.pooling)
        X = F.relu(self.classifcation_layer1(read_out))
        X = F.relu(self.classifcation_layer2(X))
        logits = self.classification_layer3(X)

        return logits
    
    



