import torch
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, Linear
from torch_geometric.utils import scatter

class Reveal(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout, readout_pooling, **kwargs):
        super().__init__()
        
        self.dropout = dropout
        self.pooling = readout_pooling
        self.layers = torch.nn.ModuleList()
        
        
        self.initial_layer = Linear(in_channels=-1, out_channels=hidden_channels)  # initial input X has different dimensions, so it must be transformed
        self.ggnn = GatedGraphConv(hidden_channels, num_layers)
        self.classification_layer1 = Linear(hidden_channels, 256)
        self.classification_layer2 = Linear(256, 128)
        self.classification_layer3 = Linear(128, 256)
        self.classification_layer4 = Linear(256, 1)
            

    def forward(self, X, edge_index, batch):
        
        X = self.initial_layer(X)
        X = F.dropout(X, training=self.training, p=self.dropout)
        X = F.relu(X)
        
        
        X = self.ggnn(X, edge_index)
        X = F.dropout(X, training=self.training, p=self.dropout)
        X = F.relu(X)
        
        
        X = self.classification_layer1(X)
        X = F.dropout(X, training=self.training, p=self.dropout)
        X = F.relu(X)
        X = self.classification_layer2(X)
        X = F.dropout(X, training=self.training, p=self.dropout)
        X = F.relu(X)
        
        X = self.classification_layer3(X)
        X = F.dropout(X, training=self.training, p=self.dropout)
        X = F.relu(X)
        
        read_out = scatter(src=X, index=batch, dim=0, reduce=self.pooling)
        logits = self.classification_layer4(read_out)
        return logits
    
    



