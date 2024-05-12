import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, Linear
from torch_geometric.utils import scatter

class BareboneRGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout, readout_pooling, **kwargs):
        super().__init__()
        
        self.dropout = dropout
        self.pooling = readout_pooling
        self.layers = torch.nn.ModuleList()
        
        self.num_relations=5 #
        # (node, TOTAL, node)={ edge_index=[2, 18] },
        # (node, AST, node)={ edge_index=[2, 9] },
        # (node, CFG, node)={ edge_index=[2, 4] },
        # (node, CG, node)={ edge_index=[2, 0] },
        # (node, DFG, node)={ edge_index=[2, 5] }
        
        self.initial_layer = RGCNConv(in_channels=162, out_channels=hidden_channels,num_relations=self.num_relations,is_sorted =True)
        
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers-1):
            self.layers.append(RGCNConv(in_channels=hidden_channels, out_channels=hidden_channels,num_relations=self.num_relations,is_sorted =True))
        
        self.classifcation_layer1 = Linear(hidden_channels, hidden_channels)
        self.classifcation_layer2 = Linear(hidden_channels, hidden_channels)
        self.classification_layer3 = Linear(hidden_channels, 1)
            

    def forward(self, X, edge_index1, edge_index2, edge_index3, edge_index4, edge_index5, batch):
        tensorsandedgetypes = [(tensor, i*torch.ones(tensor.shape[1], device=X.device)) for i, tensor in enumerate([edge_index1, edge_index2, edge_index3, edge_index4, edge_index5]) if tensor.nelement() > 0]
        edge_index = torch.concatenate([v[0] for v in tensorsandedgetypes], dim=1).to(torch.long)
        edge_type = torch.concatenate([v[1] for v in tensorsandedgetypes], dim=0).to(torch.long)
        
        X = self.initial_layer(X, edge_index, edge_type)
        X = F.dropout(X, training=self.training, p=self.dropout)
        X = F.relu(X)
        for layer in self.layers:
            X = layer(x=X, edge_index=edge_index, edge_type=edge_type)
            X = F.dropout(X, training=self.training, p=self.dropout)
            X = F.relu(X)
        
        
        read_out = scatter(src=X, index=batch, dim=0, reduce=self.pooling)
        X = F.relu(self.classifcation_layer1(read_out))
        X = F.relu(self.classifcation_layer2(X))
        logits = self.classification_layer3(X)

        return logits
    
    



