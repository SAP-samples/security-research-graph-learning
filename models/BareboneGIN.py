import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, MLP, Linear, JumpingKnowledge
from torch_geometric.utils import scatter


class BareboneGIN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.gin_convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_channels if i > 0 else -1
            self.gin_convs.append(GINConv(nn=MLP(in_channels=in_channels, hidden_channels=hidden_channels,
          out_channels=hidden_channels, num_layers=2), train_eps=False))
        
        self.jk = JumpingKnowledge(mode='cat', channels=hidden_channels, num_layers=num_layers)
        self.classifcation_layer1 = Linear(hidden_channels*num_layers, hidden_channels)
        self.classifcation_layer2 = Linear(hidden_channels, hidden_channels)
        self.classification_layer3 = Linear(hidden_channels, 1)
        

    def forward(self, X, edge_index, batch):
        xs = []
        for conv in self.gin_convs:
            X = conv(X, edge_index)
            X = F.dropout(X, p=self.dropout, training=self.training)
            X = F.relu(X)
            graph_repr = scatter(X, batch, dim=0, reduce='sum')
            xs.append(graph_repr)
        
        h = self.jk(xs) 
        X = F.relu(self.classifcation_layer1(h))
        X = F.relu(self.classifcation_layer2(X))
        logits = self.classification_layer3(X)
        return logits
    
    

if __name__ == "__main__":
    model = BareboneGIN(hidden_channels=32, num_layers=3, dropout=0.5)
    
    # pass some sample datam batch
    X = torch.randn(100, 1)
    edge_index = torch.randint(0, 30, (2, 100))
    batch = torch.randint(0, 10, (100,))
    
    # forward pass
    logits = model(X, edge_index, batch=batch)
    print(logits)
    
    print("Model has been tested successfully. Now you can use it in your project.")
