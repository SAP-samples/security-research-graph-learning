# %%
import torch 
from models.BareboneGCN import BareboneGCN
from CodeGraphDataset import CodeGraphDataset_InMemory
dataset4 = CodeGraphDataset_InMemory(pt_folder='codegraphs/diversevul/v2_undirected_withdegreecount', split='train', 
                                          cross_val_valfold_idx=0, 
                                          is_cross_val=True, cross_val_train_fraction=0.01, DS_type='all')

# %% [markdown]
# # Test GraphGLOW with DENSE

# %%
i = 0
while True:
    sample = dataset4[i]
    i += 1   
    if sample.x.shape[0]>910 and sample.x.shape[0]<1010:
        break 

# %%
sample=sample.cuda()

# %%
edge_index = sample.edge_index
# make a dense pytorch matrix out of it
from torch_geometric.utils import to_dense_adj
dense = to_dense_adj(edge_index).squeeze(0)
features = sample.x

# %%
dense.shape

# %%
len(dataset4)

# %%

import torch
# import linear from torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.utils import scatter

def squareroot_degree_inverse_undirected(adj):
    row_sum = torch.sum(adj, dim=1)
    row_sum = row_sum.pow(-0.5)
    D_12 = torch.eye(row_sum.shape[0]).to(row_sum.device) * row_sum
    return D_12


class DenseGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        x = torch.matmul(x, self.weight)
        x = torch.matmul(adj, x)
        return x + self.bias


class BareboneDenseGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout, readout_pooling,in_channels=162, **kwargs):
        super().__init__()
        
        self.dropout = dropout
        self.pooling = readout_pooling
        self.layers = torch.nn.ModuleList()
        
        self.initial_layer = DenseGCNConv(in_channels, hidden_channels)
        
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers-1):
            if i == 0:
                self.layers.append(DenseGCNConv(in_channels, hidden_channels))
            else:
                self.layers.append(DenseGCNConv(hidden_channels, hidden_channels))
            
        # self.classification_layer = Linear(hidden_channels, 1)
        
        
    def forward_initial_layer(self, x, normalized_adj):
        X = self.initial_layer(x, normalized_adj)
        X = F.dropout(X, training=self.training, p=self.dropout) # droput before relu faster
        X = F.relu(X)
        return X

    def forward(self, X, normalized_adj, initial_normalized_adj,GSL_skip_ratio):
        if GSL_skip_ratio is not None and GSL_skip_ratio>0:  # add skip connections
            for layer in self.layers:
                X_skip = layer(X, initial_normalized_adj)
                X_nonskip = layer(X, normalized_adj)
                X = F.relu(GSL_skip_ratio*X_skip + (1-GSL_skip_ratio)*X_nonskip)
                X = F.dropout(X, training=self.training, p=self.dropout)
        else:   
            for layer in self.layers:
                X = layer(X, normalized_adj)
                X = F.dropout(X, training=self.training, p=self.dropout)
                X = F.relu(X)
            
        # logits = self.classification_layer(X)

        return X
    

# %%
class StructureLearner(torch.nn.Module):
    def __init__(self, num_heads, input_channels, **kwargs):
        super().__init__()
        self.weight_tensor = torch.nn.Parameter(torch.Tensor(num_heads, input_channels)).unsqueeze(1)
        self.weight_tensor = torch.nn.Parameter(torch.nn.init.xavier_uniform_(self.weight_tensor))
        self.attention_threshold_epsilon = 0.3

    def forward(self, X):
        X = X.unsqueeze(0) 
        
        # hadamard product by broadcasting
        # (h, 1, dim) * (1, n, dim) -> (h, n, dim)
        X = torch.multiply(self.weight_tensor, X)
        Xnorm = F.normalize(X, p=2, dim=-1)
        attention = torch.bmm(Xnorm, Xnorm.transpose(1,2).detach())
        attention = torch.mean(attention, dim=0)
        attention[attention > self.attention_threshold_epsilon] = 0

        return attention

# %%
hidden_channels = 128
# model = BareboneDenseGCN(hidden_channels=hidden_channels, num_layers=2, dropout=0.5, readout_pooling='mean').cuda()


# %%
from models.GCN import GCN
edgeindex_model = GCN(hidden_channels=hidden_channels, num_layers=2, dropout=0.5, readout_pooling='mean').cuda()

# %%
sample

# %%
dense.shape

# %%
structureLearner = StructureLearner(num_heads=4, input_channels=hidden_channels).cuda()

# %%
sample = sample.cuda()

# %%
from models.BinaryClassifierReadOutLayer import BinaryClassifierReadOutLayer
head = BinaryClassifierReadOutLayer(hidden_channels=hidden_channels, num_layers=1, pooling='sum').cuda()

# %%
features.shape
batch = torch.zeros(sample.x.shape[0]).long().cuda()

# %%
num_samples = 100
batch_size = 1
iterations = 5
edgeindex_model.train()
y = torch.tensor([[1.0]]).cuda()
# optimizer1 = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(edgeindex_model.parameters(), lr=0.001)
import time
# %%

    
    
# print mean and std
# %%
from models.GraphStructureLearner import GraphStructureLearner
sparse_gsl = GraphStructureLearner(num_heads=4, input_channels=hidden_channels, attention_threshold_epsilon=0.3).cuda()
has_converged = torch.tensor([False]).cuda()

from models.BinaryClassifierReadOutLayer import BinaryClassifierReadOutLayer
head2 = BinaryClassifierReadOutLayer(hidden_channels=hidden_channels, num_layers=1, pooling='sum').cuda()
torch.cuda.empty_cache()
# %%
from tqdm.auto import tqdm
# run model
times = []
time_forward = []
time_backward = []
for j in tqdm(range(50)):
    torch.cuda.synchronize()
    start = time.time()
    for i in range(num_samples):
        torch.cuda.synchronize()
        start_forward = time.time()
        features = sample.x
        h = edgeindex_model.initial_layer(features, sample.edge_index)
        for iter in range(iterations):
            edge_index, edge_weights = sparse_gsl(h, batch, has_converged) # note that this uses the minibatch implementation (iterate over graphs in minibatch seperately), I checked that most time consuming is the bmm and the torch.argwhere, therefore the batch overhead should not be significant
            h, placeholder = edgeindex_model(features, edge_index, edge_weights, sample.edge_index, GSL_skip_ratio=0.3)
        
        logits = head(h, batch)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        torch.cuda.synchronize()
        end_forward = time.time()
        time_forward.append(end_forward-start_forward)
        torch.cuda.synchronize()
        start_backward = time.time()
        if i % batch_size == 0 and i > 0:
            loss.backward()
            optimizer2.step()
            optimizer2.zero_grad()
        else:
            loss.backward(retain_graph=True)
        torch.cuda.synchronize()
        end_backward = time.time()
        time_backward.append(end_backward-start_backward)
        
    torch.cuda.synchronize()
    end = time.time()
    times.append(end-start)
    # if j!=0:
        # print('Mean: {:.4f}, Std: {:.4f}'.format(sum(times)/len(times), sum([(x - sum(times)/(len(times)-1))**2 for x in times])**0.5))

print('Mean Final: {:.4f}, Std: {:.4f}'.format(sum(times)/len(times), sum([(x - sum(times)/(len(times)-1))**2 for x in times])**0.5))
print('Mean Forward: {:.4f}, Std: {:.4f}'.format(sum(time_forward)/len(time_forward), sum([(x - sum(time_forward)/(len(time_forward)-1))**2 for x in time_forward])**0.5))
print('Mean Backward: {:.4f}, Std: {:.4f}'.format(sum(time_backward)/len(time_backward), sum([(x - sum(time_backward)/(len(time_backward)-1))**2 for x in time_backward])**0.5))

# 10521MiB

# 29.62 sekunden pro iteration also pro 100 -> 3.37 per second
# GraphGLOW in iteration: 2.24 it per second, one it has 12 samples, 5 iter 26 per second
# Mean Final: 30.1984, Std: 4.7726
# Mean Forward: 0.1065, Std: 0.9703
# Mean Backward: 0.1954, Std: 0.5279