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


samples = [sample.cuda()]
samples.extend([dataset4[i].cuda() for _ in range(7)])
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
model = BareboneDenseGCN(hidden_channels=hidden_channels, num_layers=2, dropout=0.5, readout_pooling='mean').cuda()


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
batch_infos = [ torch.zeros(samples[i].x.shape[0]).long().cuda() for i in range(8)]
# %%
num_samples = 100
batch_size = 60
iterations = 5
model.train()
edgeindex_model.train()
y = torch.tensor([[1.0]]).cuda()
optimizer1 = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(edgeindex_model.parameters(), lr=0.001)
import time
# %%


def fill_remaining_dense(dense, max_size=1000):
    dense_filled = torch.zeros_like(dense)
    return dense+dense_filled, dense.shape[0]
    
times = []
time_forward = []
time_backward = []
from tqdm.auto import tqdm
for _ in tqdm(range(100)):
    torch.cuda.synchronize()
    start = time.time()
    for i in range(num_samples):
        edge_index = samples[0].edge_index
        features = samples[0].x
        torch.cuda.synchronize()
        start_forward = time.time()
        dense = to_dense_adj(edge_index).squeeze(0)
        normalized_dense_initial = squareroot_degree_inverse_undirected(dense)
        
        
        h = model.forward_initial_layer(features, normalized_dense_initial)
        for iter in range(iterations):
            adj = structureLearner(h)
            normalized_dense = squareroot_degree_inverse_undirected(adj)
            h = model(features, normalized_dense, normalized_dense_initial, GSL_skip_ratio=0.3) # in graphglow they use same h as the one from the initial layer
        
        logits = head(h, batch_infos[0])
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        torch.cuda.synchronize()
        end_forward= time.time()
        torch.cuda.synchronize()
        start_backward = time.time()
        if i % batch_size == 0 and i > 0:
            loss.backward()
            optimizer1.step()
            optimizer1.zero_grad()
        else:
            loss.backward(retain_graph=True)
        torch.cuda.synchronize()
        end_backward = time.time()
        time_forward.append(end_forward-start_forward)
        time_backward.append(end_backward-start_backward)
        
    torch.cuda.synchronize()
    
    end = time.time()
    times.append(end-start)
    
    
# print mean and std
print('Mean: {:.4f}, Std: {:.4f}'.format(sum(times)/len(times), sum([(x - sum(times)/(len(times)-1))**2 for x in times])**0.5))
print('Mean Forward: {:.4f}, Std Forward: {:.4f}'.format(sum(time_forward)/len(time_forward), sum([(x - sum(time_forward)/(len(time_forward)-1))**2 for x in time_forward])**0.5))
print('Mean Backward: {:.4f}, Std Backward: {:.4f}'.format(sum(time_backward)/len(time_backward), sum([(x - sum(time_backward)/(len(time_backward)-1))**2 for x in time_backward])**0.5))
# %%
from models.GraphStructureLearner import GraphStructureLearner
sparse_gsl = GraphStructureLearner(num_heads=4, input_channels=hidden_channels, attention_threshold_epsilon=0.3).cuda()
has_converged = torch.tensor([False]).cuda()

from models.BinaryClassifierReadOutLayer import BinaryClassifierReadOutLayer
head2 = BinaryClassifierReadOutLayer(hidden_channels=hidden_channels, num_layers=1, pooling='sum').cuda()

# 1.49 it per second (1 it 100 samples, 5 Graphglow iterations per sample) -> 149 per second

# max 333mib per batch (with 2000 loaded graphs)

# Mean: 1.6000 per second (1 it 100 samples, 5 Graphglow iterations per sample), # while true; do clear; nvidia-smi; sleep 1; done

# for 100: -> 51 per second
# Mean: 1.9536, Std: 3.6958
# Mean Forward: 0.0141, Std Forward: 1.1141
# Mean Backward: 0.0054, Std Backward: 0.2571


# for 8  -> 102 per second
# Mean: 0.0780, Std: 0.6656
# Mean Forward: 0.0057, Std Forward: 0.5859
# Mean Backward: 0.0040, Std Backward: 0.0890