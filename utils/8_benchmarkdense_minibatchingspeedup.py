# %%
import torch 
torch.backends.cudnn.benchmark = True
from models.BareboneGCN import BareboneGCN
from CodeGraphDataset import CodeGraphDataset_InMemory
dataset4 = CodeGraphDataset_InMemory(pt_folder='codegraphs/diversevul/v2_undirected_withdegreecount', split='train', 
                                          cross_val_valfold_idx=0, 
                                          is_cross_val=True, cross_val_train_fraction=0.01, DS_type='all')

# %% [markdown]
# # Test GraphGLOW with DENSE
# Self CPU time total: 1.086s
# Self CUDA time total: 611.991ms

# 
#Self CPU time total: 932.622ms
# Self CUDA time total: 535.430ms
#
#
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

def squareroot_degree_inverse_undirected_minibatch(adj):
    row_sum = torch.sum(adj, dim=2)
    row_sum = row_sum.pow(-0.5) # shape batch_size x num_nodes
    # replace inf with 0
    row_sum[row_sum == float('inf')] = 0
    D_12 = torch.diag_embed(row_sum)
    return D_12 @ adj @ D_12  # shape batch_size x num_nodes x num_nodes


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
        self.weight_tensor = torch.nn.Parameter(torch.Tensor(num_heads, input_channels)).unsqueeze(1).unsqueeze(0) # batching
        self.weight_tensor = torch.nn.Parameter(torch.nn.init.xavier_uniform_(self.weight_tensor))
        self.attention_threshold_epsilon = 0.3

    def forward(self, X):
        X = X.unsqueeze(1) # (8, 1, 1000, 128), (1, 4, 1, 128)
        
        # hadamard product by broadcasting
        # (h, 1, dim) * (1, n, dim) -> (h, n, dim)
        X = torch.multiply(self.weight_tensor, X)
        Xnorm = F.normalize(X, p=2, dim=-1)
        attention = torch.matmul(Xnorm, Xnorm.transpose(2,3).detach())
        attention = torch.mean(attention, dim=1)
        attention[attention > self.attention_threshold_epsilon] = 0

        return attention

# %%
hidden_channels = 128
model = BareboneDenseGCN(hidden_channels=hidden_channels, num_layers=2, dropout=0.5, readout_pooling='mean').cuda()
model = torch.compile(model, fullgraph=True, dynamic=False)

# %%
from models.GCN import GCN
edgeindex_model = GCN(hidden_channels=hidden_channels, num_layers=2, dropout=0.5, readout_pooling='mean').cuda()

# %%
sample

# %%
dense.shape

# %%
structureLearner = StructureLearner(num_heads=4, input_channels=hidden_channels).cuda()
structureLearner = torch.compile(structureLearner, fullgraph=True, dynamic=False)

# %%
sample = sample.cuda()

# %%
from models.BinaryClassifierReadOutLayer import BinaryClassifierReadOutLayerForDenseBatch
head = BinaryClassifierReadOutLayerForDenseBatch(hidden_channels=hidden_channels, num_layers=1, pooling='sum').cuda()

# %%
features.shape
batch = torch.zeros(sample.x.shape[0]).long().cuda()

# %%
num_samples = 100
batch_size = 1
iterations = 5
model.train()
edgeindex_model.train()
y = torch.tensor([[1.0]]).cuda()
optimizer1 = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(edgeindex_model.parameters(), lr=0.001)
import time
# %%


def fill_remaining_dense_features(dense, features,max_size=1000):
    dense_filled = torch.zeros_like(dense)
    
    
times = []
time_forward = []
time_backward = []
max_size = 1000
global_zero_adj_tensors = torch.zeros((batch_size,max_size,max_size)).cuda()
global_zero_feature_tensors = torch.zeros((batch_size,max_size,162)).cuda()
global_zero_feature_tensor_mask = torch.zeros((batch_size,max_size,162)).cuda()
def simulate_dataloaderbatching_preprocessing():
    global global_zero_adj_tensors
    global global_zero_feature_tensors
    # global global_zero_feature_tensor_mask
    
    global_zero_adj_tensors.zero_()
    global_zero_feature_tensors.zero_()
    # global_zero_feature_tensor_mask.zero_()
    
    
    for i in range(batch_size):
        sample_copy = sample.clone()
        # make a dense pytorch matrix out of it
        dense = to_dense_adj(sample_copy.edge_index)
        features = sample_copy.x
        global_zero_adj_tensors[i,:dense.shape[1],:dense.shape[2]] = dense.squeeze(0)
        global_zero_feature_tensors[i,:features.shape[0],:] = features
        # global_zero_feature_tensor_mask[i,...] = 1  # we should not need a mask because the non existent nodes have no edges and therefore should remain zero
    
    return global_zero_adj_tensors, global_zero_feature_tensors  


# warm up
adj, features = simulate_dataloaderbatching_preprocessing()
normalized_dense_initial = squareroot_degree_inverse_undirected_minibatch(adj)
h = model.forward_initial_layer(features, normalized_dense_initial)
adj = structureLearner(h)
normalized_dense = squareroot_degree_inverse_undirected_minibatch(adj)
h = model(features, normalized_dense, normalized_dense_initial, GSL_skip_ratio=0.3)


scaler = torch.cuda.amp.GradScaler()
from tqdm.auto import tqdm
y = torch.ones((batch_size,1)).cuda()
for _ in tqdm(range(100)):
    torch.cuda.synchronize()
    start = time.time()
    for i in range(num_samples//batch_size):
        torch.cuda.synchronize()
        # start_forward = time.time()
        
        adj, features = simulate_dataloaderbatching_preprocessing()
        normalized_dense_initial = squareroot_degree_inverse_undirected_minibatch(adj)
        
        # dense = to_dense_adj(edge_index).squeeze(0)
        # normalized_dense_initial = squareroot_degree_inverse_undirected(dense)
        
        # features = sample.x
        with torch.autocast(device_type='cuda'): # dtype=torch.float16
            h = model.forward_initial_layer(features, normalized_dense_initial)
            for iter in range(iterations):
                adj = structureLearner(h)
                normalized_dense = squareroot_degree_inverse_undirected_minibatch(adj)
                h = model(features, normalized_dense, normalized_dense_initial, GSL_skip_ratio=0.3) # in graphglow they use same h as the one from the initial layer
                # h has dim batch_size x num_nodes x hidden_channels
            
            logits = head(h, batch)
            loss = F.binary_cross_entropy_with_logits(logits, y)
        
        
        # torch.cuda.synchronize()
        # end_forward= time.time()
        # torch.cuda.synchronize()
        # start_backward = time.time()
        # loss.backward()
        # optimizer1.step()
        # optimizer1.zero_grad()
        # loss.backward(retain_graph=True)
        #  if i % batch_size == 0 and i > 0:
        #     loss.backward()
        #     optimizer1.step()
        #     optimizer1.zero_grad()
          
        # else:
        #     loss.backward(retain_graph=True)
        
        scaler.scale(loss).backward()
        # if i % batch_size == 0 and i > 0:
        # scaler.unscale_(optimizer1)
        scaler.step(optimizer1)
        scaler.update()
        optimizer1.zero_grad()
        # Updates the scale for next iteration.
        
            
        # torch.cuda.synchronize()
        # end_backward = time.time()
        # time_forward.append(end_forward-start_forward)
        # time_backward.append(end_backward-start_backward)
        
    torch.cuda.synchronize()
    
    end = time.time()
    times.append(end-start)
    
    
# print mean and std
print('Mean: {:.4f}, Std: {:.4f}'.format(sum(times)/len(times), sum([(x - sum(times)/(len(times)-1))**2 for x in times])**0.5))
# print('Mean Forward: {:.4f}, Std Forward: {:.4f}'.format(sum(time_forward)/len(time_forward), sum([(x - sum(time_forward)/(len(time_forward)-1))**2 for x in time_forward])**0.5))
# print('Mean Backward: {:.4f}, Std Backward: {:.4f}'.format(sum(time_backward)/len(time_backward), sum([(x - sum(time_backward)/(len(time_backward)-1))**2 for x in time_backward])**0.5))
# %%
from models.GraphStructureLearner import GraphStructureLearner
sparse_gsl = GraphStructureLearner(num_heads=4, input_channels=hidden_channels, attention_threshold_epsilon=0.3).cuda()
has_converged = torch.tensor([False]).cuda()

from models.BinaryClassifierReadOutLayer import BinaryClassifierReadOutLayer
head2 = BinaryClassifierReadOutLayer(hidden_channels=hidden_channels, num_layers=1, pooling='sum').cuda()


# Without compile
# for 96: -># 129 per sec
# Mean: 0.7446, Std: 0.7964 
# Mean Forward: 0.0292, Std Forward: 0.6742
# Mean Backward: 0.0328, Std Backward: 0.1274

# alows us to go up to 128 minibatch size (12-13gb)

# with compile:
#  torch.compile(model, fullgraph=True, dynamic=False,) for head and model
# for 96: -> 162
# Mean: 0.5941, Std: 1.3303
# Mean Forward: 0.0231, Std Forward: 0.0300
# Mean Backward: 0.0264, Std Backward: 1.3164

# Mean: 0.5926, Std: 1.2848
# Mean Forward: 0.0231, Std Forward: 0.0316
# Mean Backward: 0.0263, Std Backward: 1.2679


# With mixed precision:
# Mean: 0.3461, Std: 2.3997 ->277
# Mean Forward: 0.0166, Std Forward: 1.6426
# Mean Backward: 0.0122, Std Backward: 0.7604

#
# Mean: 1.2163, Std: 1.1515
# Mean Forward: 0.0058, Std Forward: 0.2292
# Mean Backward: 0.0064, Std Backward: 0.1496