import torch 
# import dataset from torch geometric
from torch_geometric.data import Data, DataLoader, HeteroData
from tqdm.auto import tqdm 
import os 
from pathlib import Path
from copy import deepcopy

# import torch geometric to dense
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import to_dense_adj
def column_normalize_dense_features_torch(mx):  
        '''Column-normalize dense features'''
        colsum = torch.sum(mx, axis=0)
        c_inv = torch.pow(colsum, -1)
        c_inv[torch.isinf(c_inv)] = 0.
        c_mat_inv = torch.diag(c_inv)
        mx = mx.mm(c_mat_inv)
        return mx

max_deg = 1000 // 2
max_triag = 250 //2
for file in tqdm(os.listdir("codegraphs/diversevul/v2_directed_withdegreecount_heterogeneous")):
    # skip if exiss already
    # if Path("v2_undirected_withdegreecount/" + file).is_file() and Path("v2_undirected_withdegreecount_heterogeneous/" + file).is_file():
        # continue
    # 44 categories
    #    # in, out, total * 4 (4 edge types) + total degree of all
        # degrees: 13, triangle_counts: 5, 44 labels, 100 word embedding
    data = torch.load("codegraphs/diversevul/v2_directed_withdegreecount_heterogeneous/" + file)
    len_nodes = len(data['node'].x)
    
    adj_matrices = [to_dense_adj(data[edge_type].edge_index, max_num_nodes=len_nodes).squeeze(0) for edge_type in data.edge_types if  edge_type[1] != 'TOTAL']
    
      # get node degrees per edge type and in-/out-degrees
        # in1, in2, in3, out1 out2 out3, total1, total2, total3, total4...
    num_types = len(adj_matrices)
    degrees = torch.zeros((len_nodes, num_types*3+1), dtype=torch.float32, device='cpu')
    for i, adj_matrix in enumerate(adj_matrices):
        degrees[:,i] = torch.sum(adj_matrix, dim=1) # out-degrees
        degrees[:,i+num_types] = torch.sum(adj_matrix, dim=0)   # in-degrees
        degrees[:,i+2*num_types] = degrees[:,i] + degrees[:,i+num_types]  # total degrees
        
    degrees[:,-1] = torch.sum(degrees[:,-5:], dim=1) # in + out degree so doubled # total degrees (also total1, 2, 3 because sparse does not support range index)
    # normalize degrees
    degrees = degrees / max_deg
    
    # print('max degree count', degrees[:,-1].max())
    # if max_deg < degrees[:,-1].max():
    #     max_deg = degrees[:,-1].max()
    #     print('max degree count', max_deg)
    
    # add triangle counts per edge type and total triangle counts
    # with triangle count and degree count, we could calculate clustering coefficient
    triangle_counts = torch.zeros((len_nodes, num_types+1), device='cpu')
    for i, adj_matrix in enumerate(adj_matrices):
        undirected_adj = (adj_matrix + adj_matrix.T).to_sparse()
        # sparse = undirected_adj.to_sparse()
        # unnormalized triangle count
        A_times_A = torch.sparse.mm(undirected_adj,undirected_adj)
        # sum(mul(A,B), dim=1) faster than trace(mm(A,B))
        triangle_counts[:,i] = torch.sparse.sum(torch.mul(A_times_A,undirected_adj), dim=1).to_dense()
    
    triangle_counts[:,-1] = torch.sum(triangle_counts[:,:-1], dim=1)
    # normalize triangle counts
    triangle_counts = triangle_counts / max_triag
    
    X = torch.cat([degrees, triangle_counts], dim=1)
    # X = column_normalize_dense_features_torch(mx=X)
    
    data['node'].x = torch.cat((X,data['node'].x[:,X.shape[1]:]),dim=1)
    # print('max triangle count', triangle_counts[:,-1].max())
    # if max_triag < triangle_counts[:,-1].max():
    #     max_triag = triangle_counts[:,-1].max()
    #     print('max triangle count', max_triag)
        
    # save data v2_directed_withdegreecount_heterogeneous_unnormalized
    torch.save(data, "codegraphs/diversevul/v2_directed_withdegreecount_heterogeneous_unnormalized/" + file)
    
        