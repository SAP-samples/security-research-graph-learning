import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
# torch.backends.cudnn.benchmark = True



def squareroot_degree_inverse_undirected_minibatch(adj):
    # with torch.no_grad():
    row_sum = torch.sum(adj, dim=2) #+ torch.finfo(adj.dtype).eps
    mask = row_sum == 0
    row_sum[mask] = 1
    row_sum = row_sum.pow(-0.5) # shape batch_size x num_nodes
    # replace inf with 0
    row_sum[mask] = 0
        # D_12 = torch.diag_embed(row_sum)
    # faster than #D_12 @ adj @ D_12 
    # return D_12 @ adj @ D_12
    return torch.mul(torch.mul(adj,row_sum.unsqueeze(-2)),row_sum.unsqueeze(-1)) #adj #D_12 @ adj @ D_12  # shape batch_size x num_nodes x num_nodes

class DenseGINConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseGINConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear1 = torch.nn.Linear(in_channels, out_channels)
        self.batchnorm1 = torch.nn.BatchNorm1d(out_channels)
        self.linear2 = torch.nn.Linear(out_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        self.batchnorm1.reset_parameters()

    def forward(self, x, unnormalized_adj):
        x = torch.matmul(unnormalized_adj, x)
        x = self.linear1(x)
        shape = x.shape
        x = self.batchnorm1(x.view(-1,shape[-1])).view(*shape)
        x = F.relu(x)
        x = self.linear2(x)
        return x


class DenseGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        # self.linear = torch.nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, normalized_adj):
        # !!! adj needs to be normalized with squareroot_degree_inverse_undirected_minibatch beforehand !!! 
        x = torch.matmul(normalized_adj, x)
        x = torch.matmul(x, self.weight)
        return x + self.bias



class GIN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout, readout_pooling,in_channels=162, **kwargs):
        super().__init__()
        
        self.dropout = dropout
        self.pooling = readout_pooling
        self.layers = torch.nn.ModuleList()
        
        self.initial_layer = DenseGINConv(in_channels, hidden_channels)
        
        self.skip_layers = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(DenseGINConv(in_channels, hidden_channels))
                self.skip_layers.append(DenseGINConv(in_channels, hidden_channels))
            else:
                self.layers.append(DenseGINConv(hidden_channels, hidden_channels))
                self.skip_layers.append(DenseGINConv(hidden_channels, hidden_channels))
        
        
        # self.classification_layer = Linear(hidden_channels, 1)
        
        
    def forward_initial_layer(self, x, normalized_adj, minibatch_mask):
        X = self.initial_layer(x, normalized_adj)
        X = F.dropout(X, training=self.training, p=self.dropout) # droput before relu faster
        X = F.relu(X)
        return torch.multiply(minibatch_mask, X)

    def forward(self, X, normalized_adj, initial_normalized_adj,GSL_skip_ratio, minibatch_mask):
        
        if GSL_skip_ratio is not None and GSL_skip_ratio>0:  # add skip connections
            for layer, skip_layer in zip(self.layers, self.skip_layers):
                X_skip = F.relu(skip_layer(X, initial_normalized_adj))
                X_nonskip = F.relu(layer(X, normalized_adj))
                X = GSL_skip_ratio*X_skip + (1-GSL_skip_ratio)*X_nonskip
                X = F.dropout(X, training=self.training, p=self.dropout)
        else:   
            for layer in self.layers:
                X = layer(X, normalized_adj)
                X = F.relu(X)
                X = F.dropout(X, training=self.training, p=self.dropout)
            
        return torch.multiply(minibatch_mask, X)
    
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout, readout_pooling,in_channels=162, **kwargs):
        super().__init__()
        
        self.dropout = dropout
        self.pooling = readout_pooling
        self.layers = torch.nn.ModuleList()
        
        self.initial_layer = DenseGCNConv(in_channels, hidden_channels)
        
        self.skip_layers = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(DenseGCNConv(in_channels, hidden_channels))
                self.skip_layers.append(DenseGCNConv(in_channels, hidden_channels))
            else:
                self.layers.append(DenseGCNConv(hidden_channels, hidden_channels))
                self.skip_layers.append(DenseGCNConv(hidden_channels, hidden_channels))
        
        
        # self.classification_layer = Linear(hidden_channels, 1)
        
        
    def forward_initial_layer(self, x, normalized_adj, minibatch_mask):
        X = self.initial_layer(x, normalized_adj)
        X = F.dropout(X, training=self.training, p=self.dropout) # droput before relu faster
        X = F.relu(X)
        return torch.multiply(minibatch_mask, X)

    def forward(self, X, normalized_adj, initial_normalized_adj,GSL_skip_ratio, minibatch_mask):
        
        if GSL_skip_ratio is not None and GSL_skip_ratio>0:  # add skip connections
            for layer, skip_layer in zip(self.layers, self.skip_layers):
                X_skip = F.relu(skip_layer(X, initial_normalized_adj))
                X_nonskip = F.relu(layer(X, normalized_adj))
                X = GSL_skip_ratio*X_skip + (1-GSL_skip_ratio)*X_nonskip
                X = F.dropout(X, training=self.training, p=self.dropout)
        else:   
            for layer in self.layers:
                X = layer(X, normalized_adj)
                X = F.relu(X)
                X = F.dropout(X, training=self.training, p=self.dropout)
                
            
        return torch.multiply(minibatch_mask, X)
    
    def reset_parameters(self):
        self.initial_layer.reset_parameters()
        for conv in self.layers:
            conv.reset_parameters()
        for conv in self.skip_layers:
            conv.reset_parameters()
        

        
class GNNAdapted(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout, readout_pooling, conv, in_channels=162, jk=True, has_extra_conv=False, **kwargs):
        super().__init__()
        
        self.dropout = dropout
        self.pooling = readout_pooling
        self.layers = torch.nn.ModuleList()
        self.hidden_channels = hidden_channels
        if has_extra_conv:
            self.initial_layer = conv(in_channels, hidden_channels)
            self.initial_batch_norm = torch.nn.BatchNorm1d(hidden_channels)
        
        self.skip_layers = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.batch_norms_skip = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0 and not has_extra_conv:
                self.layers.append(conv(in_channels, hidden_channels))
                self.skip_layers.append(conv(in_channels, hidden_channels))
            else:
                self.layers.append(conv(hidden_channels, hidden_channels))
                self.skip_layers.append(conv(hidden_channels, hidden_channels))

            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))
            self.batch_norms_skip.append(torch.nn.BatchNorm1d(hidden_channels))
        # self.classification_layer = Linear(hidden_channels, 1)
        self.jk = jk
        
    def forward_initial_layer(self, x, un_normalized_adj, minibatch_mask): # initial layer lower dim
        # X = self.initial_layer(x)
        # shape = X.shape
        # X = self.initial_batch_norm(X.view(-1,shape[-1])).view(*shape)
        # X = F.relu(X)
        # X = F.dropout(X, training=self.training, p=self.dropout)
        X = self.initial_layer(x, un_normalized_adj)
        shape = X.shape
        X = self.initial_batch_norm(X.view(-1,shape[-1])).view(*shape)
        X = F.relu(X)
        X = F.dropout(X, training=self.training, p=self.dropout) # droput before relu faster
        return torch.multiply(minibatch_mask, X)
    

    def forward(self, X, normalized_adj, initial_normalized_adj,GSL_skip_ratio, minibatch_mask,jk=False):
        
        jk_list = [X]
        if GSL_skip_ratio is not None and GSL_skip_ratio>0:  # add skip connections
            for i,(layer, skip_layer, batch_norm,batch_norm_skip) in enumerate(zip(self.layers, self.skip_layers, self.batch_norms, self.batch_norms_skip)):
                X_skip = skip_layer(X, initial_normalized_adj)
                X_nonskip = layer(X, normalized_adj)
                
                shape = X_nonskip.shape
                X_skip = batch_norm_skip(X_skip.view(-1,shape[-1])).view(*shape)
                X_nonskip = batch_norm(X_nonskip.view(-1,shape[-1])).view(*shape)
                # if not i == len(self.layers)-1:
                #     X_skip = F.relu(X_skip)
                #     X_nonskip = F.relu(X_nonskip)
                    
                X = GSL_skip_ratio*X_skip + (1-GSL_skip_ratio)*X_nonskip
                X = F.dropout(X, training=self.training, p=self.dropout)
                jk_list.append(X)
        else:   
            for i,(layer, batch_norm) in enumerate(zip(self.layers, self.batch_norms)):
                X = layer(X, normalized_adj)
                shape = X.shape
                X = batch_norm(X.view(-1,shape[-1])).view(*shape)
                # if not i == len(self.layers)-1:
                #     X = F.relu(X)
                X = F.dropout(X, training=self.training, p=self.dropout)
                torch.multiply(minibatch_mask, X)
                jk_list.append(X)
        if jk:
            X = torch.cat(jk_list, dim=1)
        return torch.multiply(minibatch_mask, X)
    
    def reset_parameters(self):
        self.initial_layer.reset_parameters()
        for conv in self.layers:
            conv.reset_parameters()
        for conv in self.skip_layers:
            conv.reset_parameters()
        for batch_norm in self.batch_norms:
            batch_norm.reset_parameters()
        for batch_norm in self.batch_norms_skip:
            batch_norm.reset_parameters()
            
class GCNAdapted(GNNAdapted):
    def __init__(self, hidden_channels, num_layers, dropout, readout_pooling,in_channels=162, jk=True, has_extra_conv=False, **kwargs):
        super().__init__(hidden_channels=hidden_channels, num_layers=num_layers, dropout=dropout, readout_pooling=readout_pooling, in_channels=in_channels, jk=jk, 
                         conv=DenseGCNConv, has_extra_conv=has_extra_conv, **kwargs)
        self.requires_normalized_adj = True

class GINAdapted(GNNAdapted):
    def __init__(self, hidden_channels, num_layers, dropout, readout_pooling,in_channels=162, jk=True, has_extra_conv=False, **kwargs):
        super().__init__(hidden_channels=hidden_channels, num_layers=num_layers, dropout=dropout, readout_pooling=readout_pooling, in_channels=in_channels, jk=jk, 
                         conv=DenseGINConv, has_extra_conv=has_extra_conv, **kwargs)
        self.requires_normalized_adj = False
        

        
        

    



