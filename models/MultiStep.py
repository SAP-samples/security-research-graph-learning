import torch 
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
from models.GCN import squareroot_degree_inverse_undirected_minibatch

# Idea
# Adj = H~*H~T > threshold
# H~ = Softmax(H * Q) * R + H
# R are n vectors shifting H into some directions
# Q is an attention matrix 
# H is nxd
# Q is dxh
# R is hxd 

    
class MultiStep(torch.nn.Module):
    def __init__(self, gslgnn_class, gslgnn_args, multistepgnn_class, multistepgnn_args, gsl_class, gsl_args, GSL_skip_ratio,  MULTISTEP_num_steps, MULTISTEP_jk, MULTISTEP_jk_aggr,**kwargs) -> None:
        super().__init__()
        self.multistep_gnn = multistepgnn_class(**multistepgnn_args)
        self.gsl_gnn_list = torch.nn.ModuleList()
        self.gsl_list = torch.nn.ModuleList()
        for i in range(MULTISTEP_num_steps):
            if i==0:
                self.gsl_gnn_list.append(gslgnn_class(**gslgnn_args, has_extra_conv=True))
            else:
                self.gsl_gnn_list.append(gslgnn_class(**gslgnn_args, in_channels=gslgnn_args['hidden_channels']))
                
            self.gsl_list.append(gsl_class(**gsl_args))
        
        self.MULTISTEP_jk = MULTISTEP_jk
        self.MULTISTEP_jk_aggr = MULTISTEP_jk_aggr
        if len(self.gsl_gnn_list) :
            self.gsl_gnn_requires_normalized_adj = self.gsl_gnn_list[0].requires_normalized_adj
        self.multistep_gnn_requires_normalized_adj = self.multistep_gnn.requires_normalized_adj
        self.GSL_skip_ratio = GSL_skip_ratio
        
        # self.test_layer1 = torch.nn.Linear(128, 128)
        # self.test_layer2 = torch.nn.Linear(128, 128)
        # self.test_layer3 = torch.nn.Linear(128, 128)
        # self.linkpred = LinkPred()
        
        # self.test_layer4 = torch.nn.Linear(128, 128)
        # self.test_layer5 = torch.nn.Linear(128, 128)
        # self.test_layer6 = torch.nn.Linear(128, 128)
    
    def freeze_gsl_and_gsl_gnn(self):
        # freeze 
        print('=== FREEZE GSL AND GSL GNNs ===')
        for gnn in self.gsl_gnn_list:
            for param in gnn.parameters():
                param.requires_grad = False
        for gsl in self.gsl_list:
            for param in gsl.parameters():
                param.requires_grad = False
    
    def forward(self, X, initial_adj, minibatch_mask):
        
        jk = []
        if len(self.gsl_gnn_list):
            initial_gsl_adj = squareroot_degree_inverse_undirected_minibatch(initial_adj) if self.gsl_gnn_requires_normalized_adj else initial_adj
        initial_multistep_adj = squareroot_degree_inverse_undirected_minibatch(initial_adj) if self.multistep_gnn_requires_normalized_adj else initial_adj
        
        h_classif = self.multistep_gnn(X, initial_multistep_adj, initial_multistep_adj, 0, minibatch_mask)
        # return h_classif
        # h = self.test_layer1(h_classif)
        # h = F.relu(h)
        # h = self.test_layer2(h)
        # h = F.relu(h)
        # h = self.test_layer3(h)
        # h = F.relu(h)
        # return h
        # h = self.test_layer4(h)
        # h = F.relu(h)
        # h = self.test_layer5(h)
        # h = F.relu(h)
        # h = self.test_layer6(h)
        # h = F.relu(h)
        
       
        jk.append(h_classif)
        if len(self.gsl_gnn_list):
            h = self.gsl_gnn_list[0].forward_initial_layer(X, initial_gsl_adj, minibatch_mask)
        else: 
            return h_classif
        for i, (gsl_gnn, gsl) in enumerate(zip(self.gsl_gnn_list, self.gsl_list)):
            adj = gsl(h)
            adj = squareroot_degree_inverse_undirected_minibatch(adj) if self.gsl_gnn_requires_normalized_adj else adj
            h = gsl_gnn(h, adj, initial_multistep_adj, self.GSL_skip_ratio, minibatch_mask)
            
            # skip ratio 0
            h_classif = self.multistep_gnn(X, adj,initial_multistep_adj, 0, minibatch_mask)
            jk.append(h_classif)
            
        if self.MULTISTEP_jk:
            if self.MULTISTEP_jk_aggr == 'mean':
                return torch.mean(torch.stack(jk), dim=1)
            elif self.MULTISTEP_jk_aggr == 'sum':
                return torch.sum(torch.stack(jk), dim=1)
            elif self.MULTISTEP_jk_aggr == 'cat':
                return torch.cat(jk, dim=-1)  # dim dimension, e.g. if minibatching -1 needed
        else:
            return h_classif
    
    def reset_multistep_head_parameters(self):
        self.multistep_gnn.reset_parameters()
    
    def reset_parameters(self):
        self.multistep_gnn.reset_parameters()
        for gnn in self.gsl_gnn_list:
            gnn.reset_parameters()
        for gsl in self.gsl_list:
            gsl.reset_parameters()
        