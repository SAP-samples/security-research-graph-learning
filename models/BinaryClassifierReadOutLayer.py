import torch 
from torch_geometric.data import Data, HeteroData 
from torch_geometric.utils import scatter
from typing import Union 
import torch.nn.functional as F

class BinaryClassifierReadOutLayer(torch.nn.Module):
    def __init__(self, hidden_channels, pooling, num_layers):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_channels, 1)
        self.jk_linear = torch.nn.Linear(hidden_channels*num_layers, 1)
        self.pooling = pooling
        self.reset_parameters()
            
    def forward(self, X, minibatch_info, jumping_knowledge_cat=None):
        # minibatch info tells which parts of X belong to the same graph
        
        # order of first mean and then transformation by W is not important
        if jumping_knowledge_cat is not None: # for GIN network only 
            read_out = scatter(src=jumping_knowledge_cat, index=minibatch_info, dim=0, reduce='sum')
            class_logits = self.jk_linear(read_out)  # no sigmoid here but in loss for numerical stability
        else:
            read_out = scatter(src=X, index=minibatch_info, dim=0, reduce=self.pooling)
            class_logits = self.linear(read_out)  # no sigmoid here but in loss for numerical stability
        
        # w* mean(a + b), mean( w*a + w*b), 
        return class_logits 
    
    def reset_parameters(self):
        self.linear.reset_parameters()
        self.jk_linear.reset_parameters()
        

class BinaryClassifierReadOutLayerDense(torch.nn.Module):
    def __init__(self, hidden_channels, pooling, num_layers, jk=False):
        super().__init__()
        
        lin1in = hidden_channels if not jk else hidden_channels* (num_layers+1)
        lin1out= hidden_channels if not jk else hidden_channels* max((num_layers+1)//2, 1)
       
        self.linear1 = torch.nn.Linear(lin1in, lin1out)
        self.linear2 = torch.nn.Linear(lin1out, hidden_channels)
        self.linear3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1)
        self.pooling = pooling
        self.reset_parameters()
            
    def forward(self, X):
        # minibatch info tells which parts of X belong to the same graph
        if self.pooling == 'sum':
            read_out = torch.sum(X, dim=1)  # minibatch
        elif self.pooling == 'mean':
            read_out = torch.mean(X, dim=1)  # minibatch
        elif self.pooling == 'max':
            read_out, _ = torch.max(X, dim=1)
            
        X = torch.relu(self.linear1(read_out))
        X = torch.relu(self.linear2(X))
        X = torch.relu(self.linear3(X))
        class_logits = self.linear(X)  
        
        # order of first mean and then transformation by W is not important
        # if jumping_knowledge_cat is not None: # for GIN network only 
        #     read_out = scatter(src=jumping_knowledge_cat, index=minibatch_info, dim=0, reduce='sum')
        #     class_logits = self.jk_linear(read_out)  # no sigmoid here but in loss for numerical stability
        # else:
        #     read_out = scatter(src=X, index=minibatch_info, dim=0, reduce=self.pooling)
        #     class_logits = self.linear(read_out)  # no sigmoid here but in loss for numerical stability
        
        # w* mean(a + b), mean( w*a + w*b), 
        return class_logits 
    
    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        self.linear3.reset_parameters()
        self.linear.reset_parameters()
        



        
# pretraining heads
# num_degree_features = 13 # mask out half
# num_triangle_features = 5 # mask out half
# num_categories = 44 # mask out the category
# num_wordvector_features = 100 # mask out 15% or dont at all
class PretrainingHeads(torch.nn.Module):
    def __init__(self, hidden_channels, num_steps, jk=False):
        super().__init__()
        lin1in = hidden_channels if not jk else hidden_channels* (num_steps+1)
        
        num_degree_features = 13 # mask out half
        num_triangle_features = 5 # mask out half
        num_categories = 44 # mask out the category
        num_wordvector_features = 100 # mask out 15% or dont at all
        num_cwes = 150
        # all heads will have 1 layer 
        # degree head
        self.degree_linear1 = torch.nn.Linear(lin1in, hidden_channels)
        self.degree_linear2 = torch.nn.Linear(hidden_channels, num_degree_features)
        # triangle head
        self.triangle_linear1 = torch.nn.Linear(lin1in, hidden_channels)
        self.triangle_linear2 = torch.nn.Linear(hidden_channels, num_triangle_features)
        
        # category head
        self.category_linear1 = torch.nn.Linear(lin1in, num_categories)
        # self.category_linear2 = torch.nn.Linear(hidden_channels, hidden_channels)
        # self.category_linear3 = torch.nn.Linear(hidden_channels, hidden_channels)
        # self.category_linear4 = torch.nn.Linear(hidden_channels, num_categories)
        # wordvector head
        self.wordvector_linear1 = torch.nn.Linear(lin1in, hidden_channels)
        self.wordvector_linear2 = torch.nn.Linear(hidden_channels, num_wordvector_features)
        
        self.linkpred_linear1 = torch.nn.Linear(lin1in, hidden_channels)
        self.linkpred_linear2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.linkpred_linear3 = torch.nn.Linear(hidden_channels, hidden_channels)
        # cwe head
        self.cwe_linear1 = torch.nn.Linear(lin1in, hidden_channels)
        self.cwe_linear2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.cwe_linear3 = torch.nn.Linear(hidden_channels, num_cwes)
        

        # these are counts of cwes in dataset
        self.cwe_weights = torch.load('codegraphs/diversevul/cwe_counts_v2_undirected_withdegreecount_unnormalized.pt').cuda()
        assert self.cwe_weights.shape[0] == num_cwes
        self.cwe_weights =  (torch.sum(self.cwe_weights)/(self.cwe_weights*num_cwes)) # average weight is 1, e.g. n/(n_c,_numclasses) * n_c/n

        
        # sum such that P_all = 1 again 
        
        # for jointly optimized loss:
        self.sigma_degree = torch.nn.Parameter(torch.tensor(1.0))
        self.sigma_triangle = torch.nn.Parameter(torch.tensor(1.0))
        self.sigma_category = torch.nn.Parameter(torch.tensor(1.0))
        self.sigma_wordvector = torch.nn.Parameter(torch.tensor(1.0))
        self.sigma_link_prediction = torch.nn.Parameter(torch.tensor(1.0))
        self.sigma_cwe_classification = torch.nn.Parameter(torch.tensor(1.0))
        
    def forward_cwe_classification_with_loss(self, h, y):
        # sum pooling
        h = torch.sum(h, dim=1)
        h = self.cwe_linear1(h)
        h = torch.relu(h)
        h = self.cwe_linear2(h)
        h = torch.relu(h)
        logits = self.cwe_linear3(h)
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction='none') #/ y.shape[-1]
        # penalyze indices where y is 1 higher, for rows where multiple classes are present, distribute it between them
        # set focus on correctly predicting classes *2
        # set 
        # loss = torch.mean((loss * (1 + (y*2)/torch.sum(y, dim=1).unsqueeze(1))) * self.cwe_weights.unsqueeze(0))
        # for semantic clarity:
        
        # weigh pos and negative classes equally
        #
        # weigh positive samples heigher, but such that weight of whole sample = 1
        y_pos_count = torch.sum(y, dim=1)
        y_neg_count = y.shape[-1] - y_pos_count
        y_pos_weight = 150/(2*y_pos_count)
        y_neg_weight = 150/(2*y_neg_count)
        loss = loss  * y * y_pos_weight.unsqueeze(1) + loss * (1-y) * y_neg_weight.unsqueeze(1)
        loss = torch.mean(loss * self.cwe_weights.unsqueeze(0))
        # loss = torch.mean(torch.sum((loss * self.cwe_weights.unsqueeze(0))/self.cwe_weights.shape[0], dim=1))  # take the /150 per row such that for each sample the probability is 1 again

        
        # get accuracy with max
        y_hat = logits > 0

        TP = torch.sum(((y_hat == 1) & (y == 1))) 
        FP = torch.sum(((y_hat == 1) & (y == 0)))
        TN = torch.sum(((y_hat == 0) & (y == 0)))
        FN = torch.sum(((y_hat == 0) & (y == 1)))
        
        return loss, TP, TN, FP, FN
    
    

    def forward_link_pred_with_loss(self, h, positive_links, negative_links):
        h = self.linkpred_linear1(h)
        h = torch.relu(h)
        h = self.linkpred_linear2(h)
        h = torch.relu(h)
        h = self.linkpred_linear3(h)
        
        pos_pairs, neg_pairs = torch.argwhere(positive_links==1), torch.argwhere(negative_links==1)
        pos_scores = torch.sum(torch.multiply(h[pos_pairs[:,0], pos_pairs[:,1]], h[pos_pairs[:,0], pos_pairs[:,2]]), dim=1)
        neg_scores = torch.sum(torch.multiply(h[neg_pairs[:,0], neg_pairs[:,1]], h[neg_pairs[:,0], neg_pairs[:,2]]), dim=1)
        TP, TN = torch.sum(pos_scores > 0), torch.sum(neg_scores < 0)
        FP, FN = pos_scores.shape[0] - TP, neg_scores.shape[0] - TN
        
        # pos and neg not exactly equal num of pairs, because we draw from bernoulli
        loss = torch.mean(- F.logsigmoid(pos_scores)) - torch.mean(F.logsigmoid(-neg_scores))  
        return loss, TP, TN, FP, FN
    
    def forward_masking_with_loss(self, h, actual_unmodified_features, num_nodes, node_hide_mask ,actual_feature_mask, pretraining_task):
        if pretraining_task == 'degree' :
            layer = self.degree_linear1
            layerlast = self.degree_linear2
            criterion = torch.nn.MSELoss()
        elif pretraining_task == 'triangle':
            layer = self.triangle_linear1
            layerlast = self.triangle_linear2
            criterion = torch.nn.MSELoss()
        elif pretraining_task == 'category':
            layerlast = self.category_linear1
            # layer2 = self.category_linear2
            # layer3 = self.category_linear3 
            # layerlast = self.category_linear4
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        elif pretraining_task == 'wordvector':
            layer = self.wordvector_linear1
            layerlast = self.wordvector_linear2
            criterion = torch.nn.MSELoss()
        
        
        if pretraining_task in ['triangle','degree','wordvector']:
            h = layer(h)
            h = torch.relu(h)
        
        # elif pretraining_task == 'category':
        #     h = layer(h)
        #     h = torch.relu(h)
        #     h = layer2(h)
        #     h = torch.relu(h)
        #     h = layer3(h)
        #     h = torch.relu(h)
        
        # h = layer2(h)
        # h = torch.relu(h)
        h = layerlast(h)
        
    
        y_hat = h * actual_feature_mask #torch.masked_select(h, actual_feature_mask)
        y = actual_unmodified_features * actual_feature_mask
        y_hat = y_hat * node_hide_mask.unsqueeze(1).unsqueeze(0)
        y = y * node_hide_mask.unsqueeze(1).unsqueeze(0)
        correct_classifications, miss_classifications = None, None
        
        loss = 0
        
        for i, n in enumerate(num_nodes):
            hats = y_hat[i, :n, :]
            truths = y[i, :n, :]
            loss += criterion(hats,truths)
            if pretraining_task == 'category':
                with torch.no_grad():
                    correct_classifications = torch.sum(torch.argmax(hats, dim=1) == torch.argmax(truths, dim=1))
                    miss_classifications = n - correct_classifications
        []
        loss = loss / num_nodes.shape[0]
        return loss, correct_classifications, miss_classifications