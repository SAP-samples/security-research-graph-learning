

# How everything is set up:
# Structure learner is only the similarity function which computes A given X 
# GNN is the encoder model
# Task head is the classifier

# One forward pass contains multiple forward_once passes, and breaks when the break_condition is met
# One forward_once pass corresponds to the innermost loop in the training loop

import contextlib
import torch
from torch_geometric.data import Data, HeteroData
from typing import Union
from tqdm.auto import tqdm
import datetime
torch.autograd.set_detect_anomaly(True)
from torch.utils.tensorboard import SummaryWriter
import os
import torch.profiler as profiler
import numpy as np
import yaml
from copy import deepcopy
from torch_geometric.utils import scatter
from models.GCN import squareroot_degree_inverse_undirected_minibatch
from contextlib import nullcontext
import  torch._dynamo as dynamo
from itertools import cycle

torch.autograd.set_detect_anomaly(True)
def quadratic_torch_sparse_slice(indices, values, start, end):
    # slicing for sparse tensor
    mask = (indices[0] >= start) & (indices[0] < end) & (indices[1] >= start) & (indices[1] < end)
    return mask, torch.sparse_coo_tensor(indices[:, mask], values[mask], device=indices.device)

def get_minibatch_intervals(minibatch_info):
        # get the minibatch indices (star1, end1), (start2, end2), ... for each graph in the batch
        changed = (minibatch_info.roll(-1) - minibatch_info).roll(1)
        changed[0] += minibatch_info.max()
        start = torch.cat([torch.tensor([0]).to(changed.device),torch.where(changed)[0], torch.tensor([minibatch_info.shape[-1]]).to(changed.device)])
        end= start.roll(-1)
        indices = torch.stack([start, end], dim=1)[:-1]
        return indices




# >>> GraphGLOW regularization
# def distribute(adj):
    # for all incoming edges for one node, normalize 
    # return torch.nn.functional.normalize(adj, p=0, dim=-1)
    # return adj / (1e-12 + torch.sum(adj, dim=-1)[0].unsqueeze(1))  # there is no max for sparse matrices

# def get_prob(sampled_edge_weights, non_sampled_edge_weights):
#     p1 = torch.log(sampled_edge_weights + 1e-12) 
#     p2 = torch.log(1 - non_sampled_edge_weights + 1e-12) 
#     return torch.sum(p1 + p2)


    

    
def get_prob_minib(p, sample):
    p1 = torch.log(p + 1e-16) * sample
    p2 = torch.log(1 - p + 1e-16) * (1 - sample)
    return torch.sum((p1 + p2),dim=(1,2)) #torch.sum((p1 + p2))

 
# def get_pg_loss(edge_weights, edge_indices, minibatch_info,sparsity_ratio):
#     '''
#     get rewards for policy gradient update
#     '''
#     graph_loss = 0

#     # if out_adj._nnz() > 0:  # we do not use the node represenation similarity constraint as almost 
#     # # none of the original yamls which produce the paper results used it
#     # # and it probably is no good prior for the code graph task
#     #     L = torch.diagflat(torch.sparse.sum(
#     #         out_adj, -1).to_dense()) - out_adj
#     #     if self.config['smoothness_ratio'] > 0:
#     #         graph_loss += self.config['smoothness_ratio'] * torch.trace(
#     #             torch.mm(features.transpose(-1, -2), torch.spmm(
#     #                 L, features))) / int(np.prod(out_adj.shape))

#     if sparsity_ratio > 0:
#         for start, end in get_minibatch_intervals(minibatch_info):
#             adjacency_normalization_factor = (end - start) ** 2
#             mask = (edge_indices[0] >= start) & (edge_indices[0] < end)
#             graph_loss += sparsity_ratio * torch.sum(
#                 torch.pow(edge_weights[mask], 2) / adjacency_normalization_factor)
    
#     return graph_loss

def get_pg_loss(adj,num_nodes,sparsity_ratio):
    '''
    get rewards for policy gradient update
    '''
    graph_loss = torch.zeros(adj.shape[0], device=adj.device)  # minibatch dim
    if sparsity_ratio > 0:
        graph_loss = sparsity_ratio * torch.sum(torch.pow(adj, 2), dim=(1,2)) / num_nodes**2
        
    return graph_loss
# <<< GraphGLOW regularization


class StructureLearning:
    def __init__(self, graph_structure_learner, gnn, task_head, multi_step, pretraining_heads, optimizer, pretraining_optimizers, supervision_criterion, config, from_saved=None,from_saved_epoch=None, cross_val_folder=None, cross_val_valindex=None,multitaskoptim=None):
        
        self.gnn = None
        self.graph_structure_learner = None
        self.task_head = None
        self.multi_step = None
        self.pretraining_heads = None
        self.pretraining_optimizers = None
        self.optimizer = None
        
            
        if config.get('DS_dense_mode',None):
            dynamo.reset() # for torch recompilation max size issue deletes old cached models recompilation
            self.task_head = torch.compile(task_head, fullgraph=True, dynamic=False)
            if config.get('MULTISTEP_enabled', False) :
                print('compile multistep')
                self.pretraining_optimizers = pretraining_optimizers
                self.multi_step = torch.compile(multi_step, fullgraph=True, dynamic=False)
                self.pretraining_heads = pretraining_heads
            else:
                print('compile graphglow')
                self.gnn = torch.compile(gnn, fullgraph=True, dynamic=True)
                self.graph_structure_learner = torch.compile(graph_structure_learner, fullgraph=True, dynamic=False)
                torch.backends.cudnn.benchmark = True
        else:
            self.graph_structure_learner = graph_structure_learner
            self.gnn = gnn
            self.task_head = task_head
            self.multi_step = multi_step
            self.pretraining_optimizers = pretraining_optimizers
            self.pretraining_heads = pretraining_heads
        
        # init bias to class ratio
        with torch.no_grad():
            if hasattr(self.task_head, 'linear'):
                self.task_head.linear.bias.fill_(-torch.log(torch.tensor(1/0.0551 - 1)))
        
        self.multitaskoptim = multitaskoptim
      
        # self.graph_structure_learner = torch.compile(graph_structure_learner, dynamic=True)
        # self.gnn = gnn # torch.compile(gnn, dynamic=True)
        # self.task_head = torch.compile(task_head, dynamic=False)
        
        self.supervision_criterion = supervision_criterion
        
        self.config = config
        self.pos_class_weight = ((self.config['pos_class_count'] + self.config['neg_class_count']))/ (2*self.config['pos_class_count'])
        self.neg_class_weight = ((self.config['pos_class_count'] + self.config['neg_class_count']))/ (2*self.config['neg_class_count'])
        
        # self.params = list(self.graph_structure_learner.parameters()) + list(self.gnn.parameters()) + list(self.task_head.parameters())
        self.optimizer = optimizer
        self.from_saved_epoch = from_saved_epoch + 1 if from_saved_epoch is not None else 0
       
        
        
        # include class names of all models
        name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_'+self.config['writer_tag']
        # include all config parameters in the summary writer name:
        # sort by key
        
        # contains_key_with_hgp = len([key for key in self.config if 'HGP' in key])>0
        
        # for key, value in sorted(self.config.items(), key=lambda x: x[0]):
        #     if key in ['writer_tag','DS_crossval','OPTIM_stopping_patience','loader_batch_size', 'DS_cv_valfraction','DS_cv_trainfraction','device','DS_folds', 'OPTIM']: 
        #         continue
        #     dict_ = {
        #         'epochs': 'ep',
        #         'episodes': 'epi',
        #         "max_iterations": "it",
        #         "GSL_converged_threshold": "Sth",
        #         "OPTIM_lr": "lr",
        #         "OPTIM_batch_size": "bs",
        #         "global_hidden_channels": "ch",
        #         "GNN_dropout": "Gdrp",
        #         "GNN_num_layers": "Gl",
        #         "GSL_num_heads": "Sh",
        #         "GSL_attention_threshold": "GSLat",
        #         "GSL_graph_regularization": "Srg",
        #         'feature_dropout': 'fdrp',
        #         'GSL_skip_ratio': 'Sskp',
        #         'OPTIM_weight_decay': 'wd',
        #         'pos_class_weight': 'pow',
        #         'GSL_sparsity_ratio': 'Ssp',
        #         'DS_root': 'DS',
        #         'H_pool': 'Hp',
        #         'DS_filtertype':'DSf',
        #         'HGP_pooling':'HGPp',
        #         'HGP_sample_neighbor':'HGPsn',
        #         'HGP_sparse_attention':'HGPsa',
        #         'HGP_sparse_attention_threshold':'HGPsat',
        #         'GSL_resethead':'GSL_rs'
                
                
        #         # 'DS_crossval':'cv'
                
        #     } 
            
        #     if key in dict_:
        #         key = dict_[key]
            
        #     if value =='BinaryClassifierReadOutLayer':
        #         value = 'BCROL'
        #     if value=='GraphStructureLearner':
        #         value = 'GSL'
                
        #     if key =='from_saved' and len(value)==0:
        #         continue
        #     if contains_key_with_hgp and 'GSL' in key:
        #         continue 
            
        #     name += f"_{key}{str(value).replace('/', '-')}"
        
        
        # name = name.replace('codegraphs-','').replace('-preprocessed','').replace('withdegreecount','withdeg').replace('_allgraphs','').replace('withoutdeg','wodeg').replace('withoutdegreecount','wodegc')
        # name = name.replace('withdegoldfiles','wdegold').replace('01_fitTRAIN_BbGCN_sl','01_fT_GCNsl')
        # name = name.replace('larger10','l').replace('smaller1000','s').replace("undirected",'ud').replace('withdeg','wdeg').replace('diversevul','divu')
        # name = name.replace("crossval",'cv')
        # name = name.replace('Barebone','Bb')
        if from_saved is not None:
            name = from_saved 
        
        if cross_val_folder is not None:
            assert self.config['DS_crossval']
            assert cross_val_valindex is not None
            
            os.makedirs(f'runs/{cross_val_folder}', exist_ok=True)
            
            os.makedirs(f'runs/{cross_val_folder}/cv_{cross_val_valindex}', exist_ok=True)
            self.writer = SummaryWriter(log_dir=f'runs/{cross_val_folder}/cv_{cross_val_valindex}/{name}')
            os.makedirs(f'runs/{cross_val_folder}/cv_{cross_val_valindex}/{name}/', exist_ok=True)
            self.model_folder = f'runs/{cross_val_folder}/cv_{cross_val_valindex}/{name}/models/'
            os.makedirs(self.model_folder, exist_ok=True)
            
            # save conf file to folder, if it does not exist yet
            if not os.path.exists(f'runs/{cross_val_folder}/conf.yml'):
                yaml.dump(self.config, open(f'runs/{cross_val_folder}/conf.yml', 'w'))

        else:
            os.makedirs(f'runs/{name}', exist_ok=True)   
            self.writer = SummaryWriter(log_dir=f'runs/{name}')
            self.model_folder = f'runs/{name}/models/'
            os.makedirs(self.model_folder, exist_ok=True)
            
            # save conf file to folder, if it does not exist yet
            if not os.path.exists(f'runs/{name}/conf.yml'):
                yaml.dump(self.config, open(f'runs/{name}/conf.yml', 'w'))

        self.best_val_loss_stopping_criterion = float('inf')
        self.best_val_loss_epoch = 0
        self.best_val_metrics, self.current_val_metrics = None, None
        
        

        def noop_record(placeholder):
            return contextlib.nullcontext() 

  
        if self.config.get('profile',None):
            self.profiler = profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA], record_shapes=True)
            self.record_fn = profiler.record_function  
        else:
            self.profiler = contextlib.ExitStack() 
            self.record_fn = noop_record
            
        
    
    
    def has_converged(self, adj, prev_adj):
        with torch.no_grad():
            # adj = adj
            # prev_adj = prev_adj
            diff_ = torch.sum(torch.pow(adj - prev_adj, 2), dim=(1,2))
            norm_ = torch.sum(torch.pow(adj, 2), dim=(1,2))
            diff_ = diff_/ torch.clamp(norm_, min=1e-16) 
            
            has_converged = diff_ <= self.config['GSL_converged_threshold']
            # print(diff_, has_converged)
        return has_converged

    
    def adj_has_converged(self, Z_t,  edge_index, prev_edge_index, edge_weight, prev_edge_weight, minibatch_info, last_has_converged, profiler):
        # has_converged = []
        
        # with self.record_fn("ADJACENCY CONVERGENCE declare"):
        # determine separately: for each graph in the minibatch if the adjacency matrix has converged
        adj = torch.sparse_coo_tensor(edge_index, edge_weight, requires_grad=False, device=edge_index.device)
        prev_adj = torch.sparse_coo_tensor(prev_edge_index, prev_edge_weight, requires_grad=False, device=prev_edge_index.device, size=adj.shape)
        
        diff_ = torch.pow(adj - prev_adj, 2)
        diff_indices = diff_.indices()
        diff_values = diff_.values()
        norm_ = torch.pow(adj, 2)
        norm_indices = norm_.indices()
        norm_values = norm_.values()
        
            # return mask, torch.sparse_coo_tensor(indices[:, mask], values[mask], device=indices.device)

        # with self.record_fn("ADJACENCY CONVERGENCE loop"):
        # final_edge_mask = torch.zeroes_like(edge_index[1], dtype=torch.bool)
        
        idx = torch.argwhere(~last_has_converged).squeeze(1)
        for i, (start, end) in enumerate(get_minibatch_intervals(minibatch_info)[idx]):
            # edge_index[0] is sufficient, since all pairs which are in graph 1 at index1 are also in graph1 at index0
            
            # diff_mask =  #& (indices[1] >= start) & (indices[1] < end)
            # norm_mask =  #& (indices[1] >= start) & (indices[1] < end)
            
            diff_current = torch.sum(diff_values[(diff_indices[0] >= start) & (diff_indices[0] < end)])
            norm_current = torch.sum(norm_values[(norm_indices[0] >= start) & (norm_indices[0] < end)])
            
            # print('curr', diff_current/norm_current)
            # print('curr', diff_current/norm_current)
            converged_edgeweights_for_regularization = []
            converged_edgeindices_for_regularization = []
            if diff_current/norm_current < self.config['GSL_converged_threshold']:
                last_has_converged[idx[i]] = True
                # zero out X, X is num_nodes x num_features
                # Z_t[start:end,:] = 0.0  # inplace operation
                mask = (edge_index[0] >= start) & (edge_index[0] < end)
                
                converged_edgeweights_for_regularization.append(edge_weight[mask])
                converged_edgeindices_for_regularization.append(edge_index[:,mask])
                
                edge_index = edge_index[:,~mask]
                edge_weight = edge_weight[~mask]
                
            # else:
                # has_converged.append(False)
            

        
        # threshold = self.config['converged_threshold']
        # for start, end in self.get_minibatch_intervals(minibatch_info):
        #     mask, adj_graph = quadratic_torch_sparse_slice(edge_index, edge_weight, start, end)
        #     _, prev_adj_graph = quadratic_torch_sparse_slice(prev_edge_index, prev_edge_weight, start, end)
        #     if len(prev_adj_graph) == 0:
        #         prev_adj_graph = torch.sparse_coo_tensor(adj_graph.coalesce().indices(), torch.zeros_like(adj_graph.values()), device=adj_graph.device)
                
        #     diff_ = torch.sum(torch.pow(adj_graph - prev_adj_graph, 2))
        #     norm_ = torch.sum(torch.pow(adj_graph, 2))
            
        #     print('curr2', diff_/norm_)
        #     minibatch_graph_converged = diff_/norm_ < threshold  # torch.norm(adj_graph - prev_adj_graph, p='fro')/ torch.norm(adj_graph, p='fro') < threshold
            
            
    
        #     has_converged.append(minibatch_graph_converged)
            
        #     if minibatch_graph_converged:
        #         # remove the graph from the minibatch
        #         edge_index = edge_index[:,~mask]
        #         edge_weight = edge_weight[~mask]
        #          # zero out X, X is num_nodes x num_features
        #         Z_t[start:end,:] = 0.0
        if len(converged_edgeweights_for_regularization)>0:
            return last_has_converged, edge_index, edge_weight, Z_t, torch.cat(converged_edgeindices_for_regularization, dim=1), torch.cat(converged_edgeweights_for_regularization, dim=0)
        else:
            return last_has_converged, edge_index, edge_weight, Z_t, False, False
    # one forward_once pass through the model-construct
    def forward_once(self, Z_t1, edge_index, initial_edge_index, GSL_skip_ratio, minibatch_info, has_converged, is_first_pass=False):
        
        if is_first_pass:
            # use the initial layer of the encoder to get gamma
            Z_t1 = self.gnn.initial_layer(Z_t1, edge_index)
        
        # with self.record_fn("FORWARD ONCE structure learner"):
        edge_index, edge_weights = self.graph_structure_learner(Z_t1, minibatch_info, has_converged) 
        
        # with self.record_fn("FORWARD ONCE GNN"):
        Z_t, jumping_knowledge_cat = self.gnn(Z_t1, edge_index, edge_weights, initial_edge_index, GSL_skip_ratio)
        Y_hat = self.task_head(Z_t, minibatch_info, jumping_knowledge_cat)
        
        return edge_index, edge_weights, Z_t, Y_hat
    
    
    
    
    def write_metrics(self, split, save_epoch, y, y_hat, loss, epoch, minibatch_index=None, episode=None, 
                      max_minibatch_index=None, max_episode=None, minibatch_size=None, 
                      graphstatistic_nonzero_edgeratio=None,
                      graphstatistic_mean_edgeprob=None,
                        loss_p0_policygradient=None,
                        loss_ptheta_entropy=None,
                        loss_supervision=None
                      ):
        def divide(x, y):
            return x/y if y != 0 else 0
        
        TP = torch.sum((y_hat == 1) * (y == 1)).item()
        FP = torch.sum((y_hat == 1) * (y == 0)).item()
        TN = torch.sum((y_hat == 0) * (y == 0)).item()
        FN = torch.sum((y_hat == 0) * (y == 1)).item()
        
        precision = divide(TP,(TP + FP))
        recall = divide(TP, (TP + FN))
        f1 = divide(2 * precision * recall, (precision + recall) )
        
        balanced_accuracy = (divide(TP, (TP+FN)) + divide(TN, (TN+FP)))/2
        
        
        # number of graphs seen
        if not save_epoch:
            already_seen = epoch * max_minibatch_index * max_episode * minibatch_size 
            current_epoch_seen = (minibatch_index+1) * (episode+1) * (minibatch_size)
            idx = already_seen + current_epoch_seen
        else:
            idx = epoch
        
        epoch_str = f'epoch_/' if save_epoch else ''
        
        self.writer.add_scalar(f'{split}/{epoch_str}loss', loss,idx)
        self.writer.add_scalar(f'{split}/{epoch_str}precision', precision, idx)
        self.writer.add_scalar(f'{split}/{epoch_str}recall', recall, idx)
        self.writer.add_scalar(f'{split}/{epoch_str}f1', f1, idx)
        self.writer.add_scalar(f'{split}/{epoch_str}balanced_accuracy', balanced_accuracy, idx)
        
        # add TP, FP, TN, FN
        self.writer.add_scalar(f'{split}/{epoch_str}TP', TP, idx)
        self.writer.add_scalar(f'{split}/{epoch_str}FP', FP, idx)
        self.writer.add_scalar(f'{split}/{epoch_str}TN', TN, idx)
        self.writer.add_scalar(f'{split}/{epoch_str}FN', FN, idx)
        
        if not graphstatistic_nonzero_edgeratio is None:
            self.writer.add_scalar(f'{split}/{epoch_str}nonzero_edgeratio', graphstatistic_nonzero_edgeratio, idx)
        if not graphstatistic_mean_edgeprob is None:
            self.writer.add_scalar(f'{split}/{epoch_str}mean_edgeprob', graphstatistic_mean_edgeprob, idx)
        if not loss_p0_policygradient is None:
            self.writer.add_scalar(f'{split}/{epoch_str}loss_p0_policygradient', loss_p0_policygradient, idx)
        if not loss_ptheta_entropy is None:
            self.writer.add_scalar(f'{split}/{epoch_str}loss_ptheta_entropy', loss_ptheta_entropy, idx)
        if not loss_supervision is None:
            self.writer.add_scalar(f'{split}/{epoch_str}loss_supervision', loss_supervision, idx)
            
            
        print(f'{idx} {split}, loss {loss:.6f} precision {precision:.4f} recall {recall:.4f} f1 {f1:.4f} balanced_accuracy {balanced_accuracy:.4f}')
        self.writer.flush()
        return idx, precision, recall, f1, balanced_accuracy, TP, FP, TN, FN
    
    
    def write_pretrain_metrics(self,is_epoch, epoch, task, current_step, loss, correct_classifications=None, miss_classifications=None, TP=None, TN=None, FP=None, FN=None,is_train_val_test=False, pretraining_block=None, std_param=None):
        idx = epoch if is_epoch else current_step
        is_train = 'train' if is_train_val_test == 'train' else 'val'
        
        # epoch_str = f'pretrain/epoch_/' if is_epoch else ''
        epoch_str = f'pretrain/'+is_train+'_'+pretraining_block+'/'+('epoch_/' if is_epoch else '')
        if task in ['degree', 'triangle', 'wordvector']:
            loss_type = 'MSE'
        elif task in ['category']:
            loss_type = 'CE'
        elif task in ['link_prediction', 'cwe_classification']:
            loss_type = 'BCE'
        else:
            loss_type = 'BCECEMSE'
            
        if task in ['degree', 'triangle', 'wordvector','category']:
            epoch_str += 'feature_masking'
            
        # write loss
        self.writer.add_scalar(f'{epoch_str}{task}/loss_{loss_type}', loss, idx)
        if std_param is not None:
            self.writer.add_scalar(f'{epoch_str}{task}/std_param', std_param, idx)
            
        if task in ['category']:
            # write accuracy
            accuracy = correct_classifications/(correct_classifications + miss_classifications)
            self.writer.add_scalar(f'{epoch_str}{task}/accuracy', accuracy, idx)
            self.writer.add_scalar(f'{epoch_str}{task}/correct_classifications', correct_classifications, idx)
            self.writer.add_scalar(f'{epoch_str}{task}/miss_classifications', miss_classifications, idx)
        elif task in ['link_prediction', 'cwe_classification']:
            # write TP, TN, FP, FN
            self.writer.add_scalar(f'{epoch_str}{task}/TP', TP, idx)
            self.writer.add_scalar(f'{epoch_str}{task}/TN', TN, idx)
            self.writer.add_scalar(f'{epoch_str}{task}/FP', FP, idx)
            self.writer.add_scalar(f'{epoch_str}{task}/FN', FN, idx)
            # write accuracy
            accuracy = (TP + TN)/(TP + TN + FP + FN)
            self.writer.add_scalar(f'{epoch_str}{task}/accuracy', accuracy, idx)
            # write precision, recall, f1
            precision = TP/(TP + FP)
            recall = TP/(TP + FN)
            f1 = 2 * precision * recall / (precision + recall)
            
            # check for nans and replace with 0
            if torch.isnan(precision):
                precision = 0
            if torch.isnan(recall):
                recall = 0
            if torch.isnan(f1):
                f1 = 0
            
            self.writer.add_scalar(f'{epoch_str}{task}/precision', precision, idx)
            self.writer.add_scalar(f'{epoch_str}{task}/recall', recall, idx)
            self.writer.add_scalar(f'{epoch_str}{task}/f1', f1, idx)
            
        self.writer.flush()
        
    # def pretrain(self, )
    def pretraining(self, train_dataloaders, val_dataloaders, is_train_val_test):
        
        
        # for pretraining_task, dataloader in dataloaders.items():
        # dataloader = train_dataloaders['dense']
        
        # pretraining block 1
        
        block1_tasks = self.config['PRETRAINING_block1_tasks']
        # len_dataloader = len(dataloader)
        # num_epochs = self.config['PRETRAINING_epochs']
        alternating_interval = self.config['PRETRAINING_alternating_interval']
        is_alternating = self.config['PRETRAINING_alternating']
        is_jointoptimization = self.config['PRETRAINING_multitaskoptim']
        
        assert not (is_jointoptimization and is_alternating)
        self.pretraining_block(self.config['PRETRAINING_epochs_block1'], self.config['PRETRAINING_block1_num_steps'],train_dataloaders, val_dataloaders, block1_tasks, is_alternating, alternating_interval, is_train_val_test, 'block1', actual_epoch=None, is_jointoptimization=is_jointoptimization)

        block2_tasks = self.config['PRETRAINING_block2_tasks']
        if len(block2_tasks) > 0:
            self.pretraining_block(self.config['PRETRAINING_epochs_block2'], self.config['PRETRAINING_block2_num_steps'],train_dataloaders, val_dataloaders, block2_tasks, False, -1, is_train_val_test, 'block2', actual_epoch=None, is_jointoptimization=is_jointoptimization)
        
        self.optimizer.zero_grad()
        
        
    def pretraining_block(self, epochs, num_steps, train_dataloaders, val_dataloaders, tasks, is_alternating, alternating_interval, is_train_val_test, blocklabel, actual_epoch=None,is_jointoptimization=None):
        print('pretraining block', blocklabel)
        if is_train_val_test == 'train':
            self.pretraining_heads.train()
            self.multi_step.train()
            dataloader = train_dataloaders['dense']
            cwe_classification_dataloader = train_dataloaders['cwe']
        elif is_train_val_test == 'val':
            self.pretraining_heads.eval()
            self.multi_step.eval()
            dataloader = val_dataloaders['dense']
            cwe_classification_dataloader = val_dataloaders['cwe']
        else:
            raise ValueError('is_train_val_test must be train or val')
        
        dataloaderiter = iter(dataloader)
        cwe_classification_dataloaderiter = iter(cwe_classification_dataloader)
        
        if is_alternating:
            len_epoch = len(dataloader)
        elif is_jointoptimization:
            len_epoch = len(dataloader)
            if 'feature_masking' in tasks:
                tasks.remove('feature_masking')
                tasks.append('feature_masking_0')
                tasks.append('feature_masking_1')
                tasks.append('feature_masking_2')
                tasks.append('feature_masking_3')
            def cyclic_next(iterable):
                cyclic_iter = cycle(iterable)
                counter = 0
                
                while True:
                    yield_value = next(cyclic_iter)
                    counter += 1
                    yield yield_value, counter % len(iterable) == 0
            task_cycler = cyclic_next(tasks)
            full_cycle_loss = 0
            
        else: # blocked pretraining
            total_epochs = epochs
            num_tasks = len(tasks)
            if is_train_val_test == 'train':
                len_epoch = len(dataloader) if num_steps < 0 else num_steps
            else:  # val
                len_epoch = len(dataloader)
                
            total_steps = total_epochs * len_epoch 
            steps_per_task = total_steps // num_tasks
            steps = []
            counter = -1
            for i in range(total_steps):
                if (i+1) % steps_per_task == 0:
                    counter += 1
                steps.append(tasks[counter])
            
        
        print('all tasks', tasks)
        for epoch in range(epochs):
            epoch_loss, epoch_correct_classifications, epoch_miss_classifications, epoch_TP, epoch_FP, epoch_TN, epoch_FN = 0, 0, 0, 0, 0, 0, 0
            for minibatch_index in tqdm(range(len_epoch)):
                correct_classifications, miss_classifications, TP, FP, TN, FN = None, None, None, None, None, None
                if is_alternating:
                    if minibatch_index % alternating_interval == 0:
                        # pick random task 
                        pretraining_task = tasks[np.random.randint(len(tasks))]
                        if  len(tasks) > 1:
                            print('task switch', pretraining_task)
                            
                elif is_jointoptimization:
                    # iterate through pretraining tasks
                    pretraining_task, is_full_cycle_for_backprop = next(task_cycler)
                else:
                    pretraining_task = steps[epoch * len_epoch + minibatch_index]
                
                if pretraining_task == 'cwe_classification':
                    try:
                        data = next(cwe_classification_dataloaderiter)
                    except StopIteration:
                        cwe_classification_dataloaderiter = iter(cwe_classification_dataloader)
                        data = next(cwe_classification_dataloaderiter)
                    
                else:
                    try:
                        data = next(dataloaderiter)
                    except StopIteration:
                        dataloaderiter = iter(dataloader)
                        data = next(dataloaderiter)
                
                data = data.to(self.config['device'], non_blocking=True)
                adj_initial, features, y, num_nodes = data.adj,data.x, data.y, data.numnodes
                if self.config.get('loader_add_naive_predictor',False):
                    # add naive predictor: 1 if num_nodes > 102 else 0 as well as add num_nodes
                    # features shape is batch x num_nodes x num_features
                    # feat = torch.cat(((num_nodes > 102).to(torch.float32).unsqueeze(-1),num_nodes.unsqueeze(-1) ), dim=-1).to(features.device)
                    # features= torch.cat((features, feat.unsqueeze(1).repeat(1,features.shape[1],1)), dim=-1)
                    # add zero tensor to features
                    features = torch.cat((features, torch.zeros((features.shape[0],features.shape[1],2), device=features.device)), dim=-1)
                    
                minibatch_mask = torch.zeros((features.shape[0],features.shape[1], 1), device=num_nodes.device, dtype=torch.bool)
                for i, n in enumerate(num_nodes):
                    minibatch_mask[i,:n,:] = True 
                    
                        
                if 'feature_masking' in pretraining_task:
                    unmodified_features = features.clone()
                    # mask out either (or all) parts of triangle features, parts of degree features, the category or the wordvector features
                    # mask out by setting the feature to -1
                    num_features = features.shape[-1]
                    num_degree_features = 13 # mask out half
                    num_triangle_features = 5 # mask out half
                    num_categories = 44 # mask out the category
                    num_wordvector_features = 100 # mask out 15% or dont at all
                    
                    # pick one of the four masks
                    if pretraining_task == 'feature_masking':

                        mask_choice = torch.randint(0,3,(1,))
                    else:
                        mask_choice = int(pretraining_task.split('_')[-1])
                        
                    feature_hide_mask = torch.zeros(num_features)
                    
                    if mask_choice == 0:  # degree mask
                        actual_mask = torch.bernoulli(0.5*torch.ones(num_degree_features)).bool()
                        actual_unmodified_features = unmodified_features[:,:,:num_degree_features]
                        feature_hide_mask[:num_degree_features] = actual_mask
                        mask_value = -1
                        task = 'degree'
                    elif mask_choice == 1:  # triangle mask
                        actual_mask = torch.bernoulli(0.5*torch.ones(num_triangle_features)).bool()
                        actual_unmodified_features = unmodified_features[:,:,num_degree_features:num_degree_features+num_triangle_features]
                        feature_hide_mask[num_degree_features:num_degree_features+num_triangle_features] = actual_mask
                        mask_value = -1
                        task = 'triangle'
                    elif mask_choice == 2:  # category mask
                        actual_mask = torch.ones(num_categories).bool()
                        actual_unmodified_features = unmodified_features[:,:,num_degree_features+num_triangle_features:num_degree_features+num_triangle_features+num_categories]
                        feature_hide_mask[num_degree_features+num_triangle_features:num_degree_features+num_triangle_features+num_categories] = actual_mask
                        mask_value = 0
                        task = 'category'
                    elif mask_choice == 3:  # wordvector mask
                        actual_mask = torch.bernoulli(0.15*torch.ones(num_wordvector_features)).bool()
                        actual_unmodified_features = unmodified_features[:,:,num_degree_features+num_triangle_features+num_categories:]
                        feature_hide_mask[num_degree_features+num_triangle_features+num_categories:] = actual_mask
                        mask_value = 0
                        task = 'wordvector'
                     
                    feature_hide_mask, actual_mask = feature_hide_mask.to(features.device), actual_mask.to(features.device)
                    if mask_choice == 0 or mask_choice == 1:  # degree or triangle mask
                        node_mask_probability = self.config['PRETRAINING_feature_masking_node_probability_degree_triangle']
                    elif mask_choice == 2:  # category mask
                        node_mask_probability = self.config['PRETRAINING_feature_masking_node_probability_category']
                    elif mask_choice == 3:  # wordvector mask
                        node_mask_probability = self.config['PRETRAINING_feature_masking_node_probability_wordvector']
                  
                     
               
                    
                    # adj_initial is batchdim x num_nodes x num_nodes
                    # features is batchdim x num_nodes x num_features
                    
                    # mask out a random node
                    node_hide_mask = torch.bernoulli(node_mask_probability*torch.ones((adj_initial.shape[1]))).bool().to(features.device)
                    
                    
                    # mask the features for selected nodes
                    for i, n in enumerate(num_nodes):
                        features[i,:n,:] = features[i,:n,:] + (node_hide_mask[:n].unsqueeze(-1) * feature_hide_mask.unsqueeze(0) * (-features[i,:n,:] + mask_value) ) 
                    
                    
                    h = self.multi_step(features, adj_initial, minibatch_mask)
                    
                    loss, correct_classifications, miss_classifications = self.pretraining_heads.forward_masking_with_loss(h, actual_unmodified_features, num_nodes, node_hide_mask, actual_mask, task)
                    # mask out a random feature
                    # feature_mask = torch.randint(0, num_features, (1,))
                    # features[:,feature_mask] = 0.0
                    
                elif pretraining_task == 'link_prediction':
                    
                    # sample as many negatives as positives
                    positive_links = torch.bernoulli(self.config['PRETRAINING_link_prediction_mask_probability']*adj_initial)
                    # per row
                    positive_link_amounts = torch.sum(positive_links, dim=2)
                    
                    # sample random negative links for the source nodes
                    # root_nodes = positive_link_amounts > 0
                    
                    negative_sample_probability = positive_link_amounts / (-positive_link_amounts + num_nodes.unsqueeze(1))
                    # quadruple the amount of negative samples
                    negative_sample_probability = negative_sample_probability * 4
                    # clamp to max
                    negative_sample_probability = torch.clamp(negative_sample_probability, max=1)
                    # leave self edges as they amount to 0 loss anyways
                    # print(adj_intial.shape, positive_links.shape, minibatch_mask.shape, negative_sample_probability.shape)
                   
                    possible_negative_links = ((torch.ones_like(adj_initial)-positive_links)*minibatch_mask*minibatch_mask.transpose(-2,-1))
                    negative_links = torch.bernoulli( possible_negative_links * negative_sample_probability.unsqueeze(-1))
                    
                    
                    # num_possible_negative_links = num_nodes * (num_nodes - 1) // 2 - positive_link_amounts
                    # sample_probability = positive_link_amounts / num_possible_negative_links
                    
                    # negative_links = torch.bernoulli((torch.ones_like(adj_initial) - adj_initial) * sample_probability.unsqueeze(1).unsqueeze(2))
                    
                    # make adj_initial 0
                    adj_without_positives = adj_initial * (1-positive_links)
                    h = self.multi_step(features, adj_without_positives, minibatch_mask)

                    loss, TP, TN, FP, FN = self.pretraining_heads.forward_link_pred_with_loss(h, positive_links, negative_links)
                    # correct_classifications, miss_classifications = None, None
                # elif pretraining_task == 'link_classification':
                
                elif pretraining_task == 'cwe_classification':
                    h = self.multi_step(features, adj_initial, minibatch_mask)
                    loss, TP, TN, FP, FN = self.pretraining_heads.forward_cwe_classification_with_loss(h, y)
                else:
                    raise ValueError(f'Pretraining task {pretraining_task} not recognized')
                # elif pretraining_task == 'context_prediction':
                
                which_optim = pretraining_task if not 'feature_masking' in pretraining_task else task
                
                if is_train_val_test == 'train':
                    std_param = None
                    if is_jointoptimization:
                        # https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
                        which_lossterm = {
                                'degree': 'MSE',
                                'triangle': 'MSE',
                                'category': 'CE',
                                'wordvector': 'MSE',
                                'link_prediction': 'BCE',
                                'cwe_classification': 'BCE'
                        }
                        if not self.config.get('PRETRAINING_multitask_addloss',False):
                            std_param = getattr(self.pretraining_heads, f'sigma_{which_optim}')
                            losstermtype = which_lossterm[which_optim]
                            if losstermtype in ['CE','BCE']:
                                loss = 1/(torch.pow(std_param,2))*loss + torch.log(1+torch.pow(std_param,2))
                                full_cycle_loss += loss
                            elif losstermtype == 'MSE':
                                # we assume isometric gaussian, with SIGMA = std_allsame**2 * I, as loss is already loss/n -> torch.log(std_param**n)/n = torch.log(std_param)
                                loss = 1/(2*torch.pow(std_param,2))*loss + torch.log(1+torch.pow(std_param,2))
                                full_cycle_loss += loss
                        else:
                            full_cycle_loss += loss
                            
                        if is_full_cycle_for_backprop:
                            self.multitaskoptim.zero_grad()
                            full_cycle_loss.backward()
                            self.multitaskoptim.step()
                            
                            self.write_pretrain_metrics(
                                is_epoch=False, epoch=epoch,
                                task='full_cycle', 
                                current_step=epoch * len_epoch + minibatch_index, 
                                loss=full_cycle_loss, 
                                correct_classifications=correct_classifications, 
                                miss_classifications=miss_classifications,
                                TP=TP, TN=TN, FP=FP, FN=FN,
                                is_train_val_test=is_train_val_test,
                                pretraining_block=blocklabel
                            )
                            full_cycle_loss = 0
                            
                            
                    else:
                        self.optimizer.zero_grad()
                        self.pretraining_optimizers[which_optim].zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        self.pretraining_optimizers[which_optim].step()
                        
                    # 
                    # if minibatch_index % 100 == 0:
                    
                        
                    self.write_pretrain_metrics(
                        is_epoch=False, epoch=epoch,
                        task=which_optim,
                        current_step=epoch * len_epoch + minibatch_index,
                        loss=loss,
                        correct_classifications=correct_classifications,
                        miss_classifications=miss_classifications,
                        TP=TP, TN=TN, FP=FP, FN=FN,
                        is_train_val_test=is_train_val_test,
                        pretraining_block=blocklabel,
                        std_param = std_param.item() if std_param is not None else None
                    )
                   
                        
                    if TP is not None:
                        epoch_TP += TP
                        epoch_TN += TN
                        epoch_FP += FP
                        epoch_FN += FN
                    elif correct_classifications is not None:
                        epoch_correct_classifications += correct_classifications
                        epoch_miss_classifications += miss_classifications
                        
                epoch_loss += loss.item()

                    
            self.write_pretrain_metrics(
                is_epoch=True, epoch=epoch if actual_epoch is None else actual_epoch,
                task='fullepoch',
                current_step=epoch * len_epoch + minibatch_index,
                loss=epoch_loss/len_epoch,
                correct_classifications=epoch_correct_classifications,
                miss_classifications=epoch_miss_classifications,
                TP=epoch_TP, TN=epoch_TN, FP=epoch_FP, FN=epoch_FN,
                is_train_val_test=is_train_val_test,
                pretraining_block=blocklabel
            )
            
            if is_train_val_test == 'train':
                with torch.no_grad():
                    self.multi_step.eval()
                    self.pretraining_heads.eval()
                    self.pretraining_block(
                        1, num_steps, train_dataloaders, val_dataloaders, tasks, is_alternating, alternating_interval, 'val', blocklabel, actual_epoch=epoch, is_jointoptimization=is_jointoptimization
                    )
                    self.multi_step.train()
                    self.pretraining_heads.train()
            
            # try:
            #     minibatch = next(loaders[target_edge_type])
            # except StopIteration: # "reinit" iterator
            #     loaders[target_edge_type] = get_hgt_linkloader(data, target_edge_type, batch_size, is_training, sampling_mode, neg_sampling_ratio, num_neighbors, num_workers, prefetch_factor, pin_memory)#iter(loaders[target_edge_type])
            #     minibatch = next(loaders[target_edge_type])
            #     pass
            
    
    def train(self, dataloader, epochs, episodes, is_train_val_test='train', val_dataloader=None, writer_epoch=None, start_epoch=0, pretraining_traindataloaders=None,pretraining_valdataloaders=None):
        
        
        assert not (self.config.get('GraphGLOW_only_enabled',False) and self.config.get('MULTISTEP_enabled',False))
        
        print("start epoch", start_epoch)
        if start_epoch == 0 and self.config.get('from_saved',False) and is_train_val_test == 'train':
            print("=========== load saved model ===============")
            self.load_full_model(
                self.config['from_saved'],
            )
        
        if is_train_val_test == 'train':
            if hasattr(self, 'multi_step') and self.multi_step is not None:
                self.multi_step.train()
            if hasattr(self, 'task_head') and self.task_head is not None: 
                self.task_head.train()
            if hasattr(self, 'gnn') and self.gnn is not None:
                self.gnn.train()
            if hasattr(self, 'graph_structure_learner') and self.graph_structure_learner is not None:
                self.graph_structure_learner.train()
          
        else:
            if hasattr(self, 'task_head') and self.task_head is not None:
                self.task_head.eval()
            if hasattr(self, 'gnn') and self.gnn is not None:
                self.gnn.eval()
            if hasattr(self, 'graph_structure_learner') and self.graph_structure_learner is not None:
                self.graph_structure_learner.eval()
            if hasattr(self, 'multi_step') and self.multi_step is not None:
                self.multi_step.eval()
            
        
        if self.config.get('GSL_mixed_precision',None):
            scaler = torch.cuda.amp.GradScaler()

        if self.config.get('MULTISTEP_enabled',False):
            if is_train_val_test == 'train' and len(self.config.get('PRETRAINING_loadpretraining_path',''))> 0:
                print('======= LOADING MULTISTEP PRETRAINED MODEL ========')
                self.load_full_model(
                    self.config['PRETRAINING_loadpretraining_path'],
                )
            
            if is_train_val_test == 'train' and self.config.get('PRETRAINING_enabled',False):
                # len(config['from_saved'])<0:
                print('=== START/CONTINUE PRETRAINING ===')
                self.pretraining(pretraining_traindataloaders, pretraining_valdataloaders, 'train')
                print('=== PRETRAINING END ===')
                
                
                if hasattr(self, 'multi_step') and self.multi_step is not None:
                    print('train model')
                    self.multi_step.train()
                if hasattr(self, 'task_head') and self.task_head is not None: 
                    print('train task head')
                    self.task_head.train()
                if hasattr(self, 'gnn') and self.gnn is not None:
                    self.gnn.train()
                if hasattr(self, 'graph_structure_learner') and self.graph_structure_learner is not None:
                    self.graph_structure_learner.train()
                

                self.save_full_model(
                    epoch=self.config['PRETRAINING_epochs_block1']+self.config['PRETRAINING_epochs_block2'],
                    multi_step=self.multi_step,
                    task_head=self.task_head,
                    optim=self.optimizer,
                    pretraining_heads=self.pretraining_heads,
                    pretraining_optimizers=self.pretraining_optimizers,
                    label='pretraining_ended'
                )
            
            if is_train_val_test == 'train' and self.config.get('MULTISTEP_freeze_gsl_and_gsl_gnn',False):
                print('=== FREEZE GSL AND GSL GNNs ===')
                self.multi_step.freeze_gsl_and_gsl_gnn()
                
        for epoch in range(start_epoch,start_epoch+epochs):
            epoch_y, epoch_y_hat, epoch_supervision_loss = [], [], 0
            epoch_filenames, epoch_logits = [], []
            
            # if self.config['writer_tag'] in ['5IterNoRBS32','5IterNoRMean']:
                # batching_loss = 0
            
            # Reference Models
            
            if self.config['GNN'] in ['BareboneGCN', 'BareboneGIN', 'Reveal','HGPSL','BareboneRGCN', 'BareboneDRGCN']:
                for minibatch_index, data in enumerate(tqdm(dataloader)):
                    data = data.to(self.config['device'], non_blocking=True)
                    if self.config['GNN'] in ['BareboneRGCN', 'BareboneDRGCN']:
                        Z_t, e1,e2,e3,e4,e5, minibatch_info, y = data['node'].x,data['node','TOTAL','node'].edge_index,data['node','AST','node'].edge_index,data['node','CFG','node'].edge_index,data['node','CG','node'].edge_index,data['node','DFG','node'].edge_index, data['node'].batch, data['node'].y.unsqueeze(1)
                        logits = self.gnn(Z_t, e1,e2,e3,e4,e5, minibatch_info)
                        
                    else:
                        Z_t, edge_index, minibatch_info, y = data.x, data.edge_index, data.batch, data.y.unsqueeze(1)
                        Z_t = torch.nn.functional.dropout(Z_t, p=self.config['feature_dropout'], training=is_train_val_test=='train')
                        logits = self.gnn(Z_t, edge_index, minibatch_info)
                    
                    Y_hat = logits > 0
                    weights = self.pos_class_weight * y + self.neg_class_weight * (1-y)
                    loss = self.supervision_criterion(logits, y, weight=weights)
                    
                    if is_train_val_test == 'train':
                        
                        loss.backward()
                        if self.config.get('loader_batch_size',None) not in [None, '']:
                            if ((minibatch_index+1)%(self.config['batch_size'] // self.config['loader_batch_size']) == 0 or minibatch_index == len(dataloader)-1):
                                self.optimizer.step()
                                self.optimizer.zero_grad(set_to_none=True)
                        else:
                            self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)
                            
                 
                    if self.config.get('loader_batch_size',None) not in [None, '']:
                        epoch_supervision_loss += loss.item()#(loss.item()/self.config['loader_batch_size'])
                    else:
                        epoch_supervision_loss += loss.item()#/self.config['batch_size']) # only if reduction!=mean
                        
                    epoch_y.append(y.detach())
                    epoch_y_hat.append(Y_hat.detach())
                    epoch_logits.append(logits.detach())
                    epoch_filenames.append(data.file_name)
                    
                epoch_supervision_loss = epoch_supervision_loss
            
            # GraphGLOW

            elif self.config['GNN'] in ['GCN','GIN'] and not self.config.get('MULTISTEP_enabled',False): # GraphGLOW: # GraphGLOW

                self.supervision_criterion= torch.nn.functional.binary_cross_entropy_with_logits
                
                for minibatch_index, data in enumerate(tqdm(dataloader)):
                    with self.profiler as prof:
                   
                                
                        
                        data = data.cuda(non_blocking=True)
                        if self.config['GNN'] != 'GIN':
                            adj_initial = squareroot_degree_inverse_undirected_minibatch(data.adj)
                        else:
                            adj_initial = data.adj
                        features, y, num_nodes =  data.x, data.y, data.numnodes
                        minibatch_mask = torch.zeros((features.shape[0],features.shape[1], 1), device=num_nodes.device, dtype=torch.bool)
                        for i, n in enumerate(num_nodes):
                            minibatch_mask[i,:n,:] = True 
                            
                        zero_tensor = torch.zeros_like(y)
                        for episode in range(episodes):
                            
                            graphstatistic_nonzero_edgeratio, graphstatistic_mean_edgeprob = zero_tensor.clone(), zero_tensor.clone()
                        
                            with torch.autocast(device_type='cuda') if self.config.get('GSL_mixed_precision',None) else nullcontext():
                                
                                final_logits = zero_tensor.clone()
                                final_t = zero_tensor.clone()
                                supervision_loss = zero_tensor.clone()
                                has_converged = zero_tensor.clone().bool()
                                loss_ptheta_entropies, loss_p0_policygradients = zero_tensor.clone(), zero_tensor.clone()
                                
                                h = self.gnn.forward_initial_layer(features, adj_initial, minibatch_mask)
                                
                                for t in range(self.config['max_iterations']):
                                    adj = self.graph_structure_learner(h,has_converged)
                                    
                                    
                                    if self.config['GNN'] != 'GIN':
                                        adj = squareroot_degree_inverse_undirected_minibatch(adj)
                                        
                                        
                                    h = self.gnn(features, adj, adj_initial, GSL_skip_ratio=self.config['GSL_skip_ratio'], minibatch_mask=minibatch_mask) # in graphglow they use same h as the one from the initial layer
                                    
                                    
                                    logits = self.task_head(h)
                                    if logits.dim()!=1:
                                        logits = logits.squeeze(-1)
                                    
                                    
                                    mask = ~has_converged
                                    weights = self.pos_class_weight * y + self.neg_class_weight * (1-y)
                                    supervision_loss = supervision_loss + mask*self.supervision_criterion(logits, y, weight=weights, reduction='none')
                                    
                                    
                                    
                                    
                                    if is_train_val_test == 'train':
                                        if self.config['GSL_graph_regularization'] and self.config['GSL_sparsity_ratio'] > 0:
                                            # sample
                                            sample_times = 5
                                            loss_list = []
                                            prob_list = []
                                    
                                            mask = (~has_converged)
                                            row_max = torch.max(adj+1e-16, dim=-1)[0].unsqueeze(2)
                                            distribution = (adj * torch.pow(row_max,-1) )#[mask,...]
                                            
                                            
                                            for sample_step in range(sample_times):
                                                with torch.no_grad():
                                                    with self.record_fn("BERNOULLI"):
                                                        adj_sample = torch.bernoulli(distribution).to(
                                                            adj.device)
                                                
                                                
                                                prob_list.append(get_prob_minib(distribution, adj_sample))
                                                with torch.no_grad():
                                                    with self.record_fn("LOSSLIST"):
                                                        loss_list.append(mask*get_pg_loss(adj_sample, num_nodes, self.config['GSL_sparsity_ratio']))
                                                    

                                            with self.record_fn("REGULARIZ"):
                                                loss_list, prob_list = torch.stack(loss_list).cuda(non_blocking=True), torch.stack(prob_list).cuda(non_blocking=True)
                                                loss_mean = torch.mean(loss_list,dim=0)
                                                loss_p0_policygradients = loss_p0_policygradients + torch.sum((loss_list - loss_mean) * prob_list, dim=0) / sample_times
                                                
                                                loss_ptheta_entropies = loss_ptheta_entropies + mask*get_pg_loss(adj, num_nodes, self.config['GSL_sparsity_ratio'])
                                            
                                    with self.record_fn("checker"):
                                        
                                        if t>0:
                                            has_converged =  has_converged | self.has_converged(adj, prev_adj)
                                            
                                        prev_adj = adj 
                                            
                                        for i, logit in enumerate(logits):
                                            if final_logits[i] == 0 and (has_converged[i] or t == self.config['max_iterations']-1):
                                                final_logits[i], final_t[i] = logit.detach(), t
                                                with torch.no_grad(): # only write once
                                                    graphstatistic_nonzero_edgeratio[i] = (torch.sum(adj[i]>0)/(num_nodes[i])**2).detach()
                                                    graphstatistic_mean_edgeprob[i] = torch.mean(adj[i][:num_nodes[i],:num_nodes[i]]).detach()
                                        
                                        if torch.sum(has_converged) == y.shape[0]:
                                            break
                                        
                                regularization_losses = loss_p0_policygradients + loss_ptheta_entropies
                                loss = torch.mean((supervision_loss + regularization_losses) / (final_t+1))
                                
                                
                            Y_hat = final_logits > 0   

                            if is_train_val_test == 'train':
                                with self.record_fn("BACKWARD"):
                                
                                    if self.config.get('GSL_mixed_precision',None):
                                        scaler.scale(loss).backward()
                                    else:
                                        loss.backward()
                                        
                                    if self.config.get('loader_batch_size',None) not in [None, '']:
                                                    
                                        if ((minibatch_index+1)%(self.config['batch_size'] // self.config['loader_batch_size']) == 0 or minibatch_index == len(dataloader)-1):
                                            if self.config.get('GSL_mixed_precision',None):
                                                scaler.step(self.optimizer)
                                                scaler.update()
                                                self.optimizer.zero_grad(set_to_none=True)
                                            else:
                                                self.optimizer.step()
                                                self.optimizer.zero_grad(set_to_none=True)
                                    else:
                                        if self.config.get('GSL_mixed_precision',None):
                                                scaler.step(self.optimizer)
                                                scaler.update()
                                                self.optimizer.zero_grad(set_to_none=True)
                                        else:
                                            self.optimizer.step()
                                            self.optimizer.zero_grad(set_to_none=True)
                                
                                if minibatch_index % 100 == 0:
                                    Y_hat = Y_hat.detach()
                                    self.write_metrics(
                                        split=is_train_val_test,
                                        save_epoch=False,
                                        y=y.detach(), y_hat=Y_hat, loss=loss.item(), 
                                        epoch=epoch,
                                        minibatch_index= minibatch_index, 
                                        episode=episode, 
                                        max_minibatch_index=len(dataloader),
                                        max_episode=episodes,
                                        minibatch_size=self.config['batch_size'],
                                        graphstatistic_nonzero_edgeratio=torch.mean(graphstatistic_nonzero_edgeratio),
                                        graphstatistic_mean_edgeprob=torch.mean(graphstatistic_mean_edgeprob),  # iid assumption
                                        loss_p0_policygradient=(torch.mean(loss_p0_policygradients) /torch.mean(final_t+1)).item() if not loss_p0_policygradients is None else None,
                                        loss_ptheta_entropy=(torch.mean(loss_ptheta_entropies)/torch.mean(final_t+1)).item() if not loss_ptheta_entropies is None else None,
                                        loss_supervision=torch.mean(supervision_loss / (final_t+1))
                                    )
                            
                            epoch_supervision_loss += loss.item()
                            
                            epoch_y.append(torch.tensor(y.cpu().tolist()))  # really strange memory leak, y does not work
                            epoch_filenames.append(data.file_name)
                            epoch_logits.append(final_logits.detach())
                            epoch_y_hat.append(torch.tensor(Y_hat.detach().cpu().tolist()))
                            # clone numpy array
                            
                        
                    
                    if self.config.get('profile',None):     
                        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))          
            # Multistep
            elif self.config.get('MULTISTEP_enabled',False):
                self.supervision_criterion= torch.nn.functional.binary_cross_entropy_with_logits
                for minibatch_index, data in enumerate(tqdm(dataloader)):
                    data = data.cuda(non_blocking=True)
                    adj_initial, features, y, num_nodes = data.adj, data.x, data.y, data.numnodes
                    
                    if self.config.get('loader_add_naive_predictor',False):
                        # add naive predictor: 1 if num_nodes > 102 else 0 as well as add num_nodes
                        # features shape is batch x num_nodes x num_features
                        feat = torch.cat(((num_nodes > 102).to(torch.float32).unsqueeze(-1),num_nodes.unsqueeze(-1) ), dim=-1).to(features.device)
                        features= torch.cat((features, feat.unsqueeze(1).repeat(1,features.shape[1],1)), dim=-1)
                    
                    # adj initial normalization is inside multi_step model, if gcn is the gnn (not for gin)
                    minibatch_mask = torch.zeros((features.shape[0],features.shape[1], 1), device=num_nodes.device, dtype=torch.bool)
                    for i, n in enumerate(num_nodes):
                        minibatch_mask[i,:n,:] = True 
                        
                    with torch.autocast(device_type='cuda') if self.config.get('GSL_mixed_precision',None) else nullcontext():
                        h_classif = self.multi_step(features, adj_initial, minibatch_mask) 
                        final_logits = self.task_head(h_classif)
                        if final_logits.dim()!=1:
                            final_logits = final_logits.squeeze(-1)
                            
                        weights = self.pos_class_weight * y + self.neg_class_weight * (1-y)
                        loss = self.supervision_criterion(final_logits, y, weight=weights, reduction='mean')
                      
                    
                    Y_hat = final_logits > 0   
                    
                    
                    # track loss
                    if  self.config.get('GSL_mixed_precision',None):
                        epoch_supervision_loss += scaler.scale(loss).item()
                        epoch_y.append(torch.tensor(y.cpu().tolist()))  # really strange memory leak, y does not work
                        epoch_y_hat.append(torch.tensor(Y_hat.detach().cpu().tolist()))
                        epoch_filenames.append(data.file_name)
                        epoch_logits.append(final_logits.detach())
                    else:
                        epoch_supervision_loss += loss.item()
                        epoch_y.append(y.detach())
                        epoch_y_hat.append(Y_hat.detach())
                        epoch_filenames.append(data.file_name)
                        epoch_logits.append(final_logits.detach())

                    if is_train_val_test == 'train':
                        if self.config.get('GSL_mixed_precision',None):
                            scaler.scale(loss).backward()
                        else: 
                            loss.backward()
                            
                        if self.config.get('loader_batch_size',None) not in [None, '']:
                                        
                            if ((minibatch_index+1)%(self.config['batch_size'] // self.config['loader_batch_size']) == 0 or minibatch_index == len(dataloader)-1):
                                if self.config.get('GSL_mixed_precision',None):
                                    scaler.step(self.optimizer)
                                    scaler.update()
                                    self.optimizer.zero_grad(set_to_none=True)
                                else:
                                    self.optimizer.step()
                                    self.optimizer.zero_grad(set_to_none=True)
                        else:
                            if self.config.get('GSL_mixed_precision',None):
                                    scaler.step(self.optimizer)
                                    scaler.update()
                                    self.optimizer.zero_grad(set_to_none=True)
                            else:
                                self.optimizer.step()
                                self.optimizer.zero_grad(set_to_none=True)
                    
                        if minibatch_index % 100 == 0:
                            Y_hat = Y_hat.detach()
                            self.write_metrics(
                                split=is_train_val_test,
                                save_epoch=False,
                                y=y.detach(), y_hat=Y_hat, loss=loss.item(), 
                                epoch=epoch,
                                minibatch_index= minibatch_index, 
                                episode=1, 
                                max_minibatch_index=len(dataloader),
                                max_episode=episodes,
                                minibatch_size=self.config['batch_size'],
                                loss_supervision=loss
                            )
                            
                            
            _, precision, recall, f1, balanced_accuracy, TP, FP, TN, FN = self.write_metrics(
                split=is_train_val_test,
                save_epoch=True,
                y=torch.concatenate(epoch_y,dim=0).squeeze(), y_hat=torch.concatenate(epoch_y_hat,dim=0).squeeze(), loss=(epoch_supervision_loss/episodes)/len(dataloader), 
                epoch=epoch if writer_epoch is None else writer_epoch,  # validation passes writer_epoch
            )
            
            
            
            
            if is_train_val_test == 'train':  # get validation stats
                # recursive call
                with torch.no_grad():
                    # set training to false 
                    if hasattr(self, 'task_head') and self.task_head is not None:
                        self.task_head.eval()

                    if hasattr(self, 'gnn') and self.gnn is not None:
                        self.gnn.eval()

                    if hasattr(self, 'graph_structure_learner') and self.graph_structure_learner is not None:
                        self.graph_structure_learner.eval()

                    if hasattr(self, 'multi_step') and self.multi_step is not None:
                        self.multi_step.eval()
                 
            
                    self.train(
                        dataloader=val_dataloader,
                        epochs=1,
                        episodes=1,
                        is_train_val_test='val',
                        writer_epoch=epoch
                    )
                    
                    if hasattr(self, 'multi_step') and self.multi_step is not None:
                        self.multi_step.train()
                    if hasattr(self, 'task_head') and self.task_head is not None: 
                        self.task_head.train()
                    if hasattr(self, 'gnn') and self.gnn is not None:
                        self.gnn.train()
                    if hasattr(self, 'graph_structure_learner') and self.graph_structure_learner is not None:
                        self.graph_structure_learner.train()
                    
                        
                    
            if is_train_val_test != 'train' :
                self.val_loss_stopping_criterion = epoch_supervision_loss/episodes
                self.current_val_metrics = {'epoch':writer_epoch,'loss':epoch_supervision_loss/episodes,'precision': precision, 'recall': recall, 'f1': f1, 'balanced_accuracy': balanced_accuracy, 'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}

                if self.config.get('save_logits',False):
                    epoch_filenames_temp = []
                    for names in epoch_filenames:
                        epoch_filenames_temp += names
                        
                    self.save_logits(torch.concatenate(epoch_y,dim=0).squeeze(), torch.concatenate(epoch_y_hat,dim=0).squeeze(), epoch_filenames, torch.concatenate(epoch_logits,dim=0).squeeze())
            
            else:  # check val loss in training
                # if best_val_loss did not decrease last 10 epochs return:
                print('check loss', self.val_loss_stopping_criterion, self.best_val_loss_stopping_criterion - 1e-5)
                if self.val_loss_stopping_criterion < self.best_val_loss_stopping_criterion - 1e-5:
                    print('best epoch, previous', epoch, self.best_val_loss_epoch)
                    print('best loss, previous', self.val_loss_stopping_criterion, self.best_val_loss_stopping_criterion)
                    self.best_val_loss_stopping_criterion = self.val_loss_stopping_criterion
                    self.best_val_loss_epoch = epoch
                    self.best_val_metrics = self.current_val_metrics
                    if self.config.get('save_best_model',False):
                             self.save_full_model(
                                epoch=None,
                                multi_step=self.multi_step,
                                task_head=self.task_head,
                                gnn=self.gnn,
                                graph_structure_learner=self.graph_structure_learner,
                                optim=self.optimizer,
                                label='best_model'
                            )
                    
                elif epoch - self.best_val_loss_epoch > self.config['OPTIM_stopping_patience']:
                    # save best val metrics
                    print('best val metrics', self.best_val_metrics)
                    # save best metrics
                    yaml.dump(self.best_val_metrics, open(f'{self.model_folder}/best_val_metrics.yml', 'w'))
                    self.writer.close()
                    return deepcopy(self.best_val_loss_stopping_criterion), deepcopy(self.best_val_metrics)
                
        
        
        if is_train_val_test == 'train':
            yaml.dump(self.best_val_metrics, open(f'{self.model_folder}/best_val_metrics.yml', 'w'))
            self.writer.close()
            return deepcopy(self.best_val_loss_stopping_criterion), deepcopy(self.best_val_metrics)  # if stopping criterion was not met, and we run out of epochs 
                    


    
    def save_full_model(self, graph_structure_learner=None, gnn=None,task_head=None, optim=None, multi_step=None, pretraining_optimizers=None, pretraining_heads=None, epoch='', label=''):
        print('save model', 'epoch', epoch, 'label:', label)
        dict_ = {}
        if graph_structure_learner is not None:
            dict_['graph_structure_learner'] = graph_structure_learner.state_dict()
        if gnn is not None:
            dict_['gnn'] = gnn.state_dict()
        if task_head is not None:
            dict_['task_head'] = task_head.state_dict()
        if multi_step is not None:
            dict_['multi_step'] = multi_step.state_dict()
        if pretraining_optimizers is not None:
            dict_['pretraining_optimizers'] = {k:v.state_dict() for k,v in pretraining_optimizers.items()}
        if pretraining_heads is not None:
            dict_['pretraining_heads'] = pretraining_heads.state_dict()
        if optim is not None:
            dict_['optimizer'] = optim.state_dict()
            
        torch.save(dict_, f'{self.model_folder}/{epoch}{label}.pt')
        
    def load_full_model(self, path):
        dict_ = torch.load(path)
        print('model loading keys:', dict_.keys())
        if 'graph_structure_learner' in dict_:
            self.graph_structure_learner.load_state_dict(dict_['graph_structure_learner'])
        if 'gnn' in dict_:
            self.gnn.load_state_dict(dict_['gnn'])
        if 'task_head' in dict_:
            self.task_head.load_state_dict(dict_['task_head'])
        if 'multi_step' in dict_:
            self.multi_step.load_state_dict(dict_['multi_step'])
        if 'pretraining_optimizers' in dict_:
            for k,v in dict_['pretraining_optimizers'].items():
                self.pretraining_optimizers[k].load_state_dict(v)
        if 'pretraining_heads' in dict_:
            self.pretraining_heads.load_state_dict(dict_['pretraining_heads'])
        if 'optimizer' in dict_:
            self.optimizer.load_state_dict(dict_['optimizer'])

    def save_logits(self, y, y_hat, filenames, logits):
        dict_ = {'y': y, 'y_hat': y_hat, 'filenames': filenames, 'logits': logits}
        torch.save(dict_, f'{self.model_folder}/logits.pt')

    def save_model(self, task_head, gnn, graph_structure_learner, optimizer, epoch):
        dict_ = {}
        dict_['task_head'] = task_head.state_dict()
        dict_['gnn'] = gnn.state_dict()
        dict_['graph_structure_learner'] = graph_structure_learner.state_dict()
        dict_['optimizer'] = optimizer.state_dict()
        torch.save(dict_, f'{self.model_folder}/{epoch}.pt')        
            
    def delete_model(self, epoch):
        if os.path.exists(f'{self.model_folder}/{epoch}.pt'):
            os.remove(f'{self.model_folder}/{epoch}.pt')

    