from models.GCN import GCN, GINAdapted, GCNAdapted, GIN  # for GraphGLOW only
from models.BinaryClassifierReadOutLayer import BinaryClassifierReadOutLayer, BinaryClassifierReadOutLayerDense, PretrainingHeads
from models.GraphStructureLearner import GraphStructureLearner
from models.StructureLearning import StructureLearning 
from models.BareboneGCN import BareboneGCN
from models.BareboneGIN import BareboneGIN
from models.HGPSL import HGPSL
from models.Reveal import Reveal
from models.MultiStep import MultiStep
from models.BareboneRGCN import BareboneRGCN
from models.BareboneDRGCN import BareboneDRGCN 
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from CodeGraphDataset import CodeGraphDataset, CodeGraphDataset_InMemory
import torch.nn.functional as F
import torch
import yaml
from torch_geometric.loader import DataLoader
import itertools
import copy
from copy import deepcopy
from pathlib import Path
import datetime
import os 
import gc




class PlaceholderModel(torch.nn.Module):
    def __init__(self, *args,**kwargs):
        super().__init__()
        
    def forward(self):
        pass

models = {
    'GNN':{
        'GCN': GCN, # used for GraphGLOW only
        'GIN': GIN, # used for GraphGLOW only
        'BareboneGCN': BareboneGCN,
        'BareboneGIN': BareboneGIN,
        'BareboneRGCN':BareboneRGCN,
        'BareboneDRGCN':BareboneDRGCN,
        'Reveal': Reveal,
        'HGPSL': HGPSL,
        'GCNAdapted':GCNAdapted,
        'GINAdapted' :GINAdapted,
        '': PlaceholderModel
    },
    'H':{
        'BinaryClassifierReadOutLayer': BinaryClassifierReadOutLayer,
        'BinaryClassifierReadOutLayerDense':BinaryClassifierReadOutLayerDense,
        '': PlaceholderModel
    },
    'GSL':{
        'GraphStructureLearner': GraphStructureLearner,
        # 'MultiStep': MultiStep,
        '': PlaceholderModel
    },
    'MULTISTEP':{
        'MultiStep': MultiStep,
        '': PlaceholderModel
    },
    'OPTIM': {
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
        '': PlaceholderModel
    },
    
    
}



def get_models(config):
    
    gnn_args = {
        'hidden_channels':config.get('global_hidden_channels', None), 
        'num_layers':config.get('GNN_num_layers', None), 
        'dropout':config.get('GNN_dropout', None),
        'pooling':config.get('HGP_pooling', None),
        'sample_neighbor':config.get('HGP_sample_neighbor', None),
        'sparse_attention':config.get('HGP_sparse_attention', None),
        'structure_learning':config.get('HGP_structure_learning', None),
        'lamb':config.get('HGP_lamb', None),
        'readout_pooling':config.get('H_pool', None),
        'num_m' : config.get('DRGCN_num_m', None),
        'num_n' : config.get('DRGCN_num_n', None),
        'num_o' : config.get('DRGCN_num_o', None),
        'aggr' : config.get('DRGCN_aggr', None),
        'jk' : config.get('GNN_jumping_knowledge', None),
    }
    gsl_args = {
        'input_channels':config.get('global_hidden_channels', None), 
        'num_heads':config.get('GSL_num_heads', None), 
        'attention_threshold_epsilon':config.get('GSL_attention_threshold', None),
    }
    head_args = {
        'hidden_channels':config.get('global_hidden_channels', None) if not config.get('MULTISTEP_enabled', False) else config.get('MULTISTEP_hidden_channels', None),
        'pooling':config.get('H_pool', None), 
        'num_layers':config.get('GNN_num_layers', None) if not config.get('MULTISTEP_enabled', False) else config.get('MULTISTEP_num_steps', None),
        'jk':(not config.get('MULTISTEP_enabled', False) and config.get('GNN_jumping_knowledge', False)) or (config.get('MULTISTEP_enabled', False) and config.get('MULTISTEP_jumping_knowledge', False)),
    }
    gnn = models['GNN'][config['GNN']](
        **gnn_args
    )
    
    gsl = models['GSL'][config['GSL']](**gsl_args)
    # num_layers for GIN Jumping Knowledge
    head = models['H'][config['H']](**head_args)
    # MULTISTEP_same_as_GNN
    if config.get('MULTISTEP_same_as_GNN',True):
        multistep_gnn_args = gnn_args
        multistep_gnn = models['GNN'][config['GNN']]
    else:
        multistep_gnn = models['GNN'][config['MULTISTEP_GNN']]
        multistep_gnn_args = {
            'hidden_channels':config.get('MULTISTEP_hidden_channels', None),
            'num_layers':config.get('MULTISTEP_num_layers', None),
            'dropout':config.get('MULTISTEP_dropout', None),
            'readout_pooling':config.get('MULTISTEP_pooling', None),
            'in_channels': config.get('MULTISTEP_in_channels', 162),
        }
    
    if 'MULTISTEP' not in config:
        multistep = models['MULTISTEP']['']()
    else:
        multistep = models['MULTISTEP'][config['MULTISTEP']](
            gslgnn_class = models['GNN'][config['GNN']],
            gslgnn_args = gnn_args,
            multistepgnn_class = multistep_gnn,
            multistepgnn_args = multistep_gnn_args,
            
            
    gsl_class = models['GSL'][config['GSL']],
        gsl_args = gsl_args,
        multistep_gnn = multistep_gnn,
        GSL_skip_ratio = config['GSL_skip_ratio'],
        MULTISTEP_num_steps = config['MULTISTEP_num_steps'],
        MULTISTEP_jk = config['MULTISTEP_jumping_knowledge'],
        MULTISTEP_jk_aggr = config['MULTISTEP_jk_aggr'],
        )
    if 'MULTISTEP_jumping_knowledge' not in config:
        pretrainingHeads = PlaceholderModel()
        pretraining_optimizers = {}
    else:
        pretrainingHeads = PretrainingHeads(
            hidden_channels=multistep_gnn_args['hidden_channels'],
            num_steps=config['MULTISTEP_num_steps'],
            jk= True if config['MULTISTEP_jumping_knowledge'] and not (config['MULTISTEP_jk_aggr']=='sum' or config['MULTISTEP_jk_aggr']=='mean') else False,
            
        )
        pretraining_optimizers = {
            'degree': models['OPTIM'][config['PRETRAINING_optim']],
            'triangle': models['OPTIM'][config['PRETRAINING_optim']],
            'category': models['OPTIM'][config['PRETRAINING_optim']],
            'wordvector': models['OPTIM'][config['PRETRAINING_optim']],
            'cwe_classification': models['OPTIM'][config['PRETRAINING_optim']],
            'link_prediction': models['OPTIM'][config['PRETRAINING_optim']]
        }
    
    optim = models['OPTIM'][config['OPTIM']]
    return gnn, gsl, head, optim, multistep, pretrainingHeads, pretraining_optimizers


cached_train_dataloader, cached_val_dataloader, cached_split_idx = None, None, None
def grid_search_config(config, cross_val_folder=None, cross_val_valindex=None):
    global cached_train_dataloader, cached_val_dataloader, cached_split_idx
    
    gnn, gsl, head, optim, multistep, pretrainingHeads, pretraining_optimizers = get_models(config)
    
    if len(config['from_saved'])>0:
        
        # names are 0.pt .. n.pt
        # get epoch name 
        if not 'None' in config['from_saved']:
            model_folder = Path('runs')/config['from_saved']/'models' 
            
            from_saved_epoch = int(sorted(model_folder.glob('*.pt'))[-1].stem.split('_')[-1])
            latest_model = sorted(model_folder.glob('*.pt'))[-1]
            if 'gnn' in data:
                gnn.load_state_dict(data['gnn'])
            if 'graph_structure_learner' in data:
                gsl.load_state_dict(data['graph_structure_learner'])
            if 'task_head' in data:
                head.load_state_dict(data['task_head'])
            if 'multistep' in data:
                multistep.load_state_dict(data['multistep'])
            if 'pretrainingHeads' in data:
                pretrainingHeads.load_state_dict(data['pretrainingHeads'])
            
        else:
            # load model inside the structurelearning class
            from_saved_epoch = None
            latest_model = config['from_saved']
            
          
            
        data = torch.load(latest_model)
 
    else:
        from_saved_epoch=None
    
    if len(pretraining_optimizers)>0:
        pretraining_optimizers['degree'] = pretraining_optimizers['degree'](list(pretrainingHeads.degree_linear1.parameters())+list(pretrainingHeads.degree_linear2.parameters()), lr=config['PRETRAINING_lr'], weight_decay=config['PRETRAINING_weight_decay'])
        pretraining_optimizers['triangle'] = pretraining_optimizers['triangle'](list(pretrainingHeads.triangle_linear1.parameters())+list(pretrainingHeads.triangle_linear2.parameters()), lr=config['PRETRAINING_lr'], weight_decay=config['PRETRAINING_weight_decay'])
        pretraining_optimizers['category'] = pretraining_optimizers['category'](list(pretrainingHeads.category_linear1.parameters()), lr=config['PRETRAINING_lr'], weight_decay=config['PRETRAINING_weight_decay'])
        pretraining_optimizers['wordvector'] = pretraining_optimizers['wordvector'](list(pretrainingHeads.wordvector_linear1.parameters())+list(pretrainingHeads.wordvector_linear2.parameters()), lr=config['PRETRAINING_lr'], weight_decay=config['PRETRAINING_weight_decay'])
        pretraining_optimizers['cwe_classification'] = pretraining_optimizers['cwe_classification'](list(pretrainingHeads.cwe_linear1.parameters())+list(pretrainingHeads.cwe_linear2.parameters())+list(pretrainingHeads.cwe_linear3.parameters()), lr=config['PRETRAINING_lr'], weight_decay=config['PRETRAINING_weight_decay'])
        pretraining_optimizers['link_prediction'] = pretraining_optimizers['link_prediction'](list(pretrainingHeads.linkpred_linear1.parameters())+list(pretrainingHeads.linkpred_linear2.parameters())+list(pretrainingHeads.linkpred_linear3.parameters()), lr=config['PRETRAINING_lr'], weight_decay=config['PRETRAINING_weight_decay']) 
        
    params = list(gnn.parameters()) + list(gsl.parameters()) + list(head.parameters()) + list(multistep.parameters())  #+list(pretrainingHeads.parameters())
    optimtemp = optim(params, lr=config['OPTIM_lr'], weight_decay=config['OPTIM_weight_decay'])
    
    
    # joint optimization of all pretraining tasks:
    multitaskparams = params + list(pretrainingHeads.parameters())
    multitaskoptim = optim(multitaskparams, lr=config['OPTIM_lr'], weight_decay=config['OPTIM_weight_decay'])
    optim = optimtemp

    device = 'cuda'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn = gnn.to(device)
    gsl = gsl.to(device)
    head = head.to(device)
    multistep = multistep.to(device)
    pretrainingHeads = pretrainingHeads.to(device)
        

    structureLearning = StructureLearning(
        graph_structure_learner=gsl,
        gnn=gnn,
        task_head=head,
        multi_step=multistep,
        pretraining_heads=pretrainingHeads,
        optimizer=optim,
        pretraining_optimizers=pretraining_optimizers,
        supervision_criterion=F.binary_cross_entropy_with_logits,
        config=config, 
        from_saved=None if (len(config['from_saved'])==0 or 'None' in config['from_saved']) else config['from_saved'],
        from_saved_epoch=from_saved_epoch,
        cross_val_folder=cross_val_folder,
        cross_val_valindex=cross_val_valindex,
        multitaskoptim = multitaskoptim,
    )
    
    
    if config.get('DS_cv_trainfraction', None) not in [None, 1] and config.get('DS_crossval', None) not in [None, False, ''] and not config.get('DS_dense_mode',None) in [True, None]:
        DS = CodeGraphDataset_InMemory
        # is_cv= True
    else:
        DS = CodeGraphDataset
        # is_cv = False
    
    if cached_train_dataloader is not None and cached_split_idx == cross_val_valindex:
        train_dataloader = cached_train_dataloader
        val_dataloader = cached_val_dataloader
    else:
        cached_split_idx = cross_val_valindex
        train_split, val_split = 'train','val'
        
        
        bs = config['batch_size'] if config.get('loader_batch_size',None) is None else config['loader_batch_size']
        
        
        if config.get('DS_alltrain',None):
            train_split = 'all_train'
            val_split = 'train'
            
        ds_train = DS(pt_folder=config['DS_root'], DS_type = config['DS_filtertype'],split=train_split, cross_val_valfold_idx=cross_val_valindex, is_cross_val=cross_val_folder is not None, cross_val_train_fraction=config.get('DS_cv_trainfraction',None)
               ,remove_degreeandtriangles=config.get('DS_nodeg',False), dense_mode=config.get('DS_dense_mode',False),
               pairs_only=config.get('DS_vulnpairs_only',False)
               )
        
        
        
        if config.get('DS_alltrain',None) is True and config.get('DS_cv_trainfraction',1) != 1:
            print('Val is same as train')
            ds_val = ds_train 
        else:
            if config.get('val_split',None) not in [None, '']:
                val_split = config['val_split']
                   
            ds_val =  DS(pt_folder=config['DS_root'], DS_type = config['DS_filtertype'], split=val_split, cross_val_valfold_idx=cross_val_valindex, is_cross_val=cross_val_folder is not None, cross_val_val_fraction=config.get('DS_cv_valfraction',None)
                ,remove_degreeandtriangles=config.get('DS_nodeg',False), dense_mode=config.get('DS_dense_mode',False),
                pairs_only=config.get('DS_vulnpairs_only',False)
                )
        
        print('num_workers:', 4 if config.get('DS_cv_trainfraction',0) == 1 or config.get('DS_dense_mode',False)  else 0)
        print('persistent_workers:', True if config.get('DS_cv_trainfraction',0) == 1 or config.get('DS_dense_mode',False) else False)
        print('pin_memory:', config.get('DS_dense_mode',False))
        print('batch_size:', bs)
     
        
        train_dataloader = DataLoader(
            ds_train,  # .to(device)
            batch_size=bs, 
            shuffle=True,
            pin_memory=False,#config.get('DS_dense_mode',False),
            num_workers=2 if config.get('DS_cv_trainfraction',0) == 1 or config.get('DS_dense_mode',False)  else 0,
            # persistent_workers=True if config.get('DS_cv_trainfraction',0) == 1 or config.get('DS_dense_mode',False) else False,
            drop_last=True,
            # prefetch_factor=2 if config.get('DS_dense_mode',False) else 0
            )
        val_dataloader = DataLoader(
            ds_val,  # .to(device)
            batch_size=bs, 
            shuffle=False,
            pin_memory=False, #config.get('DS_dense_mode',False),
            num_workers=2 if config.get('DS_cv_trainfraction',0) == 1 or config.get('DS_dense_mode',False)  else 0, #if config.get('DS_cv_trainfraction',0) == 1 else 0,
            # persistent_workers=True if config.get('DS_cv_trainfraction',0) == 1 or config.get('DS_dense_mode',False) else False,
            drop_last=True,
            # prefetch_factor=2 if config.get('DS_dense_mode',False) else 0
        )
        
        cached_train_dataloader, cached_val_dataloader = train_dataloader, val_dataloader
        

    print('train minibatch num:', len(train_dataloader))
    print('val minibatch num:', len(val_dataloader))
    
    if config.get('PRETRAINING_enabled',False):
        if config.get('DS_alltrain',None):
            train_split = 'all_train'
            val_split = 'train'
        else:
            train_split, val_split = 'train','val'
            
            
            
        bs = config['PRETRAINING_batch_size'] 
        
        pretrain_ds_train = DS(pt_folder=config['DS_root'], DS_type = config['DS_filtertype'],split=train_split, cross_val_valfold_idx=cross_val_valindex, is_cross_val=cross_val_folder is not None, cross_val_train_fraction=config.get('DS_cv_trainfraction',None)
               ,remove_degreeandtriangles=config.get('DS_nodeg',False), dense_mode=config.get('DS_dense_mode',False)
               )
        train_pretrain_dataloader = DataLoader(
            pretrain_ds_train,  # .to(device)
            batch_size=bs, 
            shuffle=True,
            pin_memory=False,
            num_workers=2 if config.get('DS_cv_trainfraction',0) == 1 or config.get('DS_dense_mode',False)  else 0,
            # persistent_workers=True if config.get('DS_cv_trainfraction',0) == 1 or config.get('DS_dense_mode',False) else False,
            drop_last=True,
            # prefetch_factor=2 if config.get('DS_dense_mode',False) else 0
            )
        pretrain_ds_val = DS(pt_folder=config['DS_root'], DS_type = config['DS_filtertype'], split=val_split, cross_val_valfold_idx=cross_val_valindex, is_cross_val=cross_val_folder is not None, cross_val_val_fraction=config.get('DS_cv_valfraction',None)
                ,remove_degreeandtriangles=config.get('DS_nodeg',False), dense_mode=config.get('DS_dense_mode',False),
                )
        val_pretrain_dataloader = DataLoader(
            pretrain_ds_val,  # .to(device)
            batch_size=bs, 
            shuffle=False,
            pin_memory=False,
            num_workers=2 if config.get('DS_cv_trainfraction',0) == 1 or config.get('DS_dense_mode',False)  else 0, #if config.get('DS_cv_trainfraction',0) == 1 else 0,
            # persistent_workers=True if config.get('DS_cv_trainfraction',0) == 1 or config.get('DS_dense_mode',False) else False,
            drop_last=True,
            # prefetch_factor=2 if config.get('DS_dense_mode',False) else 0
        )
        
        cwe_ds_train = DS(pt_folder=config['DS_root'], DS_type = config['DS_filtertype'],split=train_split, cross_val_valfold_idx=cross_val_valindex, is_cross_val=cross_val_folder is not None, cross_val_train_fraction=config.get('DS_cv_trainfraction',None)
               ,remove_degreeandtriangles=config.get('DS_nodeg',False), dense_mode=config.get('DS_dense_mode',False)
               ,pretrain_cwe=True)
        train_cwe_dataloader = DataLoader(
            cwe_ds_train,  # .to(device)
            batch_size=bs, 
            shuffle=True,
            pin_memory=False,
            num_workers=2 if config.get('DS_cv_trainfraction',0) == 1 or config.get('DS_dense_mode',False)  else 0,
            # persistent_workers=True if config.get('DS_cv_trainfraction',0) == 1 or config.get('DS_dense_mode',False) else False,
            drop_last=True,
            # prefetch_factor=2 if config.get('DS_dense_mode',False) else 0
            )
        cwe_ds_val = DS(pt_folder=config['DS_root'], DS_type = config['DS_filtertype'], split=val_split, cross_val_valfold_idx=cross_val_valindex, is_cross_val=cross_val_folder is not None, cross_val_val_fraction=config.get('DS_cv_valfraction',None)
                ,remove_degreeandtriangles=config.get('DS_nodeg',False), dense_mode=config.get('DS_dense_mode',False),
                pretrain_cwe=True)
        val_cwe_dataloader = DataLoader(
            cwe_ds_val,  # .to(device)
            batch_size=bs, 
            shuffle=False,
            pin_memory=False,
            num_workers=2 if config.get('DS_cv_trainfraction',0) == 1 or config.get('DS_dense_mode',False)  else 0, #if config.get('DS_cv_trainfraction',0) == 1 else 0,
            # persistent_workers=True if config.get('DS_cv_trainfraction',0) == 1 or config.get('DS_dense_mode',False) else False,
            drop_last=True,
            # prefetch_factor=2 if config.get('DS_dense_mode',False) else 0
        )
        
        
        pretraining_traindataloaders={
            'dense':train_pretrain_dataloader,
            'cwe': train_cwe_dataloader,
        }
        pretraining_valdataloaders={
            'dense':val_pretrain_dataloader,
            'cwe': val_cwe_dataloader,
        }
        print('cwe train minibatch num:', len(train_cwe_dataloader))
        print('cwe val minibatch num:', len(val_cwe_dataloader))
    else:
        pretraining_traindataloaders = None
        pretraining_valdataloaders = None

    best_val_loss, best_val_metrics = structureLearning.train(
        dataloader=train_dataloader,
        epochs=config['epochs'],
        episodes=config['episodes'],
        is_train_val_test='train',
        val_dataloader=val_dataloader,
        pretraining_traindataloaders=pretraining_traindataloaders,
        pretraining_valdataloaders=pretraining_valdataloaders,
    )
    
    
    
    
    return best_val_loss, best_val_metrics



def generate_configs(config):
    # ignore pretraining tasks 
    PRETRAINING_block2_tasks = config.get('PRETRAINING_block2_tasks', None)
    PRETRAINING_block1_tasks = config.get('PRETRAINING_block1_tasks', None)
    
    config['PRETRAINING_block2_tasks'] = None
    config['PRETRAINING_block1_tasks'] = None
    # Find keys with list values
    list_keys = [k for k, v in config.items() if isinstance(v, list)]

   
    # Generate all combinations of list values
    combinations = list(itertools.product(*[config[k] for k in list_keys]))

    # Generate a new config for each combination
    configs = []
    for combination in combinations:
        new_config = copy.deepcopy(config)
        special_to_current_config = {}
        for i, key in enumerate(list_keys):
            new_config[key] = combination[i]
            special_to_current_config[key] = combination[i]
        
        
        new_config['PRETRAINING_block2_tasks'] = PRETRAINING_block2_tasks
        new_config['PRETRAINING_block1_tasks'] = PRETRAINING_block1_tasks
        
        configs.append((special_to_current_config,new_config))

    return configs
    

if __name__ == "__main__":
    # read config YAML file
    
    
    def run_config(file):
        global cached_train_dataloader, cached_val_dataloader
        
        with open('configs/'+file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        config['writer_tag'] = file.split('.')[0]
        
        
        if config['DS_crossval'] not in [None, False, '']:
            
            datenow = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            tag = config['writer_tag']
            model_folder = Path(datenow+'_'+'CV'+'_'+tag)
            os.makedirs(Path('runs')/ model_folder, exist_ok=True)
            
            configs = generate_configs(config)
            loss_tracker = {}
            # We have to rewrite code below, so we only need to load the same train/val data split once
            for val_fold_index in range(config['DS_folds']): 
                for i,(unique_to_conf, conf) in enumerate(configs):
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    for k,v in conf.items():
                        print(k, v)
                    
                    if i not in loss_tracker:
                        loss_tracker[i] = {'loss':0, 'conf':conf}
                    
                    current_config_string = ""
                    for k, v in unique_to_conf.items():
                        current_config_string += f'_{k}_{v}'
                    config_folder = model_folder/current_config_string
                    os.makedirs(Path('runs')/config_folder, exist_ok=True)
                    
                    
                    val_loss, best_val_metrics = grid_search_config(conf, cross_val_folder=config_folder, cross_val_valindex=val_fold_index)
                    val_loss = deepcopy(val_loss)
                    best_val_metrics = deepcopy(best_val_metrics)
                    loss_tracker[i]['loss'] += val_loss
                    for metric, value in best_val_metrics.items():
                        if metric == 'loss':
                            metricname = 'val_loss'
                        else:
                            metricname = metric
                        
                        if metricname not in loss_tracker[i]:
                            loss_tracker[i][metricname] = []
                            
                        loss_tracker[i][metricname].append(value)
                    
                
                cached_train_dataloader, cached_val_dataloader = None, None  # free memory
                torch.cuda.empty_cache()
                gc.collect()
            
            # add average metrics to loss_tracker
            for i in loss_tracker:
                for metric in list(loss_tracker[i].keys()):
                    if metric == 'loss' or metric == 'conf':
                        continue
                    loss_values = [float(value) for value in loss_tracker[i][metric]]  # Convert values to integers
                    loss_tracker[i][metric+'_avg'] = sum(loss_values) / len(loss_values)
                    # add median 
                    loss_tracker[i][metric+'_median'] = sorted(loss_values)[len(loss_values)//2]
                    
                    
            # get best config
            best_config = min(loss_tracker, key=lambda x: loss_tracker[x]['loss'])
                
            with open(Path('runs')/model_folder/'best_config.yml', 'w') as file:
                best_config = loss_tracker[best_config]['conf']
                yaml.dump(best_config, file)
            
            # save all configs and their losses 
            with open(Path('runs')/model_folder/'all_configs.yml', 'w') as file:
                yaml.dump(loss_tracker, file)
            
            
            
            
        else:
            for unique_to_conf, conf in generate_configs(config):
                print('=========== training configuration ===========')
                for k,v in conf.items():
                    print(k, v)
                    
                grid_search_config(conf)
    
    
    # cached_train_dataloader = None
    run_config('9_FINAL_GraphGLOW_GIN_crossval_sl_ud_test.yml')

    
    
    