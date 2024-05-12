from CodeGraphDataset import CodeGraphDataset
from torch_geometric.data import DataLoader
import torch 
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


# %%
def num_nodes(adj, x, threshold, numn):
    return numn

def avg_degree(adj, x, threshold,numn):
    return torch.mean(torch.sum(adj, dim=2), dim=1)

def avg_triangle_counts(adj, x, threshold,numn):
    return torch.mean(x[:, :, 17], dim=1)  

dict_ = {
    'num_nodes': {'predictor':num_nodes,'threshold_range':[0,2000,1]} ,
    # 'avg_degree': {'predictor':avg_degree,'threshold_range':[0,1000,1, 0.05]} ,
    # 'avg_triangle_counts': {'predictor':avg_triangle_counts,'threshold_range':[0,25,1]}
}

def calc_metrics(y, y_hat):
    TP = torch.sum(y*y_hat)
    FP = torch.sum((1-y)*y_hat)
    FN = torch.sum(y*(1-y_hat))
    TN = torch.sum((1-y)*(1-y_hat))
    
    # check no zeros in denominator
    precision = TP/(TP+FP) if TP+FP != 0 else 0
    recall = TP/(TP+FN) if TP+FN != 0 else 0
    f1 = 2*(precision*recall)/(precision+recall) if precision+recall != 0 else 0
    balanced_accuracy = 0.5*(TP/(TP+FN) + TN/(TN+FP))
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    TOTAL = TP+FP+FN+TN
    return TOTAL, TP, FP, FN, TN, precision, recall, f1, balanced_accuracy, accuracy

import os 
import pandas as pd
# make threhold folder
if not os.path.exists('thresholds'):
    os.makedirs('thresholds')
    
data = [] 
for metric_name, metric in dict_.items():
    
    first_threshold = None
    test_dataset = CodeGraphDataset(pt_folder='codegraphs/diversevul/v2_undirected_withdegreecount', DS_type = 'larger10smaller1000', split='test', is_cross_val=False
                        ,remove_degreeandtriangles=False, dense_mode=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)

    for i in range(5):
        precomputed_metric, ys = [],[]
        val_dataset = CodeGraphDataset(pt_folder='codegraphs/diversevul/v2_undirected_withdegreecount', DS_type = 'larger10smaller1000', split='val', cross_val_valfold_idx=i, is_cross_val=True is not None, cross_val_val_fraction=1
                        ,remove_degreeandtriangles=False, dense_mode=True)
        
        train_dataset = CodeGraphDataset(pt_folder='codegraphs/diversevul/v2_undirected_withdegreecount', DS_type = 'larger10smaller1000', split='train', cross_val_valfold_idx=i, is_cross_val=True is not None, cross_val_val_fraction=1
                        ,remove_degreeandtriangles=False, dense_mode=True)
        
        
        
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)
        loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        for minibatch in tqdm(loader):
            # print(minibatch)
            adj, x,y,numnd = minibatch.adj, minibatch.x, minibatch.y, minibatch.numnodes
            precomputed_y_hat = metric['predictor'](adj, x, 0, numnd)
            precomputed_metric.append(precomputed_y_hat.to(torch.int))
            ys.append(y)
        
        ys = torch.cat(ys)
        precomputed_metric = torch.cat(precomputed_metric)
        threhold_data = [] 
        for threshold in range(metric['threshold_range'][0], metric['threshold_range'][1], metric['threshold_range'][2]):
            if len(metric['threshold_range']) == 4:
                threshold = threshold * metric['threshold_range'][3]
                
            # if not os.path.exists(f'thresholds/{metric_name}'):
            #     # make dir 
            #     os.makedirs(f'thresholds/{metric_name}')
            # if not os.path.exists(f'thresholds/{metric_name}/data.csv'):
            #     # make df
            #     df = pd.DataFrame(columns=['threshold', 'min_threshold','max_threshold','fold', 'TOTAL', 'TP', 'FP', 'FN', 'TN', 'precision', 'recall', 'f1', 'balanced_accuracy', 'accuracy'])
            # else: 
            #     df = pd.read_csv(f'thresholds/{metric_name}/data.csv')
                
            # for i in range(5):
                
                
            #     y_hats, ys = [], []
            #     for minibatch in tqdm(loader):
            #         # print(minibatch)
            #         adj, x,y,numnd = minibatch.adj, minibatch.x, minibatch.y, minibatch.numnodes
            #         y_hat = metric['predictor'](adj, x, threshold, numnd)
            #         # y_hat = model(adj, x, threshold)
            #         y_hats.append(y_hat)
            #         ys.append(y)
                
            #     y = torch.cat(ys)
            #     y_hat = torch.cat(y_hats)
            #     y_hat = y_hat.to(torch.int)
            y_hat = precomputed_metric > threshold
            y_hat = y_hat.to(torch.int)
            TOTAL, TP, FP, FN, TN, precision, recall, f1, balanced_accuracy, accuracy = calc_metrics(ys, y_hat)
            TOTAL = TOTAL.item()
            TP, FP, FN, TN = TP.item(), FP.item(), FN.item(), TN.item()
            
            
            precision, recall, f1, balanced_accuracy, accuracy = precision, recall, f1, balanced_accuracy.item(), accuracy.item()
            
            if type(precision) == torch.Tensor:
                precision = precision.item()
            if type(recall) == torch.Tensor:
                recall = recall.item()
            if type(f1) == torch.Tensor:
                f1 = f1.item()
            
            threhold_data.append([threshold, balanced_accuracy])
            
        # choose threshold that maximizes balanced accuracy
        # sort by balanced accuracy
        threhold_data = sorted(threhold_data, key=lambda x: x[1], reverse=True)
        # take threshold that maximizes balanced accuracy
        threshold = threhold_data[0][0]
        if i==0:
            first_threshold = threshold
        # evaluate on val set
        save_y_hats = []
        save_y_s = []
        for minibatch in tqdm(val_loader):
            # print(minibatch)
            adj, x,y,numnd = minibatch.adj, minibatch.x, minibatch.y, minibatch.numnodes
            y_hat = metric['predictor'](adj, x, threshold, numnd) > threshold
            y_hat = y_hat.to(torch.int)
            save_y_hats.append(y_hat)
            save_y_s.append(y)
            # # TOTAL, TP, FP, FN, TN, precision, recall, f1, balanced_accuracy, accuracy = calc_metrics(y, y_hat)
            # TOTAL = TOTAL.item()
            # # TP, FP, FN, TN = TP.item(), FP.item(), FN.item(), TN.item()
            
            
            # precision, recall, f1, balanced_accuracy, accuracy = precision, recall, f1, balanced_accuracy.item(), accuracy.item()
            
            # if type(precision) == torch.Tensor:
            #     precision = precision.item()
            # if type(recall) == torch.Tensor:
            #     recall = recall.item()
            # if type(f1) == torch.Tensor:
            #     f1 = f1.item()
            
            
                       
            
            
            # add to df
            # data.append({'threshold' : threshold, 'min_threshold':metric['threshold_range'][0],'max_threshold':metric['threshold_range'][1],'fold':i, 'TOTAL':TOTAL, 'TP':TP, 'FP':FP, 'FN':FN, 'TN':TN, 'precision':precision, 'recall':recall, 'f1':f1, 'balanced_accuracy':balanced_accuracy, 'accuracy':accuracy})
        val_ys = torch.cat(save_y_s)
        val_y_hats = torch.cat(save_y_hats)
        TOTAL, TP, FP, FN, TN, precision, recall, f1, balanced_accuracy, accuracy = calc_metrics(val_ys, val_y_hats)
        TOTAL = TOTAL.item()
        TP, FP, FN, TN = TP.item(), FP.item(), FN.item(), TN.item()
        
        
        precision, recall, f1, balanced_accuracy, accuracy = precision, recall, f1, balanced_accuracy.item(), accuracy.item()
        
        if type(precision) == torch.Tensor:
            precision = precision.item()
        if type(recall) == torch.Tensor:
            recall = recall.item()
        if type(f1) == torch.Tensor:
            f1 = f1.item()
            
            
        data.append([metric_name,threshold,metric['threshold_range'][0],metric['threshold_range'][1],i,TOTAL,TP,FP,FN,TN,precision,recall,f1,balanced_accuracy,accuracy])
        
        # save df
        # df.to_csv(f'thresholds/{metric_name}/data.csv', index=False)
    
    # test set performance
    test_data = []
    test_ys = []
    test_y_hats = []
    for i,minibatch in tqdm(enumerate(test_loader)):
        # print(minibatch)
        adj, x,y,numnd = minibatch.adj, minibatch.x, minibatch.y, minibatch.numnodes
        y_hat = metric['predictor'](adj, x, first_threshold, numnd) > first_threshold
        y_hat = y_hat.to(torch.int)
        test_ys.append(y)
        test_y_hats.append(y_hat)
        # TOTAL, TP, FP, FN, TN, precision, recall, f1, balanced_accuracy, accuracy = calc_metrics(y, y_hat)
        # TOTAL = TOTAL.item()
        # TP, FP, FN, TN = TP.item(), FP.item(), FN.item(), TN.item()
        
        
        # precision, recall, f1, balanced_accuracy, accuracy = precision, recall, f1, balanced_accuracy.item(), accuracy.item()
        
        # if type(precision) == torch.Tensor:
        #     precision = precision.item()
        # if type(recall) == torch.Tensor:
        #     recall = recall.item()
        # if type(f1) == torch.Tensor:
        #     f1 = f1.item()
    
    test_ys1 = torch.cat(test_ys)
    test_y_hats1 = torch.cat(test_y_hats)
    
    TOTAL, TP, FP, FN, TN, precision, recall, f1, balanced_accuracy, accuracy = calc_metrics(test_ys1, test_y_hats1)
    TOTAL = TOTAL.item()
    TP, FP, FN, TN = TP.item(), FP.item(), FN.item(), TN.item()
    
    
    precision, recall, f1, balanced_accuracy, accuracy = precision, recall, f1, balanced_accuracy.item(), accuracy.item()
    
    if type(precision) == torch.Tensor:
        precision = precision.item()
    if type(recall) == torch.Tensor:
        recall = recall.item()
    if type(f1) == torch.Tensor:
        f1 = f1.item()
        
        
    test_data.append([metric_name,first_threshold,metric['threshold_range'][0],metric['threshold_range'][1],i,TOTAL,TP,FP,FN,TN,precision,recall,f1,balanced_accuracy,accuracy])
    
        # data.append([metric_name,first_threshold,metric['threshold_range'][0],metric['threshold_range'][1],i,TOTAL,TP,FP,FN,TN,precision,recall,f1,balanced_accuracy,accuracy])
# save test dat
df = pd.DataFrame(test_data, columns=['metric_name','threshold', 'min_threshold','max_threshold','fold', 'TOTAL', 'TP', 'FP', 'FN', 'TN', 'precision', 'recall', 'f1', 'balanced_accuracy', 'accuracy'])
df.to_csv(f'thresholds/baseline_test.csv', index=False)
 
df = pd.DataFrame(data, columns=['metric_name','threshold', 'min_threshold','max_threshold','fold', 'TOTAL', 'TP', 'FP', 'FN', 'TN', 'precision', 'recall', 'f1', 'balanced_accuracy', 'accuracy'])
df.to_csv(f'thresholds/baseline.csv', index=False)



