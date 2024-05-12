import os 
import yaml 


import os
import yaml

def get_model_performance(key_metric, cv_num, cv_folder):
    """
    Get model performance based on key metric from cv_x folders
    """
    model_performance = []
    
    # get all configuration directories
    dirs = [d for d in os.listdir(cv_folder) if os.path.isdir(os.path.join(cv_folder, d))]
    if dirs[0].startswith('cv_'):
        dirs = ['']
    
    for d in dirs:
        metric_folder = os.path.join(cv_folder, d, 'cv_{}'.format(cv_num))
        # there is a child folder and then the metric file
        try:
            child_folder = os.listdir(metric_folder)[0]
            
            metric_file = os.path.join(metric_folder, child_folder,  'models','best_val_metrics.yml')
        except FileNotFoundError:
            continue
        
      
        if os.path.isfile(metric_file):
            with open(metric_file, 'r') as file:
                metrics = yaml.safe_load(file)
                try:
                    if key_metric in metrics:
                        model_performance.append((d, metrics[key_metric]))
                except TypeError as e:
                    # print('Error in file: {}'.format(metric_file))
                    continue

    model_performance.sort(key=lambda x: x[1], reverse=True)
    return model_performance
# _batch_size_64_global_hidden_channels_64_GNN_dropout_0.3_GNN_num_layers_3

def main():
   
    folder = 'runs/20240323_220149_CV_2_BareboneGIN_grid_crossval_sl_ud_nodeg'


    key_metric = 'balanced_accuracy' 
    for key_metric in ['balanced_accuracy', 'f1','precision','recall', 'TP','FP','TN','FN', 'loss', 'epoch']:
        print('')
        print(f'=================={key_metric}==================')
        total_performance = {}
        for i in range(5):
            model_performance = get_model_performance(key_metric, i, folder)
            for d, performance in model_performance:
                if d not in total_performance:
                    total_performance[d] = []
                total_performance[d].append(performance)
        
        
        sorting = []
        for d, performances in total_performance.items():
            # get mean and std
            try:
                mean = sum(performances)/len(performances)
                divisor = len(performances) - 1 
                divisor = 1 if divisor == 0 else divisor
                sample_variance = sum([(x - mean)**2 for x in performances])/(divisor)
                std = sample_variance**0.5
                sorting.append((d, mean, std))
            except ZeroDivisionError:
                continue
    
        for d, mean, std in sorted(sorting, key=lambda x: x[1], reverse=True):
            print('{} {:.4f} ± {:.4f}'.format(d, round(mean, 4), round(std, 4)), len(performances))
            
        # average across all  configurations
        performances = [x for x in total_performance.values() for x in x ]
        mean = sum(performances)/len(performances)
        divisor = len(performances) - 1 
        divisor = 1 if divisor == 0 else divisor
        sample_variance = sum([(x - mean)**2 for x in performances])/(divisor)
        std = sample_variance**0.5
        print('{:.4f} ± {:.4f}'.format(mean, std))

if __name__ == '__main__':
    main()