import os 
from pathlib import Path
from tqdm.auto import tqdm 
import torch

for file in tqdm(os.listdir('codegraphs/diversevul/v2_directed_withdegreecount_heterogeneous_unnormalized')):
    try:
        features = torch.load(f'codegraphs/diversevul/v2_directed_withdegreecount_heterogeneous_unnormalized/{file}')['node'].x 
        data_undirected = torch.load(f'codegraphs/diversevul/v2_undirected_withdegreecount/{file}')
        data_undirected.x = features
        torch.save(data_undirected, f'codegraphs/diversevul/v2_undirected_withdegreecount_unnormalized/{file}')
    except BaseException as e:
        print(e)
        print(file)
        print('\n')
        continue
        
     