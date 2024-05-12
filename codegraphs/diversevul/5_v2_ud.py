import torch 
# import dataset from torch geometric
from torch_geometric.data import Data, DataLoader, HeteroData
from tqdm.auto import tqdm 
import os 
from pathlib import Path
from copy import deepcopy

for file in tqdm(os.listdir("v2_directed_withdegreecount_heterogeneous")):
    # skip if exiss already
    # if Path("v2_undirected_withdegreecount/" + file).is_file() and Path("v2_undirected_withdegreecount_heterogeneous/" + file).is_file():
        # continue
    
    data = torch.load("v2_directed_withdegreecount_heterogeneous/" + file)
    if data is False:
        continue
    
    
    
    TOTAL_undirected = torch.cat([data['node','TOTAL','node'].edge_index, data['node','TOTAL','node'].edge_index.flip(0)], dim=1)
    # remove duplicates
    TOTAL_undirected = torch.unique(TOTAL_undirected, dim=1, sorted=True)
    
    new_data = Data(
        x = data['node'].x,
        edge_index = TOTAL_undirected,
        y = data['node'].y
    )
    
    
    new_heterogeneous_data = deepcopy(data)
    for edge_type in data.edge_types:
        # if there is non unique, print file
        # if not torch.equal(data[edge_type].edge_index, torch.unique(data[edge_type].edge_index, dim=1, sorted=True)):
        #     print(file)
        #     print(edge_type)
        
        ud = data[edge_type].edge_index
        ud = torch.cat([ud, ud.flip(0)], dim=1)
        # remove duplicates
        ud = torch.unique(ud, dim=1, sorted=True)
        new_heterogeneous_data[edge_type].edge_index = ud
        
    
    
    # save to v2_directed_withdegreecount
    # torch.save(new_data, "v2_undirected_withdegreecount/" + file)
    torch.save(new_heterogeneous_data.cpu(), "v2_undirected_withdegreecount_heterogeneous/" + file)
    