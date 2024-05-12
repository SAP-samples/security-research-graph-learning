# import torch 
# from tqdm.auto import tqdm 
# import os 
# from pathlib import Path 
# from torch_geometric.utils import to_dense_adj
# from torch.nn.functional import pad

# parentfolder = Path('codegraphs/diversevul/v2_undirected_withdegreecount')
# folder = Path('codegraphs/diversevul/v2_undirected_withdegreecount_densepreprocessed')

# files = os.listdir(parentfolder)
# for file in tqdm(files): 
#     data = torch.load(parentfolder/file)
#     # fill up to size 1000 
#     data.numnodes = torch.tensor([data.x.shape[0]])
#     data.adj = to_dense_adj(data.edge_index)
#     data.edge_index = None
#     # temp = torch.zeros(1,1000,1000)
#     # temp[:,:adj.shape[1],:adj.shape[2]] = adj 
#     # data.adj = temp
#     data.adj = pad(data.adj, (0,1000-data.adj.shape[1],0,1000-data.adj.shape[2]), value=0)
#     data.x = pad(data.x, (0,0,0,1000-data.x.shape[0]), value=0).unsqueeze(0)
#     # temp = torch.zeros(1,1000, data.x.shape[1])
#     # temp[:,:data.x.shape[0],:] = data.x
#     # data.x = temp
#     torch.save(data,folder/file)

import torch 
from tqdm.auto import tqdm 
import os 
from pathlib import Path 
from torch_geometric.utils import to_dense_adj
from torch.nn.functional import pad
from multiprocessing import Pool, cpu_count


def process_file(file): 
    if (folder/file).exists():
        return
    data = torch.load(parentfolder/file)
    # fill up to size 1000 
    data.numnodes = torch.tensor([data.x.shape[0]])
    data.adj = to_dense_adj(data.edge_index)
    data.edge_index = None
    data.adj = pad(data.adj, (0,1000-data.adj.shape[1],0,1000-data.adj.shape[2]), value=0)
    data.x = pad(data.x, (0,0,0,1000-data.x.shape[0]), value=0).unsqueeze(0)
    torch.save(data,folder/file)

parentfolder = Path('codegraphs/diversevul/v2_undirected_withdegreecount')
folder = Path('codegraphs/diversevul/v2_undirected_withdegreecount_densepreprocessed')
os.makedirs(folder, exist_ok=True)

files = os.listdir(parentfolder)

with Pool(cpu_count()) as p:
    list(tqdm(p.imap(process_file, files), total=len(files)))