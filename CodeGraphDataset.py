import torch
from torch_geometric.data import InMemoryDataset, Data, HeteroData, Dataset
from tqdm import tqdm
from pathlib import Path
import pickle
import os 
import random
from torch_geometric.utils import to_dense_adj
from torch.nn.functional import pad


def get_filenames(split,cross_val_val_fraction, cross_val_train_fraction, cross_val_valfold_idx, pt_folder, DS_type='all', pretrain_cwe=False, pairs_only=False):
    files = {
        'all': 'train_test_folds_v1_allgraphs.pkl',
        'larger10smaller1000': 'train_test_folds_v1_allgraphs_filtered_smaller10_larger1000.pkl',
        'larger10': 'train_test_folds_v1_allgraphs_filtered_smaller10.pkl',
        'smaller1000': 'train_test_folds_v1_allgraphs_filtered_larger1000.pkl',
        
    }
    # so we get same random file selects for all DS_types, use "all" here, later filter
    data = pickle.load(open(Path(pt_folder).parent/files['all'], 'rb'))
    train = data['train_folds']
    test = data['test']
    
    if DS_type == 'larger10smaller1000mixedsplits':
        
        if not os.path.exists(Path(pt_folder).parent/'mixedsplits.pkl'):
            seed = 14
            random.seed(seed)
            alltrainfiles =[]
            for fold in train:
                alltrainfiles += fold
            random.shuffle(alltrainfiles)
            train = [alltrainfiles[:int(len(alltrainfiles)/5)] for _ in range(5)]
            pickle.dump(train, open(Path(pt_folder).parent/'mixedsplits.pkl', 'wb'))
        else:
            train = pickle.load(open(Path(pt_folder).parent/'mixedsplits.pkl', 'rb'))

        DS_type = 'larger10smaller1000'
    
    splits = [[] for _ in range(len(train))]
    
    non_existing = []
    for i in range(len(train)):
        for file in train[i]:
            # check file exists
            if os.path.exists(Path(pt_folder)/file):
                splits[i].append(file)
            else:
                non_existing.append(file)
    train = splits
    

    
    non_existing_test=[]   
    test_temp = []       
    for file in test:
        if os.path.exists(Path(pt_folder)/file):
            test_temp.append(file)
        else:
            non_existing_test.append(file)
            
    test = test_temp 
    
    if len(non_existing) > 0:
        print(f"Files not found in {pt_folder}: {len(non_existing)}")
    if len(non_existing_test) > 0:
        print(f"Files not found in {pt_folder}: {len(non_existing_test)}")
    
    random.seed(14)
    if split =='val':
        f = train[cross_val_valfold_idx]
        random.shuffle(f)
        filenames = f[:int(len(f)*cross_val_val_fraction)]
                
    elif split == 'train':
        filenames = []
        for i, fold in enumerate(train):
            if i != cross_val_valfold_idx:
                # take only a percentage of the training data
                random.seed(14)  # constant
                random.shuffle(fold)
                fold = fold[:int(len(fold)*cross_val_train_fraction)]
                filenames += fold
    elif split == 'test':
        filenames = test
    elif split== 'all_train':  # also val fold
        filenames = []
        for i, fold in enumerate(train):
            # take only a percentage of the training data
            random.seed(14)  # constant
            random.shuffle(fold)
            fold = fold[:int(len(fold)*cross_val_train_fraction)]
            filenames += fold
    
    
    all_files_for_type = []
    data = pickle.load(open(Path(pt_folder).parent/files[DS_type], 'rb'))
    for fold in data['train_folds']:
        all_files_for_type += fold
    all_files_for_type += data['test']
    all_files_for_type = set(all_files_for_type)
    # filtered 
    filenames = [f for f in filenames if f in all_files_for_type]
    
    #print split size 
    for i, splitx in enumerate(train):
        print(f"Split {i}: {len([f for f in splitx if f in all_files_for_type])}")
    # filter files not ending with .pt
    filenames = [f for f in filenames if f.endswith('.pt') and f !='efa9ace68e487ddd29c2b4d6dd23242158f1f607_104293824315803097735386120227292768607_0.cpg.pt']
    if pretrain_cwe:
        cwe_info = pickle.load(open('codegraphs/diversevul/diversevul_graph_info.pkl','rb'))
        dict_ = {}
        no_cwes = set()
        all_cwes = set()
        for d in cwe_info:
            file = d['commit_id'] +'_'+str(d['hash'])+'_'+str(d['target'])+'.cpg.pt'
            dict_[file] = d['cwe']
            if len(d['cwe']) == 0:
                no_cwes.add(file)
            for cwe in d['cwe']:
                all_cwes.add(cwe)
        n_cwes = len(all_cwes)
        print(f"Number of CWEs: {n_cwes}")
        # onehot encode cwes, sort them first
        
        if not os.path.exists('codegraphs/diversevul/diversevul_cwe_label.pt'):
            assert n_cwes == 150
            all_cwes = sorted(list(all_cwes))
            eye = torch.eye(150)  # n cwes
            onehot = {cwe: eye[i] for i,cwe in enumerate(all_cwes)}
            torch.save(onehot, 'codegraphs/diversevul/diversevul_cwe_label.pt')
        else:
            onehot = torch.load('codegraphs/diversevul/diversevul_cwe_label.pt')
            
        final_dict = {}
        for k,v in dict_.items():
            # overlap cwes
            entry = torch.zeros(n_cwes)
            for cwe in v:
                entry += onehot[cwe]
            final_dict[k] = entry.unsqueeze(0)
      
        filenames = [f for f in filenames if f not in no_cwes]
    else:
        final_dict = None
        
    if pairs_only:
        # vuln non vuln pair functions, so no "peripheral" functions
        pos, neg = pickle.load(open('codegraphs/diversevul/positive_negative_files_approxdist.pkl', 'rb'))
        all = set(pos+neg)
        filenames = [f for f in filenames if f in all]
    
    return filenames, final_dict


                

    
        
class CodeGraphDataset(Dataset):
    def __init__(self, pt_folder, DS_type='all', split='train', is_heterogeneous=False, transform=None, pre_transform=None, pre_filter=None, device='cuda', cross_val_valfold_idx=None, is_cross_val=False, cross_val_train_fraction=1, cross_val_val_fraction=1, remove_degreeandtriangles=False, dense_mode=False, pretrain_cwe=False, pairs_only=False):
        '''
            DS_type:
                'all': 'train_test_folds_v1_allgraphs.pkl',
                'larger10smaller1000': 'train_test_folds_v1_allgraphs_filtered_smaller10_larger1000.pkl',
                'larger10': 'train_test_folds_v1_allgraphs_filtered_smaller10.pkl',
                'smaller1000': 'train_test_folds_v1_allgraphs_filtered_larger1000.pkl'
        '''
        # self.dense_mode = dense_mode
        root = Path(pt_folder).parent
        self.pt_folder = Path(pt_folder)
        self.dense_mode = dense_mode
        self.remove_degreeandtriangles = remove_degreeandtriangles
        # type: preprocesed (those from erik), preprocessed_with_degreecount: those with degree count 
        self.pretrain_cwe = pretrain_cwe
        
        if cross_val_valfold_idx is None:
            cross_val_valfold_idx = 0
            
        self.filenames, self.cwe_info = get_filenames(split, cross_val_val_fraction, cross_val_train_fraction, cross_val_valfold_idx, self.pt_folder, DS_type, pretrain_cwe, pairs_only)

        
        self.config = {'device': device}
        
        self.is_heterogeneous = is_heterogeneous
        if self.is_heterogeneous:
            self.file = 'data_heterogeneous.pt'
        else:
            self.file = f'data_{split}.pt'
            
        # self.load(Path(self.processed_dir)/self.file)
        
                
        super().__init__(root, transform, pre_transform, pre_filter)
        
        
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])
        # self.cached_tensors = {}
        # self.cache_if_possible = self.cache_if_possible_true
        

    @property
    def raw_file_names(self):
        return self.filenames

    @property
    def processed_file_names(self):
        return self.filenames
    
    def download(self):
        # Download to `self.raw_dir`.
        pass
    
    def len(self):
        return len(self.filenames)
    

    def process(self):
        pass 
    
    
    def get(self,idx):
        file_name = self.filenames[idx]
        
        file_path = self.pt_folder / file_name
        data = torch.load(file_path)
        
        # remove degree and triangle counts
        # is is instance of hetero object:
        if self.remove_degreeandtriangles:
            if isinstance(data, HeteroData):
                for key in data.node_types:
                    data[key].x = data[key].x[:,18:]
            else:
                data.x = data.x[:,18:]
                
        if self.dense_mode:
            data.numnodes = torch.tensor([data.x.shape[0]])
            data.adj = to_dense_adj(data.edge_index)
            data.edge_index = None
            data.adj = pad(data.adj, (0,1000-data.adj.shape[1],0,1000-data.adj.shape[2]), value=0)
            data.x = pad(data.x, (0,0,0,1000-data.x.shape[0]), value=0).unsqueeze(0)
        
        if self.pretrain_cwe:
            data.y = self.cwe_info[file_name]
            
        data.file_name = file_name
            
        return data
        
        
        
class CodeGraphDataset_InMemory(InMemoryDataset):
    def __init__(self, pt_folder, DS_type='all', split='train', is_heterogeneous=False, transform=None, pre_transform=None, pre_filter=None, device='cuda', cross_val_valfold_idx=None, is_cross_val=False, cross_val_train_fraction=1, cross_val_val_fraction=1, remove_degreeandtriangles=False, dense_mode=False):
        '''
            DS_type:
                'all': 'train_test_folds_v1_allgraphs.pkl',
                'larger10smaller1000': 'train_test_folds_v1_allgraphs_filtered_smaller10_larger1000.pkl',
                'larger10': 'train_test_folds_v1_allgraphs_filtered_smaller10.pkl',
                'smaller1000': 'train_test_folds_v1_allgraphs_filtered_larger1000.pkl'
        '''
        self.remove_degreeandtriangles = remove_degreeandtriangles
        self.pt_folder = Path(pt_folder)
        if dense_mode:
            raise Exception('Use Non-inmemory data loader')
        root = Path(pt_folder).parent
        # type: preprocesed (those from erik), preprocessed_with_degreecount: those with degree count 
        
        
        self.filenames, self.cwe_info = get_filenames(split, cross_val_val_fraction, cross_val_train_fraction, cross_val_valfold_idx, pt_folder, DS_type)

        assert len(self.filenames) < 200000, f"too many files would be loaded (GPU RAM usage): {len(self.filenames)} files"
        
        self.config = {'device': device}
        
        self.is_heterogeneous = is_heterogeneous
        if self.is_heterogeneous:
            self.file = 'data_heterogeneous.pt'
        else:
            self.file = f'data_{split}.pt'
            
        
        super().__init__(root, transform, pre_transform, pre_filter)
        print(self.pt_folder / self.filenames[0])
        
        
        try:
            data = [torch.load(self.pt_folder / filename) for filename in tqdm(self.filenames)]
        except:
            data = []
            for file in self.filenames:
                try:
                    data.append(torch.load(self.pt_folder / file))
                except:
                    print(f"Could not load {file}")
                    
        
        
        if self.remove_degreeandtriangles:
            temp_data =[]
            if isinstance(data[0], HeteroData):
                for d in data:
                    for key in d.node_types:
                        d[key].x = d[key].x[:,18:]
                    temp_data.append(d)
                    
            else:
                for d in data:
                    d.x = d.x[:,18:]
                    temp_data.append(d)
            data = temp_data
            
        
        self.loading(data)   
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        return self.filenames

    @property
    def processed_file_names(self):
        return self.filenames
    
    def download(self):
        # Download to `self.raw_dir`.
        pass
    
    def len(self):
        return len(self.filenames)
    

    def process(self):
        pass 
    
    
    def loading(self,data_list) -> None:
        r"""Saves a list of data objects to the file path :obj:`path`."""
        
        out = InMemoryDataset.collate(data_list)
        assert isinstance(out, tuple)
        assert len(out) == 2 or len(out) == 3
        if len(out) == 2:  # Backward compatibility.
            data, self.slices = out
        else:
            data, self.slices, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)
        


if __name__ == '__main__':
    i =2
    dataset4 = CodeGraphDataset(pt_folder='codegraphs/diversevul/v2_directed_withdegreecount', split='train', 
                                        cross_val_valfold_idx=i, 
                                        is_cross_val=True, cross_val_train_fraction=0.002, DS_type='all', dense_mode=True
                                        , pretrain_cwe=True)
                                        
    from torch_geometric.loader import DataLoader
  
    
    train_dataloader = DataLoader(
            dataset4,  # .to(device)
            batch_size=4, 
            shuffle=True,
            pin_memory=True,
            num_workers=0
            )
    
    datas = []
    for i in range(10):
        for data in tqdm(train_dataloader):
            data = data.to('cuda')
            print(data.file_name)
            pass
     
    
    while True:
        pass