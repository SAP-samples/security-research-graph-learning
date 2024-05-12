import torch 
import os 
from pathlib import Path
from tqdm.auto import tqdm 
import concurrent.futures

def move_to_cpu():
    if not Path('codegraphs/diversevul/moved_to_cpu').is_file():
        # for folder in ['v2_directed_withdegreecount', 'v2_directed_withdegreecount_heterogeneous', 'v2_undirected_withdegreecount', 'v2_undirected_withdegreecount_heterogeneous']:
        #     for file in tqdm(os.listdir('codegraphs/diversevul/' + folder)):
        #         data123 = torch.load('codegraphs/diversevul/' + folder + '/' + file)
        #         try:
        #             torch.save(data123.cpu(), 'codegraphs/diversevul/'+folder+'/' + file)    
        #         except AttributeError as e:
        #             print('delete file: ', 'codegraphs/diversevul/'+folder+'/' + file)
        #             os.remove('codegraphs/diversevul/'+folder+'/' + file)
                    
        def process_file(file_path):
            data123 = torch.load(file_path)
            torch.save(data123.cpu(), file_path)

        folders = [
            'v2_directed_withdegreecount',
            'v2_directed_withdegreecount_heterogeneous',
            'v2_undirected_withdegreecount',
            'v2_undirected_withdegreecount_heterogeneous'
        ]

        for folder in folders:
            folder_path = os.path.join('codegraphs/diversevul/', folder)
            files = os.listdir(folder_path)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for file in tqdm(files):
                    file_path = os.path.join(folder_path, file)
                    futures.append(executor.submit(process_file, file_path))
                for file in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    pass
     

        # make cache file in codegraphs/diversevul, that all have been moved to cpu
        Path('codegraphs/diversevul/moved_to_cpu').touch()

if __name__ == "__main__":
    move_to_cpu()