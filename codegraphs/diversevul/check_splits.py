import pickle as pkl
data = pkl.load(open('diversevul_graph_info.pkl','rb'))
import os 


existing_data = []
for d in data:
    name = d['commit_id'] +"_"+str(d['hash'])+'_'+str(d['target'])+'.cpg.pt'
    if os.path.exists('preprocessed_with_degreecount/'+name):
        existing_data.append(d)
    # else:
        # print(d['path'])


from tqdm.auto import tqdm
import networkx as nx
import concurrent.futures
import os
import pygraphviz as pgv



def check_num_nodes(file):
    # G = nx.drawing.nx_agraph.read_dot('raw/'+file)
    try:
        with open('raw/'+file, "r", encoding="utf-8") as f:
            G = nx.Graph(pgv.AGraph(f.read()))
        
        if G.number_of_nodes() > 9 and G.number_of_nodes() < 1001:
            return file
        elif G.number_of_nodes() < 10:
            return 'smaller'
        elif G.number_of_nodes() > 1000:
            return 'larger'
    except Exception as e:
        print(e)
        return None

files = os.listdir('raw')

# with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
results = []
smaller_files, larger_files = 0,0
for filename in tqdm(files):
    file = check_num_nodes(filename)
    if file is not None and file != 'smaller' and file != 'larger':
        results.append(file)
    elif file == 'smaller':
        smaller_files += 1
        print('smaller:',smaller_files,'larger:',larger_files, len(results), )
    elif file == 'larger':
        larger_files += 1
        print('smaller:',smaller_files,'larger:',larger_files, len(results), )

    


correct_num_nodes = [result for result in results if result is not None]
print(len(correct_num_nodes))


# 1it/s]smaller: 77867 larger: 3777 248118
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 329783/329783 [1:13:45<00:00, 74.52it/s]
# 248119