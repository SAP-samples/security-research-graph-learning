# %%
import pickle 
import os 

graphinfo = pickle.load(open("codegraphs/diversevul/diversevul_graph_info.pkl", "rb"))

# %%
commitcounts = {}
for d in graphinfo:
    if d['commit_id'] not in commitcounts:
        commitcounts[d['commit_id']] = 0
    commitcounts[d['commit_id']] += 1

# %%
# !pip install python-Levenshtein

# # %%
# import Levenshtein as lev

# global_store = {}
# def load_codegraph(commit_id, hash, target):
#     filename = str(commit_id)+'_'+str(hash)+'_'+str(target)+'.c'
#     filename = 'codegraphs/diversevul/c_files/' +filename
#     # return text file text
#     if filename not in global_store:
#         global_store[filename] = open(filename, 'r').read()
#     return global_store[filename]

# identify all positive files
# for each positive file find the corresponding negative file with smalles levenstein distance

# positive_files = []
# negative_files = []

# from tqdm.auto import tqdm
# for d in tqdm(graphinfo):
#     filename = str(d['commit_id'])+'_'+str(d['hash'])+'_'+str(d['target'])+'.cpg.pt'
#     if d['target'] != 1:
#         continue 
    
    
#     positive_files.append(filename)
#     # find corresponding negative file
    
#     file1 = load_codegraph(d['commit_id'], d['hash'], d['target'])
    
    
    
#     # print(file1)
#     min_dist = float('inf')
#     for d2 in graphinfo:
#         if d2['commit_id'] == d['commit_id'] and d2['project'] == d['project'] and d2['hash'] != d['hash']:
        
#             file2 = load_codegraph(d2['commit_id'], d2['hash'], d2['target'])
#             if file2.startswith(file1[:10]) and d2['target']==0:
#                 dist = lev.distance(file1, file2)
#                 if dist < min_dist:
#                     min_dist = dist
#                     min_file = str(d2['commit_id'])+'_'+str(d2['hash'])+'_'+str(d2['target'])+'.cpg.pt'
#                     min_file2 = file2
                    
#         negative_files.append(min_file)
    
        

# %%
import Levenshtein as lev
from multiprocessing import Pool
from tqdm.auto import tqdm
import os

def load_codegraph(args):
    global_store, commit_id, hash, target = args
    filename = 'codegraphs/diversevul/c_files/' + str(commit_id) + '_' + str(hash)+ '_' + str(target) + '.c'
    if filename not in global_store:
        with open(filename, 'r') as f:
            global_store[filename] = f.read()
    return global_store[filename]

def find_min_dist(args):
    global_store, d, file1 = args
    min_dist = float('inf')
    min_file = None
    for d2 in graphinfo:
        if d2['commit_id'] == d['commit_id'] and d2['project'] == d['project'] and d2['hash'] != d['hash']:
            file2 = load_codegraph((global_store, d2['commit_id'], d2['hash'], d2['target']))
            if  file1[:10] in file2 and d2['target']==0:
                dist = lev.distance(file1, file2)
                if dist < min_dist:
                    min_dist = dist
                    min_file = str(d2['commit_id']) + '_' + str(d2['hash']) + '_' + str(d2['target']) + '.cpg.pt'
                    min_file2 = file2
    return min_file

positive_files = []
negative_files = []
global_store = {}

pool = Pool(os.cpu_count())
for d in tqdm(graphinfo):
    if d['target'] != 1: continue
    filename = str(d['commit_id']) + '_' + str(d['hash']) + '_' + str(d['target'])+'.cpg.pt'
    positive_files.append(filename)
    file1 = pool.apply_async(load_codegraph, ((global_store, d['commit_id'], d['hash'], d['target']),)).get()
    min_file = pool.apply_async(find_min_dist, ((global_store, d, file1),)).get()
    if min_file: negative_files.append(min_file)
    
pool.close() 
pool.join()

# save to file in codegraph diversevul folder
pickle.dump((positive_files, negative_files), open('codegraphs/diversevul/positive_negative_files_approxdist.pkl', 'wb'))



