import os
import networkx as nx 
from tqdm.auto import tqdm 

from tqdm.auto import tqdm
import networkx as nx
import concurrent.futures
import os
import pygraphviz as pgv


for file in tqdm(os.listdir('codegraphs/diversevul/raw')):
    continue_ = False 
    try:
        G1 = nx.drawing.nx_agraph.read_dot('codegraphs/diversevul/raw/'+file)
    except BaseException as err:
        print('G1',err, file)
        continue_ = True
    
    try: 
        G2 = nx.Graph(pgv.AGraph(open('codegraphs/diversevul/raw/'+file, "r", encoding="utf-8").read()))
    except BaseException as err:
        print('G2',err, file)
        continue_ = True
        
    if continue_:
        continue
    
    if not G1.nodes(data=True)._nodes == G2.nodes(data=True)._nodes:
        print('not same nodes', file)
    
    
    for edge1, edge2 in zip(G1.edges(data=True), G2.edges(data=True)):
        for a, b in zip(edge1[1], edge2[1]):
            for key1,key2 in zip(a,b):
                if key2 != key2:
                    print('not same edge attrs', file)