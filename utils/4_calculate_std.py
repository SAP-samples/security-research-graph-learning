import os 
import yaml 
import statistics
from pprint import pprint
from copy import deepcopy 

def calculate_std(numbers):
    return statistics.stdev(numbers)




def pprintx(file):
    print('=====================')
    print(file)
    print('=====================')
    with open(file, "r") as f:
        data = yaml.safe_load(f)
    
    new_data = deepcopy(data)
    for key in data[0]:
        if isinstance(data[0][key], list):
            new_data[0][f"{key}_std"] = calculate_std(data[0][key])

    pprint(new_data)



pprintx('runs/20240312_211522_CV_01_Final_BareboneGCN_smaller1000_withoutdeg/all_configs.yml')
