import os
import numpy as np
import pandas as pd

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def get_directories_and_subdirectories_containing_a_file_eventsouttfevents(dpath):
    all_dirs = []
    for root, dirs, files in os.walk(dpath):
        for file in files:
            if file.startswith('events.out.tfevents'):
                all_dirs.append(root)
                break
            
    return all_dirs

from tqdm.auto import tqdm

def tabulate_events(dpath):

    final_out = {}
    for dpath in tqdm(get_directories_and_subdirectories_containing_a_file_eventsouttfevents(dpath)):
        for dname in os.listdir(dpath):
            if not dname.startswith('events.out.tfevents'):
                continue
            print(f"Converting run {dname}",end="")
            ea = EventAccumulator(os.path.join(dpath, dname)).Reload()
            tags = ea.Tags()['scalars']

            out = {}

            for tag in tags:
                tag_values=[]
                wall_time=[]
                steps=[]

                for event in ea.Scalars(tag):
                    tag_values.append(event.value)
                    wall_time.append(event.wall_time)
                    steps.append(event.step)

                out[tag]=pd.DataFrame(data=dict(zip(steps,np.array([tag_values,wall_time]).transpose())), columns=steps,index=['value','wall_time'])

            # if 'models' in dpath:
            dname = dpath.split('/')[-2] +'_'+ dpath.split('/')[-1] 
            # else:
                
            if len(tags)>0:      
                df= pd.concat(out.values(),keys=out.keys())
                try:
                    df.to_csv(f'csvs/{dname}.csv')
                except BaseException as e:
                    print(e)
                    pass
                print("- Done")
            else:
                print('- Not scalers to write')
            
            # print(dname)
            # final_out[dname] = df


    return final_out
if __name__ == '__main__':
    path = "runs/"
    steps = tabulate_events(path)
    # pd.concat(steps.values(),keys=steps.keys()).to_csv('all_result.csv')