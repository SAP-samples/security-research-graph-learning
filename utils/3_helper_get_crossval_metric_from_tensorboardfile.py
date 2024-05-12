import os 
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader


# file = 'runs/20240312_164104_CV_01_Final_BareboneGCN_smaller1000_withdeg/cv_3/20240312-191903_01_Final_BareboneGCN_smaller1000_withdeg__DSfsmaller1000_DSdiversevul_v1_withdeg_GNNBareboneGCN_Gdrp0.3_Gl3_GSLGSL_GSLat0.2_Sth0.002_SrgTrue_Sh4_Sskp0.4_Ssp0.05_H_Hpsum_lr0.001_wd0_batch_size128_epi1_ep1000_fdrp0_ch128_it1_pow17/events.out.tfevents.1710271143.ip-172-31-0-69.53952.3'


def get_events(file):
    tagvalues = {}
    
    loader = EventFileLoader(file)
    for event in loader.Load():
        for value in event.summary.value:
            if value.tag not in tagvalues:
                tagvalues[value.tag] = []
            # if value.tensor.dtype == "DT_FLOAT":
            tagvalues[value.tag].append(value.tensor.float_val[0])
            
    return tagvalues


import os
def main(tag,folder):
    # recursively get all files in folder which contain events.out.tfevents
    
    files = []
    for root, dirs, file in os.walk(folder):
        for f in file:
            if 'events.out.tfevents' in f:
                files.append(os.path.join(root, f))
    
    max_vals = []
    for file in files:
        tagvalues = get_events(file)
        max_vals.append((file, max(tagvalues[tag])))
        
    # get accuracy from tag=TP, TN, FP FN
    # and select maxval
    max_valsacc = []
    for file in files:
        tagvalues = get_events(file)
        # print(tagvalues)
        # max_valsacc.append((file, max(tagvalues['TP'])/(max(tagvalues['TP'])+max(tagvalues['FN']))))
    
    
    # print(files)
    # get std and mean
    # max_vals.sort(key=lambda x: x[1], reverse=True)
    # mean = sum([x[1] for x in max_vals])/len(max_vals)
    
    # print('Mean: {}'.format(mean))
    
    # sample_variance = sum([(x[1] - mean)**2 for x in max_vals])/(len(max_vals)-1)
    # std = sample_variance**0.5
    # print('Std: {}'.format(std))
    
    
    
    # max vals acc
    max_valsacc.sort(key=lambda x: x[1], reverse=True)
    mean = sum([x[1] for x in max_valsacc])/len(max_valsacc)
    
    print('Mean: {}'.format(mean))
    divisor = len(max_valsacc) - 1
    if divisor == 0:
        divisor = 1
        
    sample_variance = sum([(x[1] - mean)**2 for x in max_valsacc])/(divisor)
    std = sample_variance**0.5
    print('Std: {}'.format(std))
    

if __name__ == '__main__':
    # 'train/epoch_/f1'
    tag ='train/epoch_/f1'
    folder = 'runs/20240320_205037_CV_2_FINALV1_GraphGLOW_15iter_crossval_sl_ud/'


#     Mean: 0.6701184153556824
# Std: 0.01093289046459165

#val:
#Mean: 0.649330198764801
# Std: 0.02213338284994079 

    # folder = 'runs/20240312_211522_CV_01_Final_BareboneGCN_smaller1000_withoutdeg'
#     Mean: 0.6665405869483948
# Std: 0.006702366155793117

# val:
# Mean: 0.6468681812286377
# Std: 0.021290262534445773
    main(tag,folder)