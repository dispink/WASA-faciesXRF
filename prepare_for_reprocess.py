# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:51:18 2019

@author: An-Sheng
This is a workflow to prepare the data for Q-spec reprocessing.
The spectral files are randomly copied to new foloders and the sunspectrum for each folder is created.
"""
################## Import Packages #################### 
import os
import time
import shutil
import pandas as pd
import numpy as np

path = '~\GeopolarLabor\\#Projekte\\WASA'


##### shuffle and copy all spe to 7 folder and calculate a sumspectrum for all spe #####
spe_dir = pd.read_csv(path+'\\database\\info.cleaned.composite_spe_dir.csv', squeeze = True)

## creat a random labels list ( discrete uniform distribution )
pd.DataFrame(
        {'random_labels': np.random.randint(7, size = len(spe_dir))}
        ).to_csv(
                path+'\\database\\info.cleaned.random_labels.csv', 
                index = False
                )

tdir = path+'\\XRF\\data_shuffled'
labels_csv = path+'\\database\\info.cleaned.random_labels.csv'
spe_csv = path+'\\database\\info.cleaned.composite_spe_dir.csv'

start = time.time()
if os.path.exists(tdir) == False:
    os.makedirs(tdir, 0o775)
    
###### Input data ######
info_df = pd.DataFrame(
    {'spe_dir': pd.read_csv(spe_csv, squeeze = True),
     'labels': pd.read_csv(labels_csv, squeeze = True)}
    )

##### copy the spe to cluster folders and construct sumspectrum as well #####
    
progress = 0
    
## cluster in label taable
for cluster in np.sort(info_df.labels.unique()):
    data = info_df.groupby('labels').get_group(cluster).reset_index(drop=True)
    content_sum = np.zeros((1024,), dtype = int)
        
    ## spetrum in cluster table
    # shuffle again in the cluster     
    X_list = [_ for _ in range(len(data))]
    np.random.shuffle(X_list)
    
    for row, shuffled in enumerate(X_list):
        content = []
        file = '{}_{}'.format(shuffled, data.spe_dir[row].split('\\')[-1])
        new_dir = tdir + '\\' + str(cluster)
        new_spe = tdir + '\\' + str(cluster) + '\\' + file
        
        # if the directory isn't create, create one
        if os.path.exists(new_dir) == False:   
            os.makedirs(new_dir, 0o775)
        
        shutil.copy(data.spe_dir[row], new_spe)

        ## read spe   
        with open(data.spe_dir[row], 'r') as f:
            lines = f.readlines()
            ## copy the header of the first spe as the sumspectrum's header
            if row == 0:
                header = lines[:38]
                
            ## channel content in spectrum
            for txt in lines[38:]:
                content.append( int( txt.split('\t')[1].split('\n')[0] ))     
                
        content_sum += content
        progress += 1
        
        ## display progress
        if progress % 4000 == 0:
            print('{:03.1f}% processed'.format((100 * progress / len(info_df))))
        
    ## write the sumspectrum
    sum_dir = tdir + '\\sumspectra\\C' + str(cluster) + '_sumspectrum.spe'
    
    # if the directory isn't create, create one
    if os.path.exists(tdir + '\\sumspectra') == False:   
        os.makedirs(tdir + '\\sumspectra', 0o775)
    
    ## The previous sumspectrum (if exists) need to be deleted because I use append mode for writing
    if os.path.isfile(sum_dir):
        os.remove(sum_dir)
    
    with open(sum_dir, 'a') as f_out:
        for _ in header:
            print(_, end = '', file = f_out)
        for channel, content in enumerate(content_sum):
            print('{}\t{}'.format((channel + 1), content), file = f_out)
      
## write a whole sumspectrum.spe
content_sum = []
for channel in range(1, 1025):
    content_sum.append(
            np.loadtxt('{}\\database\\XRF_spe.cleaned.{}.csv'.format(path, channel), delimiter = ',', dtype = int)
            .sum()-channel
            )

sum_dir = tdir + '\\sumspectra\\whole_sumspectrum.spe'
if os.path.isfile(sum_dir):
    os.remove(sum_dir)

with open(sum_dir, 'a') as f_out:
    for _ in header:
        print(_, end = '', file = f_out)
    for channel, content in enumerate(content_sum):
        print('{}\t{}'.format((channel + 1), content), file = f_out)
        
dur = (time.time() - start)/60
print('It takes {:0.1f} minutes to complete the preocess'.format(dur))    