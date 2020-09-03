# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:51:48 2019

@author: An-Sheng
This script is the workflow that aims to minimize the memory usage and preocess time of constructing a spectra dataset.
Spectrum is the raw reading output from Itrax core scanner. It will be needed for reprocessing by Q-Spec.
The script follows the idea of 
https://www.dataquest.io/blog/pandas-big-data/
https://github.com/pandas-dev/pandas/issues/6683
"""


################## Import Packages #################### 
import time
import os
import numpy as np
import pandas as pd
####################################################### 


###### Set the working dorectory ######
path = '~\\GeopolarLabor\\#Projekte\\WASA\\XRF\\data_composite'
os.chdir(path)

date = time.strftime('%Y%m%d', time.localtime())
start = time.time()
#######################################


##### read in tables ####
dtypes = {'new_spe_dir': 'str', 'core_ID': 'category'}      # to reduce the memory usage
result_clean_df = pd.read_csv('WASA_cleaned_result.csv', 
                              usecols = ['new_spe_dir', 'core_ID'],
                              dtype = dtypes)

#########################

content = []
length = len(result_clean_df.new_spe_dir)

for row, spe in enumerate(result_clean_df.new_spe_dir):
    
    with open(spe, 'r') as f:
        lines = f.readlines()[38:]
        
        for col, txt in enumerate(lines):
            content.append( int( txt.split('\t')[1].split('\n')[0] ))                           
           
    if row % (length/100) == 0:
        print('Progress: {:03.1f}%'.format(100 * row / length))


body = np.array(content).reshape((length, 1024))

spe_df = pd.DataFrame(data = body, columns = [str(_) for _ in range(1, 1025)])
spe_df = spe_df.apply(pd.to_numeric, axis = 'columns', downcast = 'unsigned')

spe_df['spe_dir'] = result_clean_df.new_spe_dir
spe_df['core_ID'] = result_clean_df.core_ID
        
spe_df.to_csv('WASA_all_xrf_spe_{}.csv'.format(date), index = False)

end = time.time()
dur = (end-start)/60
print('The process takes {} minutes'.format(dur))
# 28.9 minutes, 412.6 MB memory usage