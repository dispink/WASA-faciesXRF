# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:52:19 2019

@author: An-Sheng

This script is to filter out the bad quality data points in the whole result.txt dataset.
Of course, there are many manual debug due to the inconsistecy of scanning process, but I don't
show them because I want the scipts tidy.
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


##### use result table to filter out the data points that have bad data quality #####
## read the result table
dtypes = {'validity': 'category', 'new_spe_dir': 'str', 
          'core_ID': 'category', 'core_section': 'category'}      # to reduce the memory usage
result_df = pd.read_csv('WASA_all_xrf_result_20190329.csv', 
                        usecols = ['validity', 'cps', 'Ar', 'Fe', 'new_spe_dir', 'core_ID', 'core_section'],
                        dtype = dtypes)

## criteria 1: valisity = 1 
clean_1 = result_df[result_df.validity == '1'].copy()
out_1 = result_df[~result_df.new_spe_dir.isin(clean_1.new_spe_dir)].copy()
out_1['reasons'] = ['validity = 0' for _ in range( len(out_1) ) ]

## criteria 2: Fe >= 50
clean_2 = clean_1[clean_1.Fe >= 50].copy()
out_2 = clean_1[~clean_1.new_spe_dir.isin(clean_2.new_spe_dir)].copy()
out_2['reasons'] = ['Fe < 50' for _ in range( len(out_2) ) ]

## criteria 3: ln(Ar/cps) value lower than 3 std (upper limit)
std_ArCps = 3
ArCps = np.log(clean_2.Ar / clean_2.cps)
clean_3 = (
        clean_2[np.log(clean_2.Ar / clean_2.cps) <= (ArCps.mean() + std_ArCps * ArCps.std())]
        .reset_index(drop = True)
        .copy()
        )
out_3 = clean_2[~clean_2.new_spe_dir.isin(clean_3.new_spe_dir)].copy()
out_3['reasons'] = ['ln(Ar/cps) > {} std (H.L.)'.format(std_ArCps) for _ in range( len(out_3) )]
    
## out put the results of datapoint filtering 
clean_3.to_csv('WASA_cleaned_result_{}.csv'.format(date), index = False)
out_df = out_1.append(out_2, ignore_index = True).append(out_3, ignore_index = True)
out_df.to_csv('WASA_excluded_result_{}.csv'.format(date), index = False)
