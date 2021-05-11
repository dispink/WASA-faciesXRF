import numpy as np 
import pandas as pd
from split import *

import datetime
date = datetime.datetime.now().strftime('%Y%m%d')

from time import perf_counter
start = perf_counter()

path = '/home/users/aslee/WASA_faciesXRF/'
print('Begin')

data_df = pd.read_csv('{}data/XRF_ML_cr.csv'.format(path))
X = data_df.iloc[:, 1:-2].values
y = data_df['facies_merge_2'].values
groups = data_df['core_section'].values

train_idx, test_idx = my_train_test_split(y, groups)
X_train = X[train_idx]
y_train = y[train_idx]
groups_train = groups[train_idx]

# This time I split the training set again to obtain dev set
trainn_idx, dev_idx = my_group_stratify_shuffle(X_train, y_train, groups_train)
X_trainn = X[train_idx[trainn_idx]]
X_dev = X[train_idx[dev_idx]]
y_trainn = y[train_idx[trainn_idx]]
y_dev = y[train_idx[dev_idx]]

del data_df, X, y, groups

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = make_pipeline(StandardScaler(), SVC(C=1, gamma=0.01, class_weight='balanced'))

pipe.fit(X_trainn, y_trainn)

from joblib import dump, load
dump(pipe, '{}models/roll_svc_trainn_model_{}.joblib'.format(path, date)) 

print("The computation takes {} hours.".format((perf_counter() - start)/3600))