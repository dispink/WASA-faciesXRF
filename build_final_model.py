"""
This script is used to build final maodel based on the results of grid search and error analysis.
"""
import pandas as pd
from split import my_train_test_split

import datetime
date = datetime.datetime.now().strftime('%Y%m%d')

from time import perf_counter
start = perf_counter()

path = '/home/users/aslee/WASA_faciesXRF/'
print('Build the optimal model (SVC+PCA) on training+dev set')
print('It has the calibrated probability')

data_df = pd.read_csv('{}data/XRF_ML_cr.csv'.format(path))
X = data_df.iloc[:, 1:-2].values
y = data_df['facies_merge_2'].values
groups = data_df['core_section'].values

train_idx, test_idx = my_train_test_split(y, groups)
X_train = X[train_idx]
y_train = y[train_idx]
groups_train = groups[train_idx]

del data_df, X, y, groups

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

pipe = make_pipeline(
    StandardScaler(), 
    PCA(whiten=True), 
    SVC(C=100, gamma=1e-5, class_weight='balanced', probability=True))

pipe.fit(X_train, y_train)

from joblib import dump
dump(pipe, '{}models/roll_svc_train+dev_model_{}.joblib'.format(path, date)) 

print("The computation takes {} hours.".format((perf_counter() - start)/3600))