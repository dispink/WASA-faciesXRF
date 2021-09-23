import pandas as pd
from split import my_train_test_split

import datetime
date = datetime.datetime.now().strftime('%Y%m%d')

from time import perf_counter
start = perf_counter()

path = '/home/users/aslee/WASA_faciesXRF/'
print('Build the optimal model (SVC+PCA) on training set')
print('It has the calibrated probability')

data_df = pd.read_csv('{}data/XRF_ML_cr.csv'.format(path))
X = data_df.iloc[:, 1:-2].values
y = data_df['facies_merge_2'].values
groups = data_df['core_section'].values

train_idx, test_idx = my_train_test_split(y, groups)
# This time I split the training set again to obtain dev set
trainn_idx, dev_idx = my_train_test_split(y[train_idx], groups[train_idx])
X_trainn = X[train_idx[trainn_idx]]
y_trainn = y[train_idx[trainn_idx]]

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

pipe = make_pipeline(
    StandardScaler(), 
    PCA(whiten=True), 
    SVC(C=100, gamma=1e-5, class_weight='balanced', probability=True))

pipe.fit(X_trainn, y_trainn)

from joblib import dump
dump(pipe, '{}models/roll_svc_trainn_model_{}.joblib'.format(path, date)) 

print("The computation takes {} hours.".format((perf_counter() - start)/3600))