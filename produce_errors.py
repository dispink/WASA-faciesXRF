'''
Produce the errors in both the training and dev sets. 
I use the best model in each type of data representation as the model.
They are all SVM.
'''
import numpy as np
import pandas as pd
from split import my_train_test_split
from create_2d_data import *
from joblib import load
from sklearn.metrics import balanced_accuracy_score as score

import datetime
date = datetime.datetime.now().strftime('%Y%m%d')
path = '/home/users/aslee/WASA_faciesXRF/'
print('Produce errors in both training and dev sets.')

data_list = ['Raw', 'Rolling', 'Image-like']
trainn_error_list = [] # so called bias
dev_error_list = []

####### Raw #######
data_df = pd.read_csv('{}data/XRF_ML_c.csv'.format(path))
X = data_df.iloc[:, 1:-2].values
y = data_df['facies_merge_2'].values
groups = data_df['core_section'].values
del data_df

train_idx, test_idx = my_train_test_split(y, groups)
trainn_idx, dev_idx = my_train_test_split(y[train_idx], groups[train_idx])

model = load('{}models/raw_svc_model_20210519.joblib'.format(path))
trainn_error_list.append(
    1 - score(
        y_true=y[train_idx[trainn_idx]], y_pred=model.predict(X[train_idx[trainn_idx]])
    )
)
dev_error_list.append(
    1 - score(
        y_true=y[train_idx[dev_idx]], y_pred=model.predict(X[train_idx[dev_idx]])
    )
)

####### Rolling #######
data_df = pd.read_csv('{}data/XRF_ML_cr.csv'.format(path))
X = data_df.iloc[:, 1:-2].values
y = data_df['facies_merge_2'].values
groups = data_df['core_section'].values
del data_df

train_idx, test_idx = my_train_test_split(y, groups)
trainn_idx, dev_idx = my_train_test_split(y[train_idx], groups[train_idx])

model = load('{}models/roll_svc_model_20210524.joblib'.format(path))
#roll_svc_trainn_model_20210511.joblib it's the one use both training and dev sets to run CV
trainn_error_list.append(
    1 - score(
        y_true=y[train_idx[trainn_idx]], y_pred=model.predict(X[train_idx[trainn_idx]])
    )
)
dev_error_list.append(
    1 - score(
        y_true=y[train_idx[dev_idx]], y_pred=model.predict(X[train_idx[dev_idx]])
    )
)

####### Image-like #######
X, y, groups = create_2d('{}data/XRF_ML.csv'.format(path))

train_idx, test_idx = my_train_test_split(y, groups)
trainn_idx, dev_idx = my_train_test_split(y[train_idx], groups[train_idx])

model = load('{}models/2d_svc_model_20210521.joblib'.format(path))
trainn_error_list.append(
    1 - score(
        y_true=y[train_idx[trainn_idx]], y_pred=model.predict(X[train_idx[trainn_idx]])
    )
)
dev_error_list.append(
    1 - score(
        y_true=y[train_idx[dev_idx]], y_pred=model.predict(X[train_idx[dev_idx]])
    )
)

error_df = pd.DataFrame({
    'representation': data_list,
    'train_error': trainn_error_list,
    'dev_error': dev_error_list
})

error_df.to_csv('{}results/errors_{}.csv'.format(path, date))