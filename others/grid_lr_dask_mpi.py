import numpy as np 
import pandas as pd
from split import my_train_test_split

from dask_mpi import initialize
initialize()

from dask.distributed import Client
client = Client()

import datetime
date = datetime.datetime.now().strftime('%Y%m%d')

from time import perf_counter
start = perf_counter()

path = '/home/users/aslee/WASA_faciesXRF/'
with open('{}/job_logs/tmp.txt'.format(path), 'w+') as f:
    print('Begin', file=f)

data_df = pd.read_csv('{}data/XRF_ML_cr.csv'.format(path))
X = data_df.iloc[:, 1:-2].values
y = data_df['facies_merge_1'].values
groups = data_df['core_section'].values

train_idx, test_idx = my_train_test_split(X, y, groups)
X_train = X[train_idx]
y_train = y[train_idx]
groups_train = groups[train_idx]

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from dask_ml.model_selection import GridSearchCV
from split import my_group_stratify_shuffle_cv

with open('{}/job_logs/tmp.txt'.format(path), 'a') as f:
    print('With PCA', file=f)

lr = make_pipeline(StandardScaler(), PCA(whiten = True), LogisticRegression(max_iter = 10000, class_weight = 'balanced'))

param_grid = {'logisticregression__C': [10**_ for _ in range(-4, 6)]}

mycv = my_group_stratify_shuffle_cv(X_train, y_train, groups_train)

grid = GridSearchCV(lr, param_grid = param_grid, cv = mycv, scoring = 'balanced_accuracy', n_jobs = -1)

grid.fit(X_train, y_train)

print("Best score on validation set: {:.2f}".format(grid.best_score_)) 
print("Best parameters: ", grid.best_params_)

lr_pca_df = pd.DataFrame(grid.cv_results_)
lr_pca_df.to_csv('{}results/roll_pca+lr_grid_s_{}.csv'.format(path, date))

from joblib import dump, load
dump(grid.best_estimator_, '{}models/roll_pca+lr_model_s_{}.joblib'.format(path, date)) 

########################
with open('{}/job_logs/tmp.txt'.format(path), 'a') as f:
    print("The half computation takes {} hours.".format((perf_counter() - start)/3600), file=f)
    print('Without PCA', file=f)

lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter = 10000, class_weight = 'balanced'))

param_grid = {'logisticregression__C': [10**_ for _ in range(-4, 6)]}

mycv = my_group_stratify_shuffle_cv(X_train, y_train, groups_train)

grid = GridSearchCV(lr, param_grid = param_grid, cv = mycv, scoring = 'balanced_accuracy', n_jobs = -1)

grid.fit(X_train, y_train)

print("Best score on validation set: {:.2f}".format(grid.best_score_)) 
print("Best parameters: ", grid.best_params_)

lr_df = pd.DataFrame(grid.cv_results_)
lr_df.to_csv('{}results/roll_lr_grid_s_{}.csv'.format(path, date))

dump(grid.best_estimator_, '{}models/roll_lr_model_s_{}.joblib'.format(path, date)) 

with open('{}/job_logs/tmp.txt'.format(path), 'a') as f:
    print("The computation takes {} hours.".format((perf_counter() - start)/3600), file=f)
