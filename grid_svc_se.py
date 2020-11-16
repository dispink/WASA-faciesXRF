import numpy as np 
import pandas as pd
from split import my_train_test_split

import datetime
date = datetime.datetime.now().strftime('%Y%m%d')

from time import perf_counter
start = perf_counter()

path = '/home/users/aslee/WASA_faciesXRF/'

print('Begin')

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
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from split import my_group_stratify_shuffle_cv

print('SVC with PCA')

pipe = make_pipeline(StandardScaler(), PCA(whiten = True), SVC(class_weight = 'balanced'))
param_grid = {'svc__C': np.logspace(-3, 5, 9),
             'svc__gamma': np.logspace(-6, 0, 7)}


# use my customized cv from ML_imporvement_workflow_09.ipynb
mycv = my_group_stratify_shuffle_cv(X, y, groups)

# reuturn_train_score is set false to save computation expense
grid = GridSearchCV(pipe, param_grid = param_grid, cv = mycv, scoring = 'balanced_accuracy', 
	n_jobs = -1, return_train_score = False)

grid.fit(X, y)

print("Best score on validation set: {:.2f}".format(grid.best_score_)) 
print("Best parameters: ", grid.best_params_)

pd.DataFrame(grid.cv_results_).to_csv('{}results/roll_pca+svc_grid_se_{}.csv'.format(path, date))

from joblib import dump, load
dump(grid.best_estimator_, '{}models/roll_pca+svc_model_se_{}.joblib'.format(path, date)) 

########################
print("The half computation takes {} hours.".format((perf_counter() - start)/3600))
print('SVC without PCA')

pipe = make_pipeline(StandardScaler(), SVC(class_weight = 'balanced'))

mycv = my_group_stratify_shuffle_cv(X, y, groups)

grid = GridSearchCV(pipe, param_grid = param_grid, cv = mycv, scoring = 'balanced_accuracy', 
                    n_jobs = -1, return_train_score = False)

grid.fit(X_train, y_train)

print("Best score on validation set: {:.2f}".format(grid.best_score_)) 
print("Best parameters: ", grid.best_params_)

pd.DataFrame(grid.cv_results_).to_csv('{}results/roll_svc_grid_se_{}.csv'.format(path, date))

dump(grid.best_estimator_, '{}models/roll_svc_model_se_{}.joblib'.format(path, date)) 

print("The computation takes {} hours.".format((perf_counter() - start)/3600))