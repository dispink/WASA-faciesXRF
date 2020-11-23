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
y = data_df['facies_merge_2'].values
groups = data_df['core_section'].values

train_idx, test_idx = my_train_test_split(y, groups)
X_train = X[train_idx]
y_train = y[train_idx]
groups_train = groups[train_idx]

del data_df, X, y, groups

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from split import my_group_stratify_shuffle_cv

print('Begin: SVC')

pipe = Pipeline([('scaling', StandardScaler()),('pca', PCA(whiten=True)), ('svc', SVC(class_weight='balanced'))])

param_grid = [
    {'scaling': [StandardScaler()],
     'pca': [PCA(whiten=True)],
     'svc__C': np.logspace(-3, 5, 9),
     'svc__gamma': np.logspace(-6, 0, 7)},
    {'scaling': [StandardScaler()],
     'pca': [None],
     'svc__C': np.logspace(-3, 5, 9),
     'svc__gamma': np.logspace(-6, 0, 7)}
]

mycv = my_group_stratify_shuffle_cv(X_train, y_train, groups_train)

grid = GridSearchCV(pipe, param_grid = param_grid, cv = mycv, scoring = 'balanced_accuracy', 
                    return_train_score = False, n_jobs = -1)

grid.fit(X_train, y_train)

print("Best score on validation set: {:.2f}".format(grid.best_score_)) 
print("Best parameters: ", grid.best_params_)

pca_df = pd.DataFrame(grid.cv_results_)
pca_df.to_csv('{}results/roll_svc_grid_{}.csv'.format(path, date))

from joblib import dump, load
dump(grid.best_estimator_, '{}models/roll_svc_model_{}.joblib'.format(path, date)) 

print("The computation takes {} hours.".format((perf_counter() - start)/3600))