import numpy as np 
import pandas as pd
from split import *
from creat_2d_data import *
from sklearn.metrics import balanced_accuracy_score as score

import datetime
date = datetime.datetime.now().strftime('%Y%m%d')

from time import perf_counter
start = perf_counter()

path = '/home/users/aslee/WASA_faciesXRF/'

data_df = pd.read_csv('{}data/XRF_ML_cr.csv'.format(path))
X = data_df.iloc[:, 1:-2].values
y = data_df['facies_merge_2'].values
groups = data_df['core_section'].values

train_idx, test_idx = my_train_test_split(y, groups)
# This time I split the training set again to obtain dev set
trainn_idx, dev_idx = my_train_test_split(y[train_idx], groups[train_idx])
X_trainn = X[train_idx[trainn_idx]]
X_dev = X[train_idx[dev_idx]]
y_trainn = y[train_idx[trainn_idx]]
y_dev = y[train_idx[dev_idx]]
groups_trainn = groups[train_idx[trainn_idx]]
del X, y, groups, data_df

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

print('Begin SVC on rolling element data')

pipe = Pipeline([('scaling', StandardScaler()),('pca', PCA(whiten=True)), ('svc', SVC(class_weight='balanced'))])

param_grid = [
    {'scaling': [StandardScaler()],
     'pca': [PCA(whiten=True)],
     'svc__C': np.logspace(-3, 4, 8),
     'svc__gamma': np.logspace(-6, 0, 7)},
    {'scaling': [StandardScaler()],
     'pca': [None],
     'svc__C': np.logspace(-3, 4, 8),
     'svc__gamma': np.logspace(-6, 0, 7)}
]

mycv = my_group_stratify_shuffle_cv(X_trainn, y_trainn, groups_trainn)

grid = GridSearchCV(pipe, param_grid = param_grid, cv = mycv, scoring = 'balanced_accuracy', 
                    return_train_score = False, n_jobs = -1)

grid.fit(X_trainn, y_trainn)

print("Best CV score: {:.2f}".format(grid.best_score_)) 
print("Best parameters: ", grid.best_params_)

pd.DataFrame(grid.cv_results_).to_csv('{}results/roll_svc_grid_{}.csv'.format(path, date))

from joblib import dump, load
dump(grid.best_estimator_, '{}models/roll_svc_model_{}.joblib'.format(path, date)) 

print('Balanced score of on dev set: {:.2f}'.format(score(y_true=y_dev, y_pred=grid.best_estimator_.predict(X_dev))))
print("The computation takes {} hours.".format((perf_counter() - start)/3600))