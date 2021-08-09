import numpy as np 
import pandas as pd
from wasafacies import PrepareData, split

import datetime
date = datetime.datetime.now().strftime('%Y%m%d')

from time import perf_counter
start = perf_counter()

path = '/home/users/aslee/WASA_faciesXRF/'

prepare = PrepareData(data_dir='{}data/XRF_results.cleaned.all.csv'.format(path),
                      info_dir='{}data/info.cleaned.all.csv'.format(path), 
                      recla_dir='{}data/new facies types 20210728.xlsx'.format(path))

facies, id_list = prepare.create_recla()
data_df = prepare.create_raw(facies=facies, id_list=id_list)

X = data_df.iloc[:, :-2].values
y, uniques = pd.factorize(data_df['facies'])
groups = data_df['core_section'].values

train_idx, test_idx = split.train_test_split(y, groups)
# This time I split the training set again to obtain dev set
#trainn_idx, dev_idx = my_train_test_split(y[train_idx], groups[train_idx])
#X_trainn = X[train_idx[trainn_idx]]
#X_dev = X[train_idx[dev_idx]]
#y_trainn = y[train_idx[trainn_idx]]
#y_dev = y[train_idx[dev_idx]]
#groups_trainn = groups[train_idx[trainn_idx]]
#del X, y, groups

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

print('Begin: LR the reclassified raw element data')

pipe = Pipeline([
    ('scaling', StandardScaler()),
    ('pca', PCA()), 
    ('lr', LogisticRegression(max_iter = 10000, class_weight='balanced'))])

param_grid = [
    {'pca': [PCA(whiten=True)],
     'lr__C': np.logspace(-4, 5, 10)},
    {'pca': [None],
     'lr__C': np.logspace(-4, 5, 10)}
]

mycv = split.OnegrupOnefacies_cv(y[train_idx], groups[train_idx], n_splits = 5, random_state = 24)

grid = GridSearchCV(pipe, param_grid = param_grid, cv = mycv, scoring = 'balanced_accuracy', n_jobs = -1)

grid.fit(X[train_idx], y[train_idx])

print("Best CV score: {:.2f}".format(grid.best_score_)) 
print("Best parameters: ", grid.best_params_)

pd.DataFrame(grid.cv_results_).to_csv('{}results/r_raw_lr_grid_{}.csv'.format(path, date))

from joblib import dump, load
dump(grid.best_estimator_, '{}models/r_raw_lr_model_{}.joblib'.format(path, date)) 

print("The computation takes {} hours.\n".format((perf_counter() - start)/3600))
