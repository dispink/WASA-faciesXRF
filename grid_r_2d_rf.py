import numpy as np 
import pandas as pd
from time import perf_counter
from wasafacies import PrepareData, Split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from joblib import dump

import datetime
date = datetime.datetime.now().strftime('%Y%m%d')
print('The reclassified image-like element data.')

path = '/home/users/aslee/WASA_faciesXRF/'

# prepare data
prepare = PrepareData(data_dir='{}data/XRF_results.cleaned.all.csv'.format(path),
                      info_dir='{}data/info.cleaned.all.csv'.format(path), 
                      recla_dir='{}data/new facies types 20210728.xlsx'.format(path))

facies, id_list = prepare.create_recla()
X, y, groups = prepare.create_2d(facies=facies, id_list=id_list, half_window=8)

train_idx, test_idx = Split.train_test_split(y, groups)

# set the pipes and parameters
pipe= Pipeline([('scaling', StandardScaler()),
                 ('pca', PCA()), 
                 ('rf', RandomForestClassifier(class_weight='balanced', 
                                               random_state=24, 
                                               n_jobs=-1))])

param_grid = [
    {'pca': [PCA(n_components=50, whiten=True)], 
     'rf__n_estimators':[100, 1000, 10000],
     'rf__max_depth': [3, 5, 10, 15]},
    {'scaling': [None],
     'pca': [None],
     'rf__n_estimators':[100, 1000, 10000],
     'rf__max_depth': [3, 5, 10, 15]}
]

model_name = 'rf'

# grid search
start = perf_counter()
print('Begin: {}'.format(model_name.upper()))

mycv = Split.OnegrupOnefacies_cv(y[train_idx], groups[train_idx], 
                                 n_splits = 5, random_state = 24)
grid = GridSearchCV(pipe, param_grid = param_grid, cv = mycv, 
                    scoring = 'balanced_accuracy', n_jobs = -1)
grid.fit(X[train_idx], y[train_idx])

print("Best CV score: {:.3f}".format(grid.best_score_)) 
print("Best parameters: ", grid.best_params_)

pd.DataFrame(grid.cv_results_).to_csv('{}results/r_2d_{}_grid_{}.csv'.format(path, model_name, date))
dump(grid.best_estimator_, 
     '{}models/r_2d_{}_model_{}.joblib'.format(path, model_name, date)) 

print("The computation takes {:.1f} mins.\n".format((perf_counter() - start)/60))
