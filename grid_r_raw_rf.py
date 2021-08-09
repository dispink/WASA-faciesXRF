import numpy as np 
import pandas as pd
from wasafacies import PrepareData, split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

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

print('Begin: RF the reclassified raw element data')

pipe = Pipeline([('scaling', StandardScaler()),('pca', PCA(whiten=True)), ('rf', RandomForestClassifier(class_weight='balanced', random_state=24, n_jobs=-1))])

param_grid = [
    {'pca': [PCA(whiten=True)], 
     'rf__n_estimators':[100, 1000, 5000],
     'rf__max_depth': [5, 10, 15]},
    {'scaling': [None],
     'pca': [None],
     'rf__n_estimators':[100, 1000, 5000],
     'rf__max_depth': [5, 10, 15]}
]

mycv = split.OnegrupOnefacies_cv(y[train_idx], groups[train_idx], n_splits = 5, random_state = 24)

grid = GridSearchCV(pipe, param_grid = param_grid, cv = mycv, scoring = 'balanced_accuracy', n_jobs = -1)

grid.fit(X[train_idx], y[train_idx])

print("Best CV score: {:.2f}".format(grid.best_score_)) 
print("Best parameters: ", grid.best_params_)

pd.DataFrame(grid.cv_results_).to_csv('{}results/r_raw_rf_grid_{}.csv'.format(path, date))

from joblib import dump, load
dump(grid.best_estimator_, '{}models/r_raw_rf_model_{}.joblib'.format(path, date)) 

print("The computation takes {} hours.\n".format((perf_counter() - start)/3600))
