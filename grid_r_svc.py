import numpy as np 
import pandas as pd
from wasafacies import PrepareData, Split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
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
data_df = prepare.create_roll(facies=facies, id_list=id_list)
X = data_df.iloc[:, :-2].values
y = data_df['facies'].values
groups = data_df['core_section'].values

train_idx, test_idx = Split.train_test_split(y, groups)

print('Begin: SVC the reclassified rolling element data')

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

mycv = Split.OnegrupOnefacies_cv(y[train_idx], groups[train_idx], n_splits = 5, random_state = 24)
grid = GridSearchCV(pipe, param_grid = param_grid, cv = mycv, scoring = 'balanced_accuracy', n_jobs = -1)
grid.fit(X[train_idx], y[train_idx])

print("Best CV score: {:.3f}".format(grid.best_score_)) 
print("Best parameters: ", grid.best_params_)

pd.DataFrame(grid.cv_results_).to_csv('{}results/r_roll_svc_grid_{}.csv'.format(path, date))

from joblib import dump, load
dump(grid.best_estimator_, '{}models/r_roll_svc_model_{}.joblib'.format(path, date)) 

print("The computation takes {} hours.\n".format((perf_counter() - start)/3600))