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
                      recla_dir='{}data/new facies types 20220120.xlsx'.format(path))

facies, id_list = prepare.create_recla()
data_df = prepare.create_raw(facies=facies, id_list=id_list)
X = data_df.iloc[:, :-2].values
y = data_df['facies'].values
groups = data_df['core_section'].values

train_idx, test_idx = Split.train_test_split(y, groups)

print('Begin: SVC the reclassified raw element data')

pipe = Pipeline([('scaling', StandardScaler()),('pca', PCA(whiten=True)), ('svc', SVC(class_weight='balanced'))])

param_grid = [
    {'pca': [PCA(whiten=True)],
     'svc__C': np.logspace(-3, 4, 8),
     'svc__gamma': np.logspace(-5, 0, 6)},
    {'pca': [None],
     'svc__C': np.logspace(-3, 4, 8),
     'svc__gamma': np.logspace(-5, 0, 6)}
]

mycv = Split.OnegrupOnefacies_cv(y[train_idx], groups[train_idx], n_splits = 5, random_state = 24)

grid = GridSearchCV(pipe, param_grid = param_grid, cv = mycv, scoring = 'balanced_accuracy', n_jobs = -1)

grid.fit(X[train_idx], y[train_idx])

print("Best CV score: {:.2f}".format(grid.best_score_)) 
print("Best parameters: ", grid.best_params_)

pca_df = pd.DataFrame(grid.cv_results_)
pca_df.to_csv('{}results/r_raw_svc_grid_{}.csv'.format(path, date))

from joblib import dump
dump(grid.best_estimator_, '{}models/r_raw_svc_model_{}.joblib'.format(path, date)) 

print("The computation takes {} mins.\n".format((perf_counter() - start)/60))