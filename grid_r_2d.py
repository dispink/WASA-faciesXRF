import numpy as np 
import pandas as pd
from time import perf_counter
from wasafacies import PrepareData, Split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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
                      recla_dir='{}data/new facies types 20220120.xlsx'.format(path))

facies, id_list = prepare.create_recla()
data_df = prepare.create_2d(facies=facies, id_list=id_list)
X = data_df.iloc[:, :-2].values
y = data_df['facies'].values
groups = data_df['core_section'].values

train_idx, test_idx = Split.train_test_split(y, groups)

# set the pipes and parameters
pipe_lr = Pipeline([
    ('scaling', StandardScaler()),
    ('pca', PCA()), 
    ('lr', LogisticRegression(max_iter = 10000, class_weight='balanced'))])

param_grid_lr = [
    {'pca': [PCA(n_components=50, whiten=True)],
     'lr__C': np.logspace(-4, 5, 10)},
    {'pca': [None],
     'lr__C': np.logspace(-4, 5, 10)}
]

pipe_svc = Pipeline([('scaling', StandardScaler()),
                     ('pca', PCA()), ('svc', SVC(class_weight='balanced'))])

param_grid_svc = [
    {'pca': [PCA(n_components=50, whiten=True)],
     'svc__C': np.logspace(-3, 2, 6),
     'svc__gamma': np.logspace(-5, 0, 6)},
    {'pca': [None],
     'svc__C': np.logspace(-3, 2, 6),
     'svc__gamma': np.logspace(-5, 0, 6)}
]

pipe_rf = Pipeline([('scaling', StandardScaler()),
                 ('pca', PCA()), 
                 ('rf', RandomForestClassifier(class_weight='balanced', 
                                               random_state=24, 
                                               n_jobs=-1))])

param_grid_rf = [
    {'pca': [PCA(n_components=50, whiten=True)], 
     'rf__n_estimators':[100, 1000, 10000],
     'rf__max_depth': [3, 5, 10, 15]},
    {'scaling': [None],
     'pca': [None],
     'rf__n_estimators':[100, 1000, 10000],
     'rf__max_depth': [3, 5, 10, 15]}
]

# grid search
for model_name, pipe, param_grid in zip(['lr', 'svc', 'rf'], 
                                        [pipe_lr, pipe_svc, pipe_rf], 
                                        [param_grid_lr, param_grid_svc, 
                                         param_grid_rf]):
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
