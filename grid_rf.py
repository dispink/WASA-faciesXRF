import numpy as np 
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from split import my_train_test_split, my_group_stratify_shuffle_cv

import datetime
date = datetime.datetime.now().strftime('%Y%m%d')

from time import perf_counter
start = perf_counter()

path = '/home/users/aslee/WASA_faciesXRF/'
print('Begin: RF')

data_df = pd.read_csv('{}data/XRF_ML_cr.csv'.format(path))
X = data_df.iloc[:, 1:-2].values
y = data_df['facies_merge_1'].values
groups = data_df['core_section'].values

train_idx, test_idx = my_train_test_split(X, y, groups)
X_train = X[train_idx]
y_train = y[train_idx]
groups_train = groups[train_idx]

del data_df, X, y, groups

pipe = Pipeline([('scaling', StandardScaler()),('pca', PCA(whiten=True)), ('rf', RandomForestClassifier(class_weight='balanced', random_state=24, n_jobs=-1))])

param_grid = [
    {'scaling': [StandardScaler()],
     'pca': [PCA(whiten=True)],
     'rf__n_estimators':[100, 1000, 5000]},
    {'scaling': [None],
     'pca': [None],
     'rf__n_estimators':[100, 1000, 5000]}
]

mycv = my_group_stratify_shuffle_cv(X_train, y_train, groups_train)

grid = GridSearchCV(pipe, param_grid = param_grid, cv = mycv, scoring = 'balanced_accuracy', 
                    return_train_score = False, n_jobs = 30)

grid.fit(X_train, y_train)

print("Best score on validation set: {:.2f}".format(grid.best_score_)) 
print("Best parameters: ", grid.best_params_)

pd.DataFrame(grid.cv_results_).to_csv('{}results/roll_rf_grid_ss_{}.csv'.format(path, date))

from joblib import dump, load
dump(grid.best_estimator_, '{}models/roll_rf_model_ss_{}.joblib'.format(path, date)) 

print("The computation takes {} hours.".format((perf_counter() - start)/3600))