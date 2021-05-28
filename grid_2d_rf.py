import numpy as np 
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from split import *
from create_2d_data import *
from sklearn.metrics import balanced_accuracy_score as score

import datetime
date = datetime.datetime.now().strftime('%Y%m%d')

from time import perf_counter
start = perf_counter()

path = '/home/users/aslee/WASA_faciesXRF/'
print('RF: image-like data')

X, y, groups = create_2d('{}data/XRF_ML.csv'.format(path))

train_idx, test_idx = my_train_test_split(y, groups)
X_train = X[train_idx]
y_train = y[train_idx]
groups_train = groups[train_idx]

# This time I split the training set again to obtain dev set
trainn_idx, dev_idx = my_group_stratify_shuffle(X_train, y_train, groups_train)
X_trainn = X[train_idx[trainn_idx]]
X_dev = X[train_idx[dev_idx]]
y_trainn = y[train_idx[trainn_idx]]
y_dev = y[train_idx[dev_idx]]
groups_trainn = groups[train_idx[trainn_idx]]

del X, y, groups

rf = Pipeline([('scaling', StandardScaler()),('pca', PCA(whiten=True)), ('rf', RandomForestClassifier(class_weight='balanced', random_state=24, max_depth=5, n_jobs=-1))])

param_grid = [
    {'pca': [PCA(whiten=True)], 
     'rf__n_estimators':[100, 1000, 5000],
     'rf__max_depth': [5, 10, 15]},
    {'scaling': [None],
     'pca': [None],
     'rf__n_estimators':[100, 1000, 5000],
     'rf__max_depth': [5, 10, 15]}
]

#rf = RandomForestClassifier(class_weight='balanced', random_state=24, n_jobs=-1)

#param_grid = {'max_depth': [5], 'n_estimators': [100, 1000, 5000, 10000, 50000]}

mycv = my_group_stratify_shuffle_cv(X_trainn, y_trainn, groups_trainn)

grid = GridSearchCV(rf, param_grid = param_grid, cv = mycv, scoring = 'balanced_accuracy', n_jobs = 20)

grid.fit(X_trainn, y_trainn)

print("Best score on validation set: {:.2f}".format(grid.best_score_)) 
print("Best parameters: ", grid.best_params_)

pd.DataFrame(grid.cv_results_).to_csv('{}results/2d_rf_grid_{}.csv'.format(path, date))

from joblib import dump, load
dump(grid.best_estimator_, '{}models/2d_rf_model_{}.joblib'.format(path, date)) 

print('Balanced score of on dev set: {:.2f}'.format(score(y_true=y_dev, y_pred=grid.best_estimator_.predict(X_dev))))

print("The computation takes {} hours.".format((perf_counter() - start)/3600))
