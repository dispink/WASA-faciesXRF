import numpy as np
import pandas as pd
from split import *
from create_2d_data import *
X, y, groups = create_2d('data/XRF_ML.csv')

train_idx, test_idx = my_train_test_split(y, groups)
trainn_idx, dev_idx = my_train_test_split(y[train_idx], groups[train_idx])

from joblib import load
lr = load('models/2d_lr_model_20210520.joblib')
svc = load('models/2d_svc_model_20210521.joblib')
rf = load('models/2d_rf_model_20210522.joblib')

y_df = pd.DataFrame(y[train_idx[dev_idx]], columns=['y'])
for col, model in zip(['y_lr', 'y_svc', 'y_rf'], [lr, svc, rf]):
    y_df[col] = model.predict(X[train_idx[dev_idx]])
    
y_df['core_section'] = groups[train_idx[dev_idx]]

y_df.to_csv('results/2d_dev_y.csv')