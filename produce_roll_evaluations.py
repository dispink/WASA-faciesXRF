import pandas as pd
from split import *
data_df = pd.read_csv('data/XRF_ML_cr.csv')
X = data_df.iloc[:, 1:-2].values
y = data_df['facies_merge_2'].values
groups = data_df['core_section'].values

train_idx, test_idx = my_train_test_split(y, groups)
trainn_idx, dev_idx = my_train_test_split(y[train_idx], groups[train_idx])

from joblib import load
lr = load('models/roll_lr_model_20210525.joblib')
svc = load('models/roll_svc_model_20210524.joblib')
rf = load('models/roll_rf_model_20210525.joblib')

y_df = pd.DataFrame(y[train_idx[dev_idx]], columns=['y'])
for col, model in zip(['y_lr', 'y_svc', 'y_rf'], [lr, svc, rf]):
    y_df[col] = model.predict(X[train_idx[dev_idx]])
    
y_df['core_section'] = groups[train_idx[dev_idx]]

y_df.to_csv('results/roll_dev_y.csv')