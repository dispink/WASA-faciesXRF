"""
It is to build the optimal model on the reclassified rolling data
having calibrated probability. The parameters are adopted from the
grid search.
"""
from wasafacies import PrepareData, Split

import datetime
date = datetime.datetime.now().strftime('%Y%m%d')

from time import perf_counter
start = perf_counter()

path = '/home/users/aslee/WASA_faciesXRF/'

print('Build the optimal model (SVC) on training set')
print('It has the calibrated probability')

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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = make_pipeline(
    StandardScaler(), 
    SVC(C=1, gamma=1e-3, class_weight='balanced', probability=True))

pipe.fit(X[train_idx], y[train_idx])

from joblib import dump
dump(pipe, '{}models/r_roll_svc_model_prob_{}.joblib'.format(path, date)) 

print("The computation takes {} mins.".format((perf_counter() - start)/60))