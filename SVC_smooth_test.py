#!/usr/bin/env python
"""
This script is used to predict facies in the test set by using the SVC model and smoothing function.
"""

if __name__ == '__main__':
    import pandas as pd
    from split import *
    from post_process import *
    from joblib import load
    import datetime
    date = datetime.datetime.now().strftime('%Y%m%d')
    
    data_df = pd.read_csv('data/XRF_ML_cr.csv')
    X = data_df.iloc[:, 1:-2].values
    y = data_df['facies_merge_2'].values
    groups = data_df['core_section'].values
    train_idx, test_idx = my_train_test_split(y, groups)
    
    svc = load('models/roll_svc_model_20210621.joblib')
    
    # window size 75 is chosen based on the result of post_process.ipynb.
    y_pred = smooth(svc.predict(X[test_idx]), groups[test_idx], 75)
    
    pd.DataFrame({
        'y': y[test_idx],
        'y_svc_s': y_pred,
        'core_section': groups[test_idx]
    }, index=data_df.composite_id[test_idx]).to_csv('results/roll_test_y_{}.csv'.format(date))
