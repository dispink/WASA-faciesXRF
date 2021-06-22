#!/usr/bin/env python
"""
This script is used to predict facies in the test set by using the SVC model and smoothing function.
"""
import numpy as np

def smooth(y, groups, window):
    """
    This function smoothes the labels in each core section.
    y is the label array.
    groups is the core section list.
    window is the window size each time the function looking at.
    """
    y_s = []
    for core_section in np.unique(groups):
        y_g = y[groups == core_section]
        i = 0

        # the last elements not enough for a window size will be excluded 
        while (i + window) < len(y_g):
            value, count = np.unique(y_g[i: i+window], return_counts=True)
            # set the dominant value in the window as the values
            y_s = np.hstack((y_s, np.repeat(value[np.argmax(count)], window)))
            i += window
        # add the last part of data that can't be smoothed to the new array
        y_s = np.hstack((y_s, y_g[i:]))
    return y_s.astype(int)

if __name__ == '__main__':
    import pandas as pd
    from split import *
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
