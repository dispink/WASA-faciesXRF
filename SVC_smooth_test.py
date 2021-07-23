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
    
    y_df = pd.DataFrame(svc.decision_function(X[test_idx]))
    y_df['y'] = y[test_idx]
    y_df['y_pred'] = svc.predict(X[test_idx])
    y_df['y_pred_s'] = smooth(y_df.y_pred, groups[test_idx], 15)
    y_df = pd.concat([y_df, 
                      data_df.composite_id[test_idx].reset_index(drop=True), 
                      data_df.core_section[test_idx].reset_index(drop=True)], 
                     axis=1, join='inner')
    
    objects_df = add_facies(detect_object(y_df, col='y_pred_s'))
    objects_df['facies_replaced'] = objects_df.apply(replace, axis=1)
    y_df = pd.concat([y_df.set_index('composite_id'), transform_back(objects_df)], join='inner', axis=1)
    
    # Outputs
    objects_df.to_csv('results/roll_post_obj_test_{}.csv'.format(date))
    y_df.to_csv('results/roll_post_y_test_{}.csv'.format(date))

    # window size 75 is chosen based on the result of post_process.ipynb.
    #y_pred = smooth(svc.predict(X[test_idx]), groups[test_idx], 75)
    
    #pd.DataFrame({
    #    'y': y[test_idx],
    #    'y_svc_s': y_pred,
    #    'core_section': groups[test_idx]
    #}, index=data_df.composite_id[test_idx]).to_csv('results/roll_test_y_{}.csv'.format(date))
