#!/usr/bin/env python
"""
This script is used to compile different post-process codes and excute the post-process on the dev set.
"""
import numpy as np
import pandas as pd

def smooth(y, groups, window):
    """
    This function smoothes the labels in each core section.
    y is the label array.
    groups is the core section list.
    window is the window size (data points) each time the function looking at.
    A ndarray is returned. 
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

def return_do(ar):
    """
    This function can return the dominant facies among the list of facies.
    If the amount of facies are equal, the facies corresponding to the first sorted occurrence is returned.
    This is due to the rule of np.argmax().
    """
    uniques, counts = np.unique(ar, return_counts=True)
    return uniques[np.argmax(counts)]


def detect_object(y_df, col='y_pred'):
    """
    The y_df needs to be a pd.DataFrame containing decision scores on each facies (X12), 
    prediction (y_pred) or other modified y, core section and composite_id.
    col stands for the column (y) you want to analyze on. In default, it's y_pred.
    A pd.DataFrame is returned.
    """
    id_ob = []
    thick_ob = []
    fa_ob = []
    fa_sec_ob = []
    section_ob = []

    # initialize, the values will be deleted at the end
    fa = -1
    section = -1
    id_list = [1]
    fa_sec_list = [1]


    for _, row in y_df.iterrows():
        # refresh when facies or core section changes
        if row[col] != fa or row.core_section != section:
            # record the previoius infos
            id_ob.append(id_list)
            ## I simply use the data amount to calculate thickness
            ## this way may lead to wrong thickness when there are missing
            ## data due to low data quality, but I don't think it matters a lot
            ## resolution: 2 mm
            thick_ob.append(len(id_list)*2)
            fa_ob.append(fa)
            fa_sec_ob.append(return_do(fa_sec_list))
            section_ob.append(section)

            # refresh
            id_list = [row.composite_id]
            fa = row[col]
            fa_sec_list = [np.argsort(row[:12].values)[-2]]
            section = row.core_section
        else:
            id_list.append(row.composite_id)
            # pick up the facies having second high score
            fa_sec_list.append(np.argsort(row[:12].values)[-2])

    # append the last object, which can't be recorded by above codes 
    # because there is no change in prediction or core section after it        
    id_ob.append(id_list)
    thick_ob.append(len(id_list)*2)
    fa_ob.append(fa)
    fa_sec_ob.append(return_do(fa_sec_list))
    section_ob.append(section)   
    
    # drop the first row, which records the initial infos
    objects = np.stack([np.array(id_ob, dtype=object), section_ob, thick_ob, fa_ob, fa_sec_ob], axis=0)[:, 1:]    
    return pd.DataFrame(objects.T, columns=['composite_id', 'core_section', 'thickness_mm', 'facies', 'facies_second'])

def add_facies(objects_df):
    """
    The objects_df needs to be the pd.DataFrame of the transposed matrix of detect_object(),
    having 'composite_id', 'core_section', 'thickness_mm', 'facies', 'facies_second' as columns.
    This function executes within each core section and return a pd.DataFrame.
    """
    fa_ab = []
    fa_bl = []
    for section in objects_df.core_section.unique():
        X = objects_df[objects_df.core_section == section].copy()
        for i in X.index:
            try:
                fa_ab.append(X.facies[i-1])
            except KeyError:
                fa_ab.append(None)
            try:
                fa_bl.append(X.facies[i+1])
            except KeyError:
                fa_bl.append(None)
    objects_df['facies_above'] = fa_ab
    objects_df['facies_below'] = fa_bl
    return objects_df

def vote(series):
    """
    The series must be pd.Series having facies_second, facies_above and facies_below as columns.
    This function can return the dominant facies among the list of facies (dominant second, above and below facies).
    If the amount of facies are equal, the above facies is returned. If the above facies is missing, the below facies is returned.
    This is different from return_do().
    An integer, representing facies, is returned.
    """
    series = series.astype(float)
    uniques, counts = np.unique(series, return_counts=True)
    if series.facies_second != series.facies_above != series.facies_below:
        try:
            return int(series.facies_above)
        except ValueError: # when above facies is None
            return int(series.facies_below)
    else:
        return int(uniques[np.argmax(counts)])
    
def replace(series):
    """
    The series need to be pd.Series having facies, thickness_mm,
    facies_second, facies_above and facies_below as columns.
    Basically, the series is one of the rows in objects_df.
    """
    
    thickness_dict = {
        # Shoreface
        '0': 100,
        # Channel
        '1': 100, 
        # Beach-foreshore
        '2': 100,
        # Sand flat
        '3': 100,
        # Mud flat
        '4': 100,
        # Lagoon
        '5': 50,
        # Peat
        '6': 50,
        # Soil
        '7': 30,
        # Eolian/fluvial (w)
        '8': 200,
        # Shallow marine
        '9': 200,
        # Moraine
        '10': 200,
        # Eolian/fluvial
        '11': 200
    }
    
    if series.thickness_mm < thickness_dict[str(series.facies)]:
        return vote(series[-3:])
    else:
        return series.facies
    
def transform_back(objects_df):
    """
    objects_df needs to be a pd.Dataframe containing composite_id and facies_replaced. It's generated after replacedments.
    """
    ids = []
    y_s = []

    for _, row in objects_df.iterrows():
        for composite_id in row.composite_id:
            ids.append(composite_id)
            y_s.append(row.facies_replaced)
    return pd.Series(y_s, index=ids, name='y_smooth')

def count_boundary(y, groups):
    """
    y is the label to count boundary within each section.
    groups is the core section.
    These two inputs can be list, ndarray and pd.Series.
    """
    bd = []
    for section in np.unique(groups):
        y_g = y[groups == section]
        for i in range(len(y_g) - 1):
            if y_g[i] != y_g[i+1]:
                bd.append(True)
            else:
                bd.append(False)
    return np.sum(bd)
    
if __name__ == '__main__':
    from split import *
    from joblib import load
    import datetime
    date = datetime.datetime.now().strftime('%Y%m%d')
    
    svc = load('models/roll_svc_model_20210524.joblib')
    data_df = pd.read_csv('data/XRF_ML_cr.csv')
    X = data_df.iloc[:, 1:-2].values
    y = data_df['facies_merge_2'].values
    groups = data_df['core_section'].values

    train_idx, test_idx = my_train_test_split(y, groups)
    trainn_idx, dev_idx = my_train_test_split(y[train_idx], groups[train_idx])
    
    y_df = pd.DataFrame(svc.decision_function(X[train_idx[dev_idx]]))
    y_df['y'] = y[train_idx[dev_idx]]
    # maybe I can try little smoothing prior to the sophisticate smoothing
    y_df['y_pred'] = svc.predict(X[train_idx[dev_idx]])
    y_df['y_pred_s'] = smooth(y_df.y_pred, groups[train_idx[dev_idx]], 15)
    y_df = pd.concat([y_df, data_df.composite_id[train_idx[dev_idx]].reset_index(drop=True), data_df.core_section[train_idx[dev_idx]].reset_index(drop=True)], 
                     axis=1, join='inner')
    
    objects_df = add_facies(detect_object(y_df, col='y_pred_s'))
    objects_df['facies_replaced'] = objects_df.apply(replace, axis=1)
    y_df = pd.concat([y_df.set_index('composite_id'), transform_back(objects_df)], join='inner', axis=1)
    
    # Outputs
    objects_df.to_csv('results/roll_post_obj_dev_{}.csv'.format(date))
    y_df.to_csv('results/roll_post_y_dev_{}.csv'.format(date))
