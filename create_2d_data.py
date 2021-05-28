'''
Create the 2d elemental data since it is way faster to create it from the raw data than read the finished dataset.
This function still needs to be more general  and maybe included into a class.
'''
import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean

def create_2d(file_dir):
    data_df = pd.read_csv(file_dir)
    x = data_df.iloc[:, 6:].values
    # replace the zero to 1
    x = np.where(x == 0, 1, x) 
    x = np.log(x / gmean(x, axis = 1).reshape(x.shape[0], 1))
    
    norm_df = pd.concat(
    [pd.DataFrame(x, columns = data_df.columns[6:]), data_df[['composite_id', 'core_section', 'facies_merge_2']]],
    join = 'inner', axis = 1
    )
    norm_df = norm_df.sort_values('composite_id')
    
    X = []
    id_list = []
    for section in norm_df.core_section.unique():
        for index in norm_df.index[norm_df.core_section == section][8:-8]:
            X.append(norm_df.iloc[index-8: index+9, :-3].values.ravel())
            id_list.append(index)
            
    return np.array(X), data_df.facies_merge_2[id_list].values, data_df.core_section[id_list].values
