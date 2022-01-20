"""
This module is built for the process after reclassification.

NOTES for future update:
1. Ploshing the codes. 
"""

import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean

class PrepareData():
    """
    This is a class collecting functions to create data, which helps 
    decreasing data storage and loading time.
    """
    
    def __init__(self, data_dir='data/XRF_results.cleaned.all.csv', 
                 info_dir='data/info.cleaned.all.csv', 
                 recla_dir='data/new facies types 20220120.xlsx', 
                 elements=['Si', 'S', 'Cl', 'K', 'Ca', 'Ti', 
                           'Fe', 'Br', 'Rb', 'Sr', 'Zr', 'Ba']):        
        self.data_dir = data_dir
        self.info_dir = info_dir
        self.recla_dir = recla_dir
        self.elements = elements

    def clr(self, x):
        """
        This function replace the zero values to 1 and Centred log-ratio 
        transorm the data.
        x need to be a ndarray (matrix) of the 12 element intensities.
        """
        x = np.where(x == 0, 1, x) 
        return np.log(x / gmean(x, axis = 1).reshape(x.shape[0], 1))

    def create_recla(self):
        """
        The script creates the lists of reclassified facies (index not str) 
        and filtered composite_id.
        recla_dir is the directory of the reclassification excel.
        info_dir is the directory of the info file.
        """
        # read the excel of new classified labels and depths
        excel_df = pd.read_excel(self.recla_dir, skiprows=5)
        fa_list = []
        section_list = []
        # the boundary is in cm
        up_list = []
        bl_list = []

        for _, row in excel_df.iterrows():
            for seg in row['Core sections'].split('//'):
                if (seg != '') & (seg != ' '):      
                    fa_list.append(row.Label)
                    section_list.append(seg.split()[0])
                    up_list.append(int(seg.split()[1].split('-')[0]))
                    bl_list.append(int(seg.split()[1].split('-')[1]))

        # filter the data points according to the boundaries
        info_df = pd.read_csv(self.info_dir, index_col=0, 
                              usecols=['composite_id', 'core_section', 
                                       'section_depth_mm'])
        id_list = []
        facies = []

        for i in range(len(section_list)):
            X = info_df.index[
                (info_df.core_section == section_list[i]) & 
                (info_df.section_depth_mm >= up_list[i]*10) & 
                (info_df.section_depth_mm < bl_list[i]*10)
            ]

            facies += [fa_list[i] for _ in range(len(X))]
            id_list = np.hstack((id_list, X.values))
        return facies, id_list

    def create_raw(self, facies, id_list):
        """
        The function creates the clr transformed elemental 
        (also called raw in comparison with the other two represented 
        datasets) data with the labels and id_list from create_recla().
        The output dataframe consists composite_id (as index), the clr
        transformed 12 elements, factorized facies and core_section. 
        """
        data_df = pd.read_csv(self.data_dir, 
                              index_col=0).loc[id_list, self.elements]
        data_df['facies'] = facies
        norm_df = pd.concat(
            [pd.DataFrame(self.clr(data_df.iloc[:, :-1].values), 
                          index = data_df.index, columns = data_df.columns[:-1]), 
             data_df.facies,
             pd.read_csv(self.info_dir, index_col=0, 
                         usecols=['composite_id', 'core_section'])],
            join = 'inner', axis = 1)
        return norm_df


    def create_roll(self, facies, id_list, window=17):
        """
        The function creates the rolling elemental data with the labels 
        and id_list from create_recla().
        It is modified from ML_element_01.ipynb.
        The output dataframe consists composite_id (as index), the 
        rolling data developed from clr transformed 12 elements, 
        factorized facies and core_section. 
        """
        data_df = pd.read_csv(self.data_dir, 
                              index_col=0).loc[id_list, self.elements]

        # build column names
        new_cols = []
        for fun in ['_mean', '_std']:
            new_cols = np.hstack((new_cols, 
                                  [col+fun for col in data_df.columns]))

        data_df['facies'] = facies
        norm_df = pd.concat(
            [pd.DataFrame(self.clr(data_df.iloc[:, :-1].values), 
                          index = data_df.index, 
                          columns = data_df.columns[:-1]), 
             data_df.facies,
             pd.read_csv(self.info_dir, index_col=0, 
                         usecols=['composite_id', 'core_section'])],
            join = 'inner', axis = 1)

        # make sure the order is sorted by the composite_id and then 
        # the order within section is sorted by the section depth simutaniously.
        norm_df = norm_df.sort_values('composite_id')

        # start rolling
        r_df = pd.DataFrame()
        cols = norm_df.columns[:-2]
        for section in np.unique(norm_df.core_section):
            rolling = norm_df.loc[norm_df.core_section == section, cols].rolling(window = window, center = True)
            r_df = r_df.append(pd.concat([rolling.mean(), 
                                          rolling.std()], 
                                         axis = 1, join = 'inner'))

        r_df.columns = new_cols
        r_df = pd.concat([r_df, norm_df.loc[:, ['facies', 'core_section']]], 
                         join = 'inner', axis = 1)
        r_c_df = r_df.dropna(axis = 0)

        # return the pd.DataFrame
        return r_c_df

    def create_2d(self, facies, id_list, half_window=8):
        """
        The function creates the image-like elemental data with the 
        labels and id_list from create_recla(). Since the data size is
        huge, three ndarrays are output intead of the pd>DataFrame like 
        create_raw() and create_roll. They are the image-like data 
        developed from clr transformed 12 elements, factorized facies, 
        core_section and index. 
        """
        # adopt and sort the clr transformed raw element data
        # and then use numbers as the index
        # sorting is to make sure the depth are not mixed
        norm_df = self.create_raw(facies=facies, id_list=id_list).sort_values('composite_id').reset_index()

        X = []
        id_list = []
        for section in norm_df.core_section.unique():
            for index in norm_df.index[norm_df.core_section == section][half_window:-half_window]:
                X.append(norm_df.iloc[index-half_window: index+half_window+1, 1:-2].values.ravel())
                id_list.append(index)

        #return np.array(X), norm_df.facies[id_list].values, norm_df.core_section[id_list].values, norm_df.composite_id[id_list].values
        out_df = pd.DataFrame(np.array(X))
        out_df['facies'] = norm_df.facies[id_list].values
        out_df['core_section'] = norm_df.core_section[id_list].values
        out_df.index = norm_df.composite_id[id_list].values
        
        return out_df
    
class Split():
    """
    This class combines the functions related to data spliting.
    """
    
    def OnegrupOnefacies_cv(y, groups, n_splits = 5, random_state = 24):
        """
        This function is for integrating with sklearn.GridSearchCV.
        It picks up one section in each facies as the test set randomly 
        while the rest are as training set.
        """
        np.random.seed(random_state) 

        for _ in range(n_splits):
            # pick up one section from each facies to test set
            sections_test = []
            for fa in np.unique(y):
                sections_test = np.hstack([sections_test, 
                                           np.random.choice(np.unique(groups[y == fa]), 1)])

            # build the indices for data points
            test_idxs = []
            for section in np.unique(sections_test):
                test_idxs = np.hstack([test_idxs, np.where(groups == section)[0]])
            test_idxs = test_idxs.astype(int)

            # the training indices are the rest of indices
            train_idxs = np.array(
                list(
                    set(np.arange(0, len(y), 1)) - set(test_idxs)
                )
            )

            yield train_idxs, test_idxs
            
    def train_test_split(y, groups, random_state = 24):
        """
        This function is to directly generate indices for the training/
        test set, which follows the strategy of OnegroupOnefacies_cv.
        """
        np.random.seed(random_state) 

        # pick up one section from each facies to test set
        sections_test = []
        for fa in np.unique(y):
            sections_test = np.hstack([sections_test, 
                                       np.random.choice(
                                           np.unique(groups[y == fa]), 1)])
        
        # build the indices for data points
        test_idxs = []
        for section in np.unique(sections_test):
            test_idxs = np.hstack([test_idxs, np.where(groups == section)[0]])
        test_idxs = test_idxs.astype(int)
        
        # the training indices are the rest of indices
        train_idxs = np.array(
            list(
                set(np.arange(0, len(y), 1)) - set(test_idxs)
            )
        )
        
        return train_idxs, test_idxs

class Evaluation():
    def __init__(self, 
                 recla_dir='data/new facies types 20220120.xlsx'):        
        self.recla_dir = recla_dir

    def get_Facies(self, out='Abbreviation'):
        """
        This is a function to out put a array of facies names from
        the excel used in def create_recla. You can choose to out put
        the full or abbreviation of facies names.  
        """
        excel_df = pd.read_excel(self.recla_dir, skiprows=5)
        if out in ['Abbreviation', 'Facies']:
            return excel_df[out].values
        else:
            print('Parameter out should only be Abbreviation or Facies.')

    def detect_conjuction(data_df, y_preds, facies_amount):
        """
        data_df is the dataframe having core_section and labels.
        y_preds is a list of the column names of labels in data_df you
        want to plot. facies_amount gives ability to adopt different 
        classification categories.
        The function return a tuple of conjunction matrices.
        """
        
        data_df['id'] = range(len(data_df))
        mat_dict = {}

        for y_pred in y_preds:
            y_mean = []
            composite_id = []
            model_name = y_pred[2:] if len(y_pred)>1 else 'description'
            X_df = data_df[~data_df[y_pred].isna()]
            
            for section in np.unique(X_df.core_section):
                y_mean = np.hstack((y_mean, X_df.loc[X_df.core_section == section, y_pred].rolling(window = 2).mean()))
                composite_id = np.hstack((composite_id, X_df[X_df.core_section == section].index))

            # merge the rolling with the original label
            # this will lost the first data poit in each section, but it doesn't mater
            # because the boundary shouldn't be at the begining of each section
            tmp_df = pd.concat([X_df, pd.DataFrame({'y_mean': y_mean}, index = composite_id).dropna()], join = 'inner', axis = 1)

            # if y_pred is not same as y_mean, it means there is a boundary (facies change) between this and above depth
            bd_df = tmp_df[tmp_df[y_pred] != tmp_df.y_mean].copy()
            bd_df['ab_id'] = bd_df.id - 1
            if len(model_name) > 6:
                print('There are {} boundaries in description.'.format(len(bd_df)))
            else:
                print('There are {} boundaries in the optimized {} model\'s predictions.'.format(len(bd_df), model_name))

            # build matrix of boundaries
            bd_mat = np.zeros(shape = (facies_amount, facies_amount), dtype = int)

            for row in bd_df.iterrows():
                tmp = np.zeros(shape = (facies_amount, facies_amount), dtype = int)
                tmp[int(data_df[y_pred][int(row[1].ab_id)]), 
                    int(data_df[y_pred][int(row[1].id)])] = 1
                bd_mat += tmp
                
            mat_dict[y_pred] = bd_mat
        
        return mat_dict

    def make_recall(data_df, y_preds, Facies):
        """
        data_df is the dataframe having core_section and labels.
        y_preds is a list of the column names of labels you want to 
        plot.
        Faceis is a list of facies name, which should correspond to
        the order in recla_dir.
         The function ouput a tuple of pd.DataFrames of confusion 
        matrices.
        """
        from sklearn.metrics import confusion_matrix

        con_per = {}
        for col in y_preds:
            # make confusion matrix between prediction and actual labels
            X_df = data_df[~data_df[col].isna()]
            confusion = confusion_matrix(X_df.y, X_df[col])

            # normalize the counts in each row and present in percent
            x = np.copy(confusion).astype(float)
            x /= x.sum(axis = 1, keepdims = True)
            con_per[col] = pd.DataFrame((x*100).astype(int), index = Facies, columns = Facies)

        return con_per
            