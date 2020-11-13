import numpy as np

def my_group_stratify_shuffle_cv(X, y, groups, n_splits = 5, random_state = 24):
    np.random.seed(random_state) 

    for _ in range(n_splits):
        # make sure the test set has at least one section for each facies
        sections_test = []
        for fa in np.unique(y):
            sections_test = np.hstack([sections_test, np.random.choice(np.unique(groups[y == fa]), 1)])
        # randomly pick 40 sections from the whole dataset and add them to the test set
        sections_test = np.hstack([sections_test, np.random.choice(np.unique(groups), 40)])

        # build the indices for data points
        test_idxs = []
        for section in np.unique(sections_test):
            test_idxs = np.hstack([test_idxs, np.where(groups == section)[0]]).astype(int)
        
        # the training indices are the rest of indices
        train_idxs = np.array(
            list(
                set(np.arange(0, len(y), 1)) - set(test_idxs)
            )
        )
        
        yield train_idxs, test_idxs
        
def my_train_test_split(X, y, groups, random_state = 24):
    np.random.seed(random_state) 

    # make sure the test set has at least one section for each facies
    sections_test = []
    for fa in np.unique(y):
        sections_test = np.hstack([sections_test, np.random.choice(np.unique(groups[y == fa]), 1)])
    # randomly pick 40 sections from the whole dataset and add them to the test set
    sections_test = np.hstack([sections_test, np.random.choice(np.unique(groups), 40)])

    # build the indices for data points
    test_idxs = []
    for section in np.unique(sections_test):
        test_idxs = np.hstack([test_idxs, np.where(groups == section)[0]]).astype(int)

    # the training indices are the rest of indices
    train_idxs = np.array(
        list(
            set(np.arange(0, len(y), 1)) - set(test_idxs)
        )
    )

    #return X[train_idxs], X[test_idxs], y[train_idxs], y[test_idxs], groups[train_idxs], groups[test_idxs]
    return train_idxs, test_idxs

def my_dask_cv(X, y, groups, n_splits = 5, random_state = 24):
    " X, y, groups are in dask.array format"
    np.random.seed(random_state) 

    for _ in range(n_splits):
        # make sure the test set has at least one section for each facies
        sections_test = []
        for fa in np.unique(y).compute_chunk_sizes():
            sections_test = np.hstack([sections_test, np.random.choice(np.unique(groups[y == fa]), 1)])
        # randomly pick 40 sections from the whole dataset and add them to the test set
        sections_test = np.hstack([sections_test, np.random.choice(np.unique(groups), 40)])

        # build the indices for data points
        test_idxs = []
        for section in np.unique(sections_test).compute_chunk_sizes():
            test_idxs = np.hstack([test_idxs, np.where(groups == section)[0]]).astype(int)
        
        # the training indices are the rest of indices
        train_idxs = np.array(
            list(
                set(np.arange(0, len(y), 1)) - set(test_idxs)
            )
        )
        
        yield train_idxs, test_idxs