import os
import pickle

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer, LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from .FeatureBinarizer import FeatureBinarizer

def predefined_dataset(name, binary_y=False):
    """
    Define how to read specific datasets and return structured X and Y data.
    
    Args
        name (str): the name of the dataset to read.
        binary_y (bool): if True, force the dataset to only have two classes.
        
    Returns
        table_X (DataFrame): instances, values can be strings or numbers.
        table_Y (DataFrame): labels, values can be strings or numbers.
        categorical_cols (list): A list of column names that are categorical data. 
        numerical_cols (list): A list of column names that are numerical data.
    """
    
    dir_path = os.path.dirname(os.path.realpath(__file__)) # .py
    
    ### UCI datasets
    if name == 'adult':
        # https://archive.ics.uci.edu/ml/datasets/adult
        # X dim: (30162, 14)
        # Y counts: {'<=50K': 22654, '>50K': 7508}
        table = pd.read_csv(dir_path + '/adult/adult.data', header=0, na_values='?', skipinitialspace=True).dropna()
        table_X = table.iloc[:, :-1].copy()
        table_Y = table.iloc[:, -1].copy()
        categorical_cols = None
        numerical_cols = None
        
    elif name == 'magic':
        # http://archive.ics.uci.edu/ml/datasets/MAGIC+GAMMA+Telescope
        # X dim: (19020, 10/90)
        # Y counts: {'g': 12332, 'h': 6688}
        table = pd.read_csv(dir_path + '/magic/magic04.data', header=0, na_values='?', skipinitialspace=True).dropna()
        table_X = table.iloc[:, :-1].copy()
        table_Y = table.iloc[:, -1].copy()
        categorical_cols = None
        numerical_cols = None
    
    ### OpenML datasets
    elif name == 'house':
        # https://www.openml.org/d/821
        # X dim: (22784, 16/132)
        # Y counts: {'N': 6744, 'P': 16040}
        table = pd.read_csv(dir_path + '/house/house_16H.csv', header=0, skipinitialspace=True)
        table_X = table.iloc[:, :-1].copy()
        table_Y = table.iloc[:, -1].copy()
        categorical_cols = None
        numerical_cols = None
    
    ### Others
    elif name == 'heloc':
        # https://community.fico.com/s/explainable-machine-learning-challenge?tabset-3158a=2&tabset-158d9=3
        # X dim: (2502, 23)
        # Y counts: {'Bad': 1560, 'Good': 942}
        table = pd.read_csv(dir_path + '/heloc/heloc_dataset_v1.csv', header=0, na_values=['-7', '-8', '-9'], skipinitialspace=True)#.dropna()
        table_X = table.iloc[:, 1:].copy()
        table_Y = table.iloc[:, 0].copy()
        categorical_cols = None
        numerical_cols = None
        
    else:
        raise NameError(f'The input dataset is not found: {name}.')

    return table_X, table_Y, categorical_cols, numerical_cols

def transform_dataset(name, method='ordinal', negations=False, labels='ordinal'):
    """
    Transform values in datasets (from predefined_dataset) into real numbers or binary numbers.
    
    Args
        name (str): the name of the dataset.
        method (str): specify how the instances are encoded:
            'origin': encode categorical features as integers and leave the numerical features as they are (float).
            'ordinal': encode all features as integers; numerical features are discretized into intervals.
            'onehot': one-hot encode the integer features transformed using 'ordinal' method.
            'onehot-compare': one-hot encode the categorical features just like how they are done in 'onehot' method; 
                one-hot encode numerical features by comparing them with different threhsolds and encode 1 if they are smaller than threholds. 
        negations (bool): whether append negated binary features; only valid when method is 'onehot' or 'onehot-compare'. 
        labels (str): specify how the labels are transformed.
            'ordinal': output Y is a 1d array of integer values ([0, 1, 2, ...]); each label is an integer value.
            'binary': output Y is a 1d array of binary values ([0, 1, 0, ...]); each label is forced to be a binary value (see predefined_dataset).
            'onehot': output Y is a 2d array of one-hot encoded values ([[0, 1, 0], [1, 0, 0], [0, 0, 1]]); each label is a one-hot encoded 1d array.
    
    Return
        X (DataFrame): 2d float array; transformed instances.
        Y (np.array): 1d or 2d (labels='onehot') integer array; transformed labels;.
        X_headers (list|dict): if method='ordinal', a dict where keys are features and values and their categories; otherwise, a list of binarized features.
        Y_headers (list): the names of the labels, indexed by the values in Y.
    """
    
    METHOD = ['origin', 'ordinal', 'onehot', 'onehot-compare']
    LABELS = ['ordinal', 'binary', 'onehot']
    if method not in METHOD:
        raise ValueError(f'method={method} is not a valid option. The options are {METHOD}')
    if labels not in LABELS:
        raise ValueError(f'labels={labels} is not a valid option. The options are {LABELS}')
    
    table_X, table_Y, categorical_cols, numerical_cols = predefined_dataset(name, binary_y=labels == 'binary')

    # By default, columns with object type are treated as categorical features and rest are numerical features
    # All numerical features that have fewer than 5 unique values should be considered as categorical features
    if categorical_cols is None:
        categorical_cols = list(table_X.columns[(table_X.dtypes == np.dtype('O')).to_numpy().nonzero()[0]])
    if numerical_cols is None:
        numerical_cols = [col for col in table_X.columns if col not in categorical_cols and np.unique(table_X[col].to_numpy()).shape[0] > 5]
        categorical_cols = [col for col in table_X.columns if col not in numerical_cols]
            
    # Fill categorical nan values with most frequent value and numerical nan values with the mean value
    if len(categorical_cols) != 0:
        imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        table_X[categorical_cols] = imp_cat.fit_transform(table_X[categorical_cols])
    if len(numerical_cols) != 0:
        imp_num = SimpleImputer(missing_values=np.nan, strategy='mean')
        table_X[numerical_cols] = imp_num.fit_transform(table_X[numerical_cols])
        
    if np.nan in table_X or np.nan in table_Y:
        raise ValueError('Dataset should not have nan value!')
        
    # Encode instances
    X = table_X.copy()
    
    col_categories = []
    if method in ['origin', 'ordinal'] and len(categorical_cols) != 0:
        # Convert categorical strings to integers that represent different categories
        ord_enc = OrdinalEncoder()
        X[categorical_cols] = ord_enc.fit_transform(X[categorical_cols])
        col_categories = {col: list(categories) for col, categories in zip(categorical_cols, ord_enc.categories_)}

    col_intervals = []
    if method in ['ordinal', 'onehot'] and len(numerical_cols) != 0:
        # Discretize numerical values to integers that represent different intervals
        kbin_dis = KBinsDiscretizer(encode='ordinal', strategy='kmeans')
        X[numerical_cols] = kbin_dis.fit_transform(X[numerical_cols])
        col_intervals = {col: [f'({intervals[i]:.2f}, {intervals[i+1]:.2f})' for i in range(len(intervals) - 1)] for col, intervals in zip(numerical_cols, kbin_dis.bin_edges_)}

        if method in ['onehot']:
            # Make numerical values to interval strings so that FeatureBinarizer can process them as categorical values
            for col in numerical_cols:
                X[col]  = np.array(col_intervals[col]).astype('object')[X[col].astype(int)]

    if method in ['onehot', 'onehot-compare']:
        # One-hot encode categorical values and encode numerical values by comparing with thresholds
        fb = FeatureBinarizer(colCateg=categorical_cols, negations=negations)
        X = fb.fit_transform(X)
    
    if method in ['origin']:
        # X_headers is a list of features
        X_headers = [column for column in X.columns]
    if method in ['ordinal']:
        # X_headers is a dict where keys are features and values and their categories
        X_headers = {col: col_categories[col] if col in col_categories else col_intervals[col] for col in table_X.columns}
    else:
        # X_headers is a list of binarized features
        X_headers = ["".join(map(str, column)) for column in X.columns]
        
    if method not in ['origin']:
        X = X.astype(int)
    
    # Encode labels
    le = LabelEncoder()
    Y = le.fit_transform(table_Y).astype(int)
    Y_headers = le.classes_
    if labels == 'onehot':
        lb = LabelBinarizer()
        Y = lb.fit_transform(Y)
    
    return X, Y, X_headers, Y_headers

def split_dataset(X, Y, test=0.2, shuffle=None):    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test, random_state=shuffle)
    
    return X_train, X_test, Y_train, Y_test

def kfold_dataset(X, Y, k=5, shuffle=None):
    kf = StratifiedKFold(n_splits=k, shuffle=bool(shuffle), random_state=shuffle)
    datasets = [(X.iloc[train_index], X.iloc[test_index], Y[train_index], Y[test_index]) 
                for train_index, test_index in kf.split(X, Y if len(Y.shape) == 1 else Y.argmax(1))]
    
    return datasets

def nested_kfold_dataset(X, Y, outer_k=5, inner_k=5, shuffle=None):
    inner_kf = StratifiedKFold(n_splits=inner_k, shuffle=bool(shuffle), random_state=shuffle)
    
    datasets = []
    for dataset in kfold_dataset(X, Y, k=outer_k, shuffle=shuffle):
        X_train_valid, X_test, Y_train_valid, Y_test = dataset
        
        nested_datasets = []
        for train_index, valid_index in inner_kf.split(
            X_train_valid, Y_train_valid if len(Y.shape) == 1 else Y_train_valid.argmax(1)):
            X_train = X.iloc[train_index]
            X_valid = X.iloc[valid_index]
            Y_train = Y[train_index]
            Y_valid = Y[valid_index]
            nested_datasets.append([X_train, X_valid, Y_train, Y_valid])
        datasets.append([X_train_valid, X_test, Y_train_valid, Y_test, nested_datasets])
    
    return datasets