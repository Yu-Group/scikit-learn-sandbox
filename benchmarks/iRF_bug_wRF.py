# isolated iRF bug to weighted random forest

#matplotlib inline
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
import numpy as np
from functools import reduce

# Needed for the scikit-learn wrapper function
from sklearn.tree import irf_utils
from sklearn.ensemble import RandomForestClassifier
from math import ceil
from sklearn.model_selection import train_test_split

import itertools
import sys
sys.path.append('../benchmarks')
import py_irf_benchmarks_utils

for i in range(1): # loop through all parameters

    # load features
    features = np.loadtxt('./data/breast_cancer_features.csv', delimiter=',')
    responses = np.loadtxt('./data/breast_cancer_responses.csv', delimiter=',')

    # load specs
    specs = py_irf_benchmarks_utils.yaml_to_dict(inp_yaml='./specs/iRF_mod_bug.yaml')

    print(specs)

    X_train, X_test, y_train, y_test = train_test_split(
        features, responses, train_size=specs['train_split_propn'], random_state = 24)

    feature_importances = \
        [ 0.,          0.01013135,  0.        ,  0.00061531,  0.        ,  0.        ,  0.,
          0.17372871,  0.        ,  0.        ,  0.        ,  0.        ,  0.00091531,
          0.,          0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.,
          0.10789531,  0.02599266,  0.29088859,  0.11955769,  0.00040694,  0.        ,  0.,
          0.269499,    0.        ,  0.00036912]



    rf = RandomForestClassifier(n_estimators=specs['n_estimators'], random_state = 409)
    rf.fit(X=X_train, y=y_train, feature_weight=feature_importances)
