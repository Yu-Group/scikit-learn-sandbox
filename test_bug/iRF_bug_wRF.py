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

# load features
features = np.loadtxt('./data/breast_cancer_features.csv', delimiter=',')
responses = np.loadtxt('./data/breast_cancer_responses.csv', delimiter=',')

# load specs
specs = py_irf_benchmarks_utils.yaml_to_dict(inp_yaml='./specs/iRF_mod_bug.yaml')

print(specs)

X_train, X_test, y_train, y_test = train_test_split(
    features, responses, train_size=specs['train_split_propn'], random_state = 24)

feature_importances = \
    [  0.00000000e+00,   7.51318453e-03,   0.00000000e+00,   5.17122730e-03,
       0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00643423e-01,
       0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   2.70698692e-04,
       1.45406955e-03,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
       0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
       1.65213916e-01,   2.05304918e-02,   3.18705597e-01,   8.43881639e-02,
       7.43857708e-04,   0.00000000e+00,   6.64732679e-04,   2.93743220e-01,
       0.00000000e+00,   9.57419390e-04]


rf = RandomForestClassifier(n_estimators=specs['n_estimators'], random_state = 409)
rf.fit(X=X_train, y=y_train, feature_weight=feature_importances)
# rf.fit(X=X_train, y=y_train) # try with unweighted rf -- this should work fine

for idx, dtree in enumerate(rf.estimators_):
    print(idx)

    #dtree_out = irf_utils._get_tree_data(X_train=X_train,
    #                           X_test=X_test,
    #                           y_test=y_test,
    #                           dtree=dtree,
    #                           root_node_id=0)
    all_features_idx = np.array(range(np.shape(X_train)[1]), dtype='int64')
    node_features_raw_idx = dtree.tree_.feature
    print(np.array(node_features_raw_idx))
    node_features_idx = all_features_idx[np.array(node_features_raw_idx)]
