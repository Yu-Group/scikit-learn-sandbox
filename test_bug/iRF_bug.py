# this is the skeleton code for running benchmarks:
# in other words, it sets up the loop and runs iRF many times in hopes of
# finding the error
# it doesn't output any metrics or dictionaries.
# it literally just runs iRF many times

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

irf_utils.run_iRF_bug_test(X_train=X_train,
                      X_test=X_test,
                      y_train=y_train,
                      y_test=y_test,
                      K=specs['n_iter'],
                      n_estimators=specs['n_estimators'],
                      B=specs['n_bootstraps'],
                      random_state_classifier=405,
                      propn_n_samples=specs['propn_n_samples'],
                      bin_class_type=specs['bin_class_type'],
                      M=specs['n_RIT'],
                      max_depth=specs['max_depth'],
                      noisy_split=specs['noisy_split'],
                      num_splits=specs['num_splits'],
                      n_estimators_bootstrap=specs['n_estimators_bootstrap'])
