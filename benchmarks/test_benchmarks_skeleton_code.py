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
import py_irf_benchmarks2

# Import our custom utilities
from imp import reload
import sys
sys.path.insert(0, '../jupyter')

from utils import irf_jupyter_utils
reload(irf_jupyter_utils)

# load features
features = np.loadtxt('./data/breast_cancer_features.csv', delimiter=',')
responses = np.loadtxt('./data/breast_cancer_responses.csv', delimiter=',')

# load specs
specs = py_irf_benchmarks2.yaml_to_dict(inp_yaml='./specs/iRF_mod01.yaml')

# set up loop to go through all
varNames = sorted(specs)
spec_comb = [dict(zip(varNames, prod)) \
    for prod in itertools.product(*(specs[name] for name in varNames))]
print(spec_comb)

# run iRF_mod01
print(len(spec_comb))
for i in range(len(spec_comb)): # loop through all parameters

    print(spec_comb[i])

    # parse data
    [X_train, X_test, y_train, y_test] =\
             py_irf_benchmarks2.parse_data(features, responses, \
                        train_split_propn = spec_comb[i]['train_split_propn'],\
                        N_obs = spec_comb[i]['N_obs'], \
                        N_features = spec_comb[i]['N_features'], \
                        seed = 200)

    assert np.shape(X_train)[0] == np.shape(y_train)[0]
    assert np.shape(X_test)[0] == np.shape(y_test)[0]

    for j in range(spec_comb[i]['n_trials']): # loop for different trials
        # literally just running iRF
        irf_utils.run_iRF(X_train=X_train,
                              X_test=X_test,
                              y_train=y_train,
                              y_test=y_test,
                              K=spec_comb[i]['n_iter'],
                              n_estimators=spec_comb[i]['n_estimators'],
                              B=spec_comb[i]['n_bootstraps'],
                              random_state_classifier=152,
                              propn_n_samples=spec_comb[i]['propn_n_samples'],
                              bin_class_type=spec_comb[i]['bin_class_type'],
                              M=spec_comb[i]['n_RIT'],
                              max_depth=spec_comb[i]['max_depth'],
                              noisy_split=spec_comb[i]['noisy_split'],
                              num_splits=spec_comb[i]['num_splits'],
                              n_estimators_bootstrap=spec_comb[i]['n_estimators_bootstrap'])
