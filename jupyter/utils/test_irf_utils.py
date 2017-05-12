#!/usr/bin/python

from . import irf_jupyter_utils
from . import irf_utils
from sklearn.datasets import load_breast_cancer
import numpy as np
from functools import reduce

# Load the breast cancer dataset
breast_cancer = load_breast_cancer()

# Generate the training and test datasets
X_train, X_test, y_train, \
    y_test, rf = irf_jupyter_utils.generate_rf_example(
        sklearn_ds=breast_cancer, n_estimators=10)

# Get all of the random forest and decision tree data
all_rf_tree_data = \
                   irf_utils.get_rf_tree_data(rf=rf,
                                              X_train=X_train,
                                              y_train=y_train,
                                              X_test=X_test,
                                              y_test=y_test)

# Get the RIT data and produce RITs
np.random.seed(12)
gen_random_leaf_paths = irf_utils.generate_rit_samples(
    all_rf_tree_data=all_rf_tree_data,
    bin_class_type=1)

 # Build single Random Intersection Tree
 # This is not using noisy splits i.e. 5 splits per node
rit0 = irf_utils.build_tree(
    feature_paths=gen_random_leaf_paths,
    max_depth=3,
    noisy_split=False,
    num_splits=5)

# Build single Random Intersection T
# This is using noisy splits i.e. {5, 6} splits per node
rit1 = irf_utils.build_tree(
    max_depth=3,
    noisy_split=True,
    feature_paths=gen_random_leaf_paths,
    num_splits=5)

# Build single Random Intersection Tree of depth 1
# This is not using noisy splits i.e. 5 splits per node
# This should only have a single (root) node
rit2 = irf_utils.build_tree(
    max_depth=1,
    noisy_split=True,
    feature_paths=gen_random_leaf_paths,
    num_splits=5)


# Test that the train test observations sum to the
# total data set observations
def test_generate_rf_example1():

    # Check train test feature split from `generate_rf_example`
    # against the original breast cancer dataset
    assert X_train.shape[0] + X_test.shape[0] \
        == breast_cancer.data.shape[0]

    assert X_train.shape[1] == breast_cancer.data.shape[1]

    assert X_test.shape[1] == breast_cancer.data.shape[1]

    # Check feature and outcome sizes
    assert X_train.shape[0] + X_test.shape[0] \
        == y_train.shape[0] + y_test.shape[0]


# Test build RIT
def test_build_tree():
    assert(len(rit0) <= 1 + 5 + 5**2)
    assert(len(rit1) <= 1 + 6 + 6**2)
    assert(len(rit2) == 1)
