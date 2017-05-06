#!/usr/bin/python

from . import irf_utils
import nose
import pydotplus
import numpy as np

from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import _tree
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

# Check that the train test observations sum to the total data set observations
def test_generate_rf_example1():
    X_train, X_test, y_train, y_test, rf = utils.generate_rf_example(sklearn_ds = load_breast_cancer()
                                                                     , train_split_propn = 0.9
                                                                     , n_estimators = 3
                                                                     , random_state_split = 2017
                                                                     , random_state_classifier = 2018)

    breast_cancer = load_breast_cancer()

    # Check against the original dataset
    assert X_train.shape[0] + X_test.shape[0] == breast_cancer.data.shape[0]

    # Check feature and outcome sizes
    assert X_train.shape[0] + X_test.shape[0] == y_train.shape[0] + y_test.shape[0]

def test_getTreeData():
    X_train, X_test, y_train, y_test, rf = utils.generate_rf_example(sklearn_ds = load_breast_cancer()
                                                                     , train_split_propn = 0.9
                                                                     , n_estimators = 3
                                                                     , random_state_split = 2017
                                                                     , random_state_classifier = 2018)
