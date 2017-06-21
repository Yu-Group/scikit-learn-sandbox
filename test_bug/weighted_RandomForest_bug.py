import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
import numpy as np
from functools import reduce

# Import our custom utilities
from imp import reload
#from utils import irf_jupyter_utils
from sklearn.tree import irf_utils

# Import RF related functions
from sklearn.ensemble import RandomForestClassifier

# draw data
n_samples = 2000
n_features = 2
random_state_classifier = 2018
np.random.seed(random_state_classifier)
X_train = np.random.uniform(low=0, high=1, size=(n_samples, n_features))
y_train = np.random.choice([0, 1], size=(n_samples,), p=[.5, .5])
X_test = np.random.uniform(low=0, high=1, size=(n_samples, n_features))
y_test = np.random.choice([0, 1], size=(n_samples,), p=[.5, .5])

# fit the classifier
rf = RandomForestClassifier(
    n_estimators=20, random_state=random_state_classifier)
#feature_weight = [.6, .4]
feature_weight = [.1, .9]
rf.fit(X=X_train, y=y_train, feature_weight=feature_weight)

# extract features
"""for idx, dtree in enumerate(rf.estimators_):
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
"""
print('finished')
