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
gen_random_leaf_paths = irf_utils._generate_rit_samples(
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

# Get the entire RIT data
all_rit_tree_data = irf_utils.get_rit_tree_data(
    all_rf_tree_data=all_rf_tree_data,
    bin_class_type=1,
    random_state=12,
    n_estimators=10,
    max_depth=3,
    noisy_split=False,
    num_splits=2)

# Manually construct an RIT example
# Get the unique feature paths where the leaf
# node predicted class is just 1
# We are just going to get it from the first decision tree
# for this test case
uniq_feature_paths \
    = all_rf_tree_data['dtree0']['all_uniq_leaf_paths_features']
leaf_node_classes \
    = all_rf_tree_data['dtree0']['all_leaf_node_classes']
ones_only \
    = [i for i, j in zip(uniq_feature_paths, leaf_node_classes)
       if j == 1]

## Manually extract the last seven values for our example
# Just pick the last seven cases
# we are going to manually construct
# We are going to build a BINARY RIT of depth 3
# i.e. max `2**3 -1 = 7` intersecting nodes
ones_only_seven = ones_only[-7:]

# Manually build the RIT
# Construct a binary version of the RIT manually!
node0 = ones_only_seven[0]
node1 = np.intersect1d(node0, ones_only_seven[1])
node2 = np.intersect1d(node1, ones_only_seven[2])
node3 = np.intersect1d(node1, ones_only_seven[3])
node4 = np.intersect1d(node0, ones_only_seven[4])
node5 = np.intersect1d(node4, ones_only_seven[5])
node6 = np.intersect1d(node4, ones_only_seven[6])

intersected_nodes_seven \
    = [node0, node1, node2, node3, node4, node5, node6]

leaf_nodes_seven = [node2, node3, node5, node6]

rit_output \
    = reduce(np.union1d, (node2, node3, node5, node6))

# Now we can create the RIT using our built irf_utils
# build the generator of 7 values
ones_only_seven_gen = (n for n in ones_only_seven)

# Build the binary RIT using our irf_utils
rit_man0 = irf_utils.build_tree(
    feature_paths=ones_only_seven_gen,
    max_depth=3,
    noisy_split=False,
    num_splits=2)

# Calculate the union values

# First on the manually constructed RIT
rit_union_output_manual \
    = reduce(np.union1d, (node2, node3, node5, node6))

# Lastly on the RIT constructed using a function
rit_man0_union_output \
    = reduce(np.union1d, [node[1]._val
                          for node in rit_man0.leaf_nodes()])

# Test the manually constructed binary RIT
def test_manual_binary_RIT():
    # Check all node values
    assert [node[1]._val.tolist()
            for node in rit_man0.traverse_depth_first()] \
                == [node.tolist()
                    for node in intersected_nodes_seven]

    # Check all leaf node intersected values
    assert [node[1]._val.tolist()
            for node in rit_man0.leaf_nodes()] ==\
                [node.tolist() for node in leaf_nodes_seven]

    # Check the union value calculation
    assert rit_union_output_manual.tolist()\
        == rit_man0_union_output.tolist()

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
