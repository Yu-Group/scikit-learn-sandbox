#!/usr/bin/python

# The following is used to draw the random forest decision tree
# graph and display it interactively in the jupyter notebook
from sklearn import tree
import pydotplus
import numpy as np
from sklearn.tree import _tree
from IPython.display import display, Image
import pprint

def draw_tree(inp_tree
              , out_file = None
              , filled=True
              , rounded=True
              , special_characters=True
              , node_ids=True):
    """This will visually display the decision tree in the jupyter notebook
       This is useful for validation purposes of the key metrics collected
       from the decision tree object
    """
    dot_data = tree.export_graphviz(inp_tree
                                    , out_file = out_file
                                    , filled   = filled
                                    , rounded  = rounded
                                    , special_characters = special_characters
                                    , node_ids=node_ids)
    graph = pydotplus.graph_from_dot_data(dot_data)
    img = Image(graph.create_png())
    display(img)

def allTreePaths(dtree, root_node_id = 0):
    """Get all the individual tree paths from root node
       to the leaves
    """

    # Use these lists to parse the tree structure
    children_left  = dtree.tree_.children_left
    children_right = dtree.tree_.children_right

    if root_node_id is None:
        paths      = []

    if root_node_id == _tree.TREE_LEAF:
        raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

    # if left/right is None we'll get empty list anyway
    if children_left[root_node_id] != _tree.TREE_LEAF:
        paths = [np.append(root_node_id, l) for l in allTreePaths(dtree, children_left[root_node_id]) + allTreePaths(dtree, children_right[root_node_id])]

    else:
        paths = [root_node_id]
    return paths

def getTreeData(X_train, dtree, root_node_id = 0):
    """This returns all of the required summary results from an
        individual decision tree
        """

    max_node_depth  = dtree.tree_.max_depth
    n_nodes         = dtree.tree_.node_count
    value           = dtree.tree_.value
    predict         = dtree.tree_.predict

    # Get the total number of features in the training data
    tot_num_features = X_train.shape[1]

    # Get indices for all the features used - 0 indexed and ranging
    # to the total number of possible features in the training data
    all_features_idx = np.array(range(tot_num_features), dtype = 'int64')

    # Get the raw node feature indices from the decision tree classifier attribute
    # positive and negative - we want only non-negative indices for the
    # It is hard to tell which features this came from i.e. indices are zero,
    # corresponding feature columns for consistency in reference
    node_features_raw_idx   = dtree.tree_.feature

    # Get the refined non-negative feature indices for each node
    # Start with a range over the total number of features and
    # subset the relevant indices from the raw indices array
    node_features_idx = all_features_idx[np.array(node_features_raw_idx)]
    #np.array(range(tot_num_features))[all_features_idx]

    # Count the unique number of features used
    num_features_used = (np.unique(node_features_idx)).shape[0]

    # Get all of the paths used in the tree
    all_leaf_node_paths = allTreePaths(dtree = dtree, root_node_id = root_node_id)

    # Get list of leaf nodes
    # In all paths it is the final node value
    all_leaf_nodes = [path[-1] for path in all_leaf_node_paths]

    # Final number of training samples predicted in each class at each leaf node
    all_leaf_node_values = [value[node_id].astype(int) for node_id in all_leaf_nodes]

    # Total number of training samples predicted in each class at each leaf node
    tot_leaf_node_values = [np.sum(leaf_node_values) for leaf_node_values in all_leaf_node_values]

    # All leaf node depths
    # The depth is 0 indexed i.e. root node has depth 0
    leaf_nodes_depths = [np.size(path) - 1 for path in all_leaf_node_paths]

    # Predicted Classes
    # Check that we correctly account for ties in determining the class here
    all_leaf_node_classes = [all_features_idx[np.argmax(value)] for value in all_leaf_node_values]

    # Get all of the features used along the leaf node paths i.e. features used to split a node
    # CHECK: Why does the leaf node have a feature associated with it? Investigate further
    # Removed the final leaf node value so that this feature does not get included currently
    all_leaf_paths_features = [node_features_idx[path[:-1]] for path in all_leaf_node_paths]
    #print(all_features_idx)
    #print(np.array(range(tot_num_features))[all_features_idx])
    #print(np.array(node_features_raw_idx)[all_features_idx])
    #print(np.array(node_features_raw_idx))
    #print(all_features_idx[np.array(node_features_raw_idx)])
    #print(node_features_idx)
    #print(all_leaf_node_paths)

    # Get the unique list of features along a path
    # NOTE: This removes the original ordering of the features along the path
    # The original ordering could be preserved using a special function but will increase runtime
    all_uniq_leaf_paths_features = [np.unique(feature_path) for feature_path in all_leaf_paths_features]

    #print("number of node features", num_features_used, sep = ":\n")
    #print("node feature indices", node_features_idx, sep = ":\n")
    #print("Max node depth in tree", max_node_depth, sep = ":\n")
    #print("number of nodes in tree", n_nodes, sep = ":\n")
    #print("all leaf node depths", leaf_nodes_depths, sep = ":\n")
    #print("all leaf node predicted values", all_leaf_node_values, sep = ":\n")
    #print("total leaf node predicted values", tot_leaf_node_values, sep = ":\n")
    #print("all leaf node predicted classes", all_leaf_node_classes, sep = ":\n")
    #print("all features in leaf node paths", all_leaf_paths_features, sep = ":\n")
    #print("all unique features in leaf node paths", all_uniq_leaf_paths_features, sep = ":\n")

    # Dictionary of all tree values
    tree_data = {"num_features_used" : num_features_used,
                 "node_features_idx" : node_features_idx,
                 "max_node_depth" : max_node_depth,
                 "n_nodes" : n_nodes,
                 "all_leaf_node_paths" : all_leaf_node_paths,
                 "all_leaf_nodes" : all_leaf_nodes,
                 "leaf_nodes_depths" : leaf_nodes_depths,
                 "all_leaf_node_values" : all_leaf_node_values,
                 "tot_leaf_node_values" : tot_leaf_node_values,
                 "all_leaf_node_classes" : all_leaf_node_classes,
                 "all_leaf_paths_features" : all_leaf_paths_features,
                 "all_uniq_leaf_paths_features" : all_uniq_leaf_paths_features}
    return tree_data

def prettyPrintDict(inp_dict, indent_val = 4):
    """This is used to pretty print the dictionary
       this is particularly useful for printing the dictionary of outputs
       from each decision tree
    """
    pp = pprint.PrettyPrinter(indent = indent_val)
    pp.pprint(inp_dict)
