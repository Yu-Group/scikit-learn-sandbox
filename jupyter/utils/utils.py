#!/usr/bin/python

# The following is used to draw the random forest decision tree
# graph and display it interactively in the jupyter notebook
import pydotplus
import numpy as np
import pprint
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import _tree
from IPython.display import display, Image
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

def generate_rf_example(sklearn_ds = load_breast_cancer()
                        , train_split_propn = 0.7
                        , n_estimators = 10
                        , random_state_split = 2017
                        , random_state_classifier = 2018):
    """ This fits a random forest classifier to the breast cancer/ iris datasets
        This can be called from the jupyter notebook so that analysis
        can take place quickly
    """

    # Load the relevant scikit learn data
    raw_data = sklearn_ds

    # Create the train-test datasets
    X_train, X_test, y_train, y_test = train_test_split(raw_data.data
                                                        , raw_data.target
                                                        , train_size = train_split_propn
                                                        , random_state = random_state_split)

    # Just fit a simple random forest classifier with 2 decision trees
    rf = RandomForestClassifier(n_estimators = n_estimators
                                , random_state = random_state_classifier)

    # fit the classifier
    rf.fit(X = X_train, y = y_train)

    return X_train, X_test, y_train, y_test, rf

def draw_tree(decision_tree
              , out_file = None
              , filled = True
              , rounded = False
              , special_characters = True
              , node_ids = True
              , max_depth = None
              , feature_names = None
              , class_names = None
              , label = 'all'
              , leaves_parallel = False
              , impurity = True
              , proportion = False
              , rotate = False):

    """This will visually display the decision tree in the jupyter notebook
       This is useful for validation purposes of the key metrics collected
       from the decision tree object
    """
    dot_data = tree.export_graphviz(decision_tree = decision_tree
                                    , out_file = out_file
                                    , filled = filled
                                    , rounded = rounded
                                    , special_characters = special_characters
                                    , node_ids = node_ids
                                    , max_depth = max_depth
                                    , feature_names = feature_names
                                    , class_names = class_names
                                    , label = label
                                    , leaves_parallel = leaves_parallel
                                    , impurity = impurity
                                    , proportion = proportion
                                    , rotate = rotate)
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
        paths = [np.append(root_node_id, l)
                 for l in allTreePaths(dtree, children_left[root_node_id]) +
                          allTreePaths(dtree, children_right[root_node_id])]

    else:
        paths = [root_node_id]
    return paths


def getValidationMetrics(inp_class_reg_obj, y_true, X_test):
    """ Get the various Random Forest/ Decision Tree metrics
        This is currently setup only for classification forests and trees
        TODO/ CHECK: We need to update this for regression purposes later
        TODO/ CHECK: We need validation for the input object being a forest
               or tree classifier/ regression object
        TODO/ CHECK: For classification we need to validate that the maximum number of
               labels is 2 for the training/ testing data
    """

    # Get the predicted values on the validation data
    y_pred = inp_class_reg_obj.predict(X = X_test)

    ### CLASSIFICATION metrics calculations

    # Cohen’s kappa: a statistic that measures inter-annotator agreement.
    #cohen_kappa_score = metrics.cohen_kappa_score(y1, y2[, labels, ...])

    # Compute Area Under the Curve (AUC) using the trapezoidal rule
    #fpr, tpr, thresholds = metrics.roc_curve(y_true = y_true, y_pred = y_pred)
    #auc = metrics.auc(fpr, tpr)

    # Compute average precision (AP) from prediction scores
    #average_precision_score = metrics.average_precision_score(y_true = y_true, y_score)

    # Compute the Brier score.
    #metrics.brier_score_loss(y_true = y_true, y_prob[, ...])

    # Compute the F-beta score
    #metrics.fbeta_score(y_true = y_true, y_pred = y_pred, beta[, ...])

    # Average hinge loss (non-regularized)
    #metrics.hinge_loss(y_true = y_true, pred_decision[, ...])

    # Compute the Matthews correlation coefficient (MCC) for binary classes
    #metrics.matthews_corrcoef(y_true = y_true, y_pred[, ...])

    # Compute precision-recall pairs for different probability thresholds
    #metrics.precision_recall_curve(y_true = y_true, ...)

    # Compute precision, recall, F-measure and support for each class
    #metrics.precision_recall_fscore_support(...)

    # Compute Area Under the Curve (AUC) from prediction scores
    #metrics.roc_auc_score(y_true = y_true, y_score[, ...])

    # Compute Receiver operating characteristic (ROC)
    #metrics.roc_curve(y_true = y_true, y_score[, ...])

    # Compute the F1 score, also known as balanced F-score or F-measure
    f1_score = metrics.f1_score(y_true = y_true, y_pred = y_pred)

    # Compute the average Hamming loss.
    hamming_loss = metrics.hamming_loss(y_true = y_true, y_pred = y_pred)

    # Jaccard similarity coefficient score
    jaccard_similarity_score = metrics.jaccard_similarity_score(y_true = y_true, y_pred = y_pred)

    # Log loss, aka logistic loss or cross-entropy loss.
    log_loss = metrics.log_loss(y_true = y_true, y_pred = y_pred)

    # Compute the precision
    precision_score = metrics.precision_score(y_true = y_true, y_pred = y_pred)

    # Compute the recall
    recall_score = metrics.recall_score(y_true = y_true, y_pred = y_pred)

    # Accuracy classification score
    accuracy_score = metrics.accuracy_score(y_true = y_true, y_pred = y_pred)

    # Build a text report showing the main classification metrics
    classification_report = metrics.classification_report(y_true = y_true, y_pred = y_pred)

    # Compute confusion matrix to evaluate the accuracy of a classification
    confusion_matrix = metrics.confusion_matrix(y_true = y_true, y_pred = y_pred)

    # Zero-one classification loss.
    zero_one_loss = metrics.zero_one_loss(y_true = y_true, y_pred = y_pred)

    # Load all metrics into a single dictionary
    classification_metrics = {"hamming_loss" : hamming_loss,
                              "log_loss" : log_loss,
                              "recall_score" : recall_score,
                              "precision_score" : precision_score,
                              "accuracy_score" : accuracy_score,
                              "f1_score" : f1_score,
                              "classification_report" : classification_report,
                              "confusion_matrix" : confusion_matrix,
                              "zero_one_loss" : zero_one_loss}

    return classification_metrics

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

    # Get the unique list of features along a path
    # NOTE: This removes the original ordering of the features along the path
    # The original ordering could be preserved using a special function but will increase runtime
    all_uniq_leaf_paths_features = [np.unique(feature_path) for feature_path in all_leaf_paths_features]

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
