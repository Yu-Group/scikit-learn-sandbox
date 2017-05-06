#!/usr/bin/python

# The following is used to draw the random forest decision tree
# graph and display it interactively in the jupyter notebook
import pydotplus
import pprint
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display, Image
from sklearn.datasets import load_breast_cancer


def generate_rf_example(sklearn_ds=load_breast_cancer(),
                        train_split_propn=0.7, n_estimators=10,
                        random_state_split=2017, random_state_classifier=2018):
    """ This fits a random forest classifier to the breast cancer/ iris datasets
        This can be called from the jupyter notebook so that analysis
        can take place quickly
    """

    # Load the relevant scikit learn data
    raw_data = sklearn_ds

    # Create the train-test datasets
    X_train, X_test, y_train, y_test = train_test_split(
        raw_data.data, raw_data.target, train_size=train_split_propn,
        random_state=random_state_split)

    # Just fit a simple random forest classifier with 2 decision trees
    rf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state_classifier)

    # fit the classifier
    rf.fit(X=X_train, y=y_train)

    return X_train, X_test, y_train, y_test, rf


def draw_tree(decision_tree, out_file=None, filled=True, rounded=False,
              special_characters=True, node_ids=True, max_depth=None,
              feature_names=None, class_names=None, label='all',
              leaves_parallel=False, impurity=True, proportion=False,
              rotate=False):
    """This will visually display the decision tree in the jupyter notebook
       This is useful for validation purposes of the key metrics collected
       from the decision tree object
    """
    dot_data = tree.export_graphviz(decision_tree=decision_tree,
                                    out_file=out_file, filled=filled,
                                    rounded=rounded,
                                    special_characters=special_characters,
                                    node_ids=node_ids,
                                    max_depth=max_depth,
                                    feature_names=feature_names,
                                    class_names=class_names, label=label,
                                    leaves_parallel=leaves_parallel,
                                    impurity=impurity,
                                    proportion=proportion, rotate=rotate)
    graph = pydotplus.graph_from_dot_data(dot_data)
    img = Image(graph.create_png())
    display(img)


def prettyPrintDict(inp_dict, indent_val=4):
    """This is used to pretty print the dictionary
        this is particularly useful for printing the dictionary of outputs
        from each decision tree
        """
    pp = pprint.PrettyPrinter(indent=indent_val)
    pp.pprint(inp_dict)
