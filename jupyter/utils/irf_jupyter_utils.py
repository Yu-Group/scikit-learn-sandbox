#!/usr/bin/python

import pydotplus
import pprint
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display, Image
from sklearn.datasets import load_breast_cancer


def generate_rf_example(sklearn_ds=load_breast_cancer(),
                        train_split_propn=0.9, n_estimators=3,
                        random_state_split=2017, random_state_classifier=2018):
    """
    This fits a random forest classifier to the breast cancer/ iris datasets
    This can be called from the jupyter notebook so that analysis
    can take place quickly

    Parameters
    ----------
    sklearn_ds : sklearn dataset
        Choose from the `load_breast_cancer` or the `load_iris datasets`
        functions from the `sklearn.datasets` module

    train_split_propn : float
        Should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split.

    n_estimators : int, optional (default=10)
        The index of the root node of the tree. Should be set as default to
        3 and not changed by the user

    random_state_split: int (default=2017)
        The seed used by the random number generator for the `train_test_split`
        function in creating our training and validation sets

    random_state_classifier: int (default=2018)
        The seed used by the random number generator for
        the `RandomForestClassifier` function in fitting the random forest

    Returns
    -------
    X_train : array-like or sparse matrix, shape = [n_samples, n_features]
        Training features vector, where n_samples in the number of samples and
        n_features is the number of features.
    X_test : array-like or sparse matrix, shape = [n_samples, n_features]
        Test (validation) features vector, where n_samples in the
        number of samples and n_features is the number of features.
    y_train : array-like or sparse matrix, shape = [n_samples, n_classes]
        Training labels vector, where n_samples in the number of samples and
        n_classes is the number of classes.
    y_test : array-like or sparse matrix, shape = [n_samples, n_classes]
        Test (validation) labels vector, where n_samples in the
        number of samples and n_classes is the number of classes.
    rf : RandomForestClassifier object
        The fitted random forest to the training data

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> X_train, X_test, y_train, y_test,
        rf = generate_rf_example(sklearn_ds =
                                load_breast_cancer())
    >>> print(X_train.shape)
    ...                             # doctest: +SKIP
    ...
    (512, 30)
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
    """
    This will visually display the decision tree in the jupyter notebook
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


def pretty_print_dict(inp_dict, indent_val=4):
    """
    This is used to pretty print the dictionary
    this is particularly useful for printing the dictionary of outputs
    from each decision tree
    """
    pp = pprint.PrettyPrinter(indent=indent_val)
    pp.pprint(inp_dict)
