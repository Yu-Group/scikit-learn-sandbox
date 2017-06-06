# iRF benchmarks
import numpy as np
import time

import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display, Image
from sklearn.datasets import load_breast_cancer

# Import our custom utilities
from imp import reload
from utils import irf_jupyter_utils
from utils import irf_utils


def RF_benchmarks(features, responses,
                  n_trials=10,
                  train_split_propn=0.8,
                  n_estimators=20,
                  seed=2017):
    """
    Run RF benchmarks

    Parameters
    ----------
    n_trials : int, optional (default = 10)
        Number of times to run RF

    train_test_split : float, int, optional (default  = 0.8)
        If float, should be between 0.0 and 1.0 and represent the proportion of
        the dataset to include in the train split. If int, represents the
        absolute number of train samples.

    n_estimators : integer, optional (default=20)
        The number of trees in the forest.

    seed : integer, optional (default = 2017)
        ranodm seed for reproducibility

    Output
    ----------
    metrics_all : dict
        dictionary containing the metrics from all trials

    metrics_summary : dict
        dictionary summarizing the metrics from all trials
        (gives average, standard deviation)

    feature_importances : dict
        dictionary containing feature importances from all trials

    """

    # set seed
    np.random.seed(seed)

    # split into testing and training
    X_train, X_test, y_train, y_test = train_test_split(
        features, responses, train_size=train_split_propn)

    rf_time = np.array([])
    metrics_tmp = {}
    feature_importances = {}
    for i in range(n_trials):
        t0 = time.time()

        # run random forest and time
        rf = RandomForestClassifier(n_estimators=n_estimators)
        rf.fit(X=X_train, y=y_train)
        rf_time = np.append(rf_time, time.time() - t0)

        # get metrics
        metrics_tmp[i] = irf_utils.get_validation_metrics(rf, y_test, X_test)

        # get feature importances
        feature_importances[i] = rf.feature_importances_

    # aggregate metrics
    metrics_all = {}
    for k in metrics_tmp[0].keys():
        metrics_all[k] = [metrics_tmp[i][k] for i in range(n_trials)]

    metrics_all['time'] = rf_time

    # compute summaries of metrics
    metrics_summary = {}

    for k in metrics_all.keys():
        metrics_summary[k] = \
            [np.mean(metrics_all[k], 0), np.std(metrics_all[k], 0)]

    return(metrics_all, metrics_summary, feature_importances)


def iRF_benchmarks(features, responses, n_trials=10,
                   K=5,
                   train_split_propn=0.8,
                   n_estimators=20,
                   B=30,
                   propn_n_samples=.2,
                   bin_class_type=1,
                   M=20,
                   max_depth=5,
                   noisy_split=False,
                   num_splits=2,
                   n_estimators_bootstrap=5,
                   seed=2018):
    """
    Run iRF benchmarks

    Parameters
    ----------
        ...

    Output
    ----------
    metrics_all : dict
        dictionary containing the metrics from all trials

    metrics_summary : dict
        dictionary summarizing the metrics from all trials
        (gives average, standard deviation)

    feature_importances : dict
        dictionary containing feature importances from the Kth forest
        from all trials

    stability_all : dict
        interactions and their respective stability scores from all trials

    """

    # set seed
    np.random.seed(seed)

    # split into testing and training
    X_train, X_test, y_train, y_test = train_test_split(
        features, responses, train_size=train_split_propn)

    iRF_time = np.array([])
    metrics_tmp = {}
    feature_importances = {}
    stability_all = {}

    for i in range(n_trials):
        # run iRF and time
        t0 = time.time()
        _, all_K_iter_rf_data, _, _, stability_score = \
            irf_utils.run_iRF(X_train=X_train,
                              X_test=X_test,
                              y_train=y_train,
                              y_test=y_test,
                              K=K,
                              n_estimators=n_estimators,
                              B=n_estimators,
                              random_state_classifier=None,
                              propn_n_samples=propn_n_samples,
                              bin_class_type=bin_class_type,
                              M=M,
                              max_depth=max_depth,
                              noisy_split=noisy_split,
                              num_splits=num_splits,
                              n_estimators_bootstrap=n_estimators_bootstrap)

        iRF_time = np.append(iRF_time, time.time() - t0)

        # get metrics from last forest
        rf_final = all_K_iter_rf_data['rf_iter' + str(K - 1)]['rf_obj']
        metrics_tmp[i] = irf_utils.get_validation_metrics(
            rf_final, y_test, X_test)

        # get feature importances from last forest
        feature_importances[i] = rf_final.feature_importances_

        # get stability scores
        stability_all[i] = stability_score

    # aggregate metrics
    metrics_all = {}

    for k in metrics_tmp[1].keys():
        metrics_all[k] = [metrics_tmp[i][k] for i in range(n_trials)]

    metrics_all['time'] = iRF_time

    # compute summaries of metrics
    metrics_summary = {}

    for k in metrics_all.keys():
        metrics_summary[k] = \
            [np.mean(metrics_all[k], 0), np.std(metrics_all[k], 0)]

    return(metrics_all, metrics_summary, stability_all, feature_importances)
