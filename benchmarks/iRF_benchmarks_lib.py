# iRF benchmarks
import numpy as np
import time

import matplotlib.pyplot as plt
import os
import yaml

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display, Image
from sklearn.datasets import load_breast_cancer

import sys
sys.path.insert(0, '../jupyter')

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

    rf_bm = {'metrics_all' : metrics_all, 'metrics_summary': metrics_summary,
            'feature_importances': feature_importances}
    return(rf_bm)


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

    iRF_bm = {'metrics_all': metrics_all,
                'metrics_summary': metrics_summary,
                'stability_all': stability_all,
                'feature_importances': feature_importances}

    return(iRF_bm)

def consolidate_bm_RF(features, responses, specs, seed = None):

    np.random.seed(seed)

    # figure out which parameter is being looped over
    # there should only be one parameter to be looped over
    # i.e. only one element of the "specs" dictionary should be a list
    err_1param = 0
    for k in specs.keys():
        if np.max(np.shape([specs[k]])) > 1:
            print(k)
            loop_spec = k
            err_1param += 1

    assert(err_1param <= 1) # should only be one parameter being looped over

    # replicate keys
    if err_1param == 0:
        loop_spec = 'n_trials'
        specs[loop_spec] = list([specs[loop_spec]]) * \
            np.max(np.shape([specs[loop_spec]]))

    n_loops = np.max(np.shape([specs[loop_spec]]))

    print(specs[loop_spec])

    for k in specs.keys():
        if k != loop_spec:
            specs[k] = list([specs[k]]) * n_loops
    print(specs)

    rf_bm = {}

    for i in range(n_loops):
        # subsample data if n parameter is passed
        N = np.shape(features)[0]
        P = np.shape(features)[1]
        if specs['N_obs'][i] != N:
            indx = np.random.choice(N, specs['N_obs'], replace = False)
            features_subset = features[indx, :]
            responses_subset = responses[indx, :]
        else:
            features_subset = features
            responses_subset = responses

        # subsample features if p parameter is passed
        if specs['N_features'][i] != P:
            indx = np.random.choice(P, specs['N_features'], replace = False)
            features_subset = features[:, indx]
            responses_subset = responses[:, indx]
        else:
            features_subset = features
            responses_subset = responses

        rf_bm[i] = RF_benchmarks(features_subset, responses_subset,
                      n_trials=specs['n_trials'][i],
                      train_split_propn=specs['train_split_propn'][i],
                      n_estimators=specs['n_estimators'][i],
                      seed=None)
    return(rf_bm)


def consolidate_bm_iRF(features, responses, specs, seed = None):

    # figure out which parameter is being looped over
    # there should only be one parameter to be looped over
    # i.e. only one element of the "specs" dictionary should be a list
    err_1param = 0
    for k in specs.keys():
        if np.max(np.shape([specs[k]])) > 1:
            print(k)
            loop_spec = k
            err_1param += 1

    assert(err_1param <= 1) # should only be one parameter being looped over

    # replicate keys
    if err_1param == 0:
        loop_spec = 'n_trials'
        specs[loop_spec] = list([specs[loop_spec]]) * \
            np.max(np.shape([specs[loop_spec]]))

    n_loops = np.max(np.shape([specs[loop_spec]]))

    print(specs[loop_spec])

    for k in specs.keys():
        if k != loop_spec:
            specs[k] = list([specs[k]]) * n_loops
    print(specs)

    iRF_bm = {}

    for i in range(n_loops):
        # subsample data if n parameter is passed
        N = np.shape(features)[0]
        P = np.shape(features)[1]
        if specs['N_obs'][i] != N:
            indx = np.random.choice(N, specs['N_obs'], replace = False)
            features_subset = features[indx, :]
            responses_subset = responses[indx, :]
        else:
            features_subset = features
            responses_subset = responses

        # subsample features if p parameter is passed
        if specs['N_features'][i] != P:
            indx = np.random.choice(P, specs['N_features'], replace = False)
            features_subset = features[:, indx]
            responses_subset = responses[:, indx]
        else:
            features_subset = features
            responses_subset = responses

        iRF_bm[i] = iRF_benchmarks(features_subset, responses_subset,
                        n_trials=specs['n_trials'][i],
                           K=specs['n_iter'][i],
                           train_split_propn=specs['train_split_propn'][i],
                           n_estimators=specs['n_estimators'][i],
                           B=specs['n_bootstraps'][i],
                           propn_n_samples=specs['propn_n_samples'][i],
                           bin_class_type=specs['bin_class_type'][i],
                           M=specs['n_RIT'][i],
                           max_depth=specs['max_depth'][i],
                           noisy_split=specs['noisy_split'][i],
                           num_splits=specs['num_splits'][i],
                           n_estimators_bootstrap=specs['n_estimators_bootstrap'][i],
                           seed=seed)
    return(iRF_bm)

def plot_bm(bm, specs, param, metric):
    x = specs[param]
    y = [bm[i]['metrics_summary'][metric][0] \
                for i in range(len(specs[param]))]
    sd = [bm[i]['metrics_summary'][metric][1] \
                for i in range(len(specs[param]))]

    plt.clf()
    plt.errorbar(x, y, yerr = sd)
    plt.xlabel(param )
    plt.ylabel(metric)
    plt.show()


# =============================================================================
# Read in yaml file as a Python dictionary
# =============================================================================


def yaml_to_dict(inp_yaml):
    """ Helper function to read in a yaml file into
        Python as a dictionary

    Parameters
    ----------
    inp_yaml : str
        A yaml text string containing to be parsed into a Python
        dictionary

    Returns
    -------
    out : dict
        The input yaml string parsed as a Python dictionary object
    """
    with open(inp_yaml, 'r') as stream:
        try:
            out = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return out

# =============================================================================
# Convert Python dictionary to yaml
# =============================================================================


def dict_to_yaml(inp_dict, out_yaml_dir, out_yaml_name):
    """ Helper function to convert Python dictionary
        into a yaml string file

    Parameters
    ----------
    inp_dict: dict
        The Python dictionary object to be output as a yaml file

    out_yaml_dir : str
        The output directory for yaml file created

    out_yaml_name : str
        The output filename for yaml file created
        e.g. for 'test.yaml' just set this value to 'test'
             the '.yaml' will be added by the function

    Returns
    -------
    out : str
        The yaml file with specified name and directory from
        the input Python dictionary
    """
    if not os.path.exists(out_yaml_dir):
        os.makedirs(out_yaml_dir)

    out_yaml_path = os.path.join(out_yaml_dir,
                                 out_yaml_name) + '.yaml'

    # Write out the yaml file to the specified path
    with open(out_yaml_path, 'w') as outfile:
        yaml.dump(inp_dict, outfile, default_flow_style=False)
