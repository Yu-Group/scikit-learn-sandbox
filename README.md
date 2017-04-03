# scikit-learn-sandbox
This is a sandbox repo to test out various scikit-learn features that will be used in the development of the iterative Random Forests (iRF) implementation.

It will eventually be **deprecated** once the iRF implementation is completed. It will be useful for us to do setup work and quick general scikit learn experiments.

## Basic Setup

### Installing the `conda` environments

Firstly you need to [install Anaconda](https://www.continuum.io/downloads) on your computer

First fork and then clone the repo locally on your computer. Change directory to the repo folder.

To install the 3 conda environments just use the `Makefile` as shown below.

To create **all 3 environments** use:

```
make conda_all
```

Or to install them **individually** you can run the following commands separately

```
make conda_dev0
make conda_dev1
make conda_prod0
```

### Quick info on each of the conda environments

`sklearndev0` : Is built to [scikit learn dev requirements.txt](https://github.com/scikit-learn/scikit-learn/blob/master/build_tools/appveyor/requirements.txt). Useful for our scikit-learn development.

`sklearndev1` : Same as `sklearndev0` but includes pandas and jupyter notebook for additional interactive testing

`sklearnprod0`: This contains the latest conda **production** `scikit-learn` build and is useful for current production testing of `sklearn`
