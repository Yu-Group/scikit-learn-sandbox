# scikit-learn-sandbox
This is a sandbox repo to test out various scikit-learn features that will be used in the development of the iterative Random Forests (iRF) implementation.

It will eventually be **deprecated** once the iRF implementation is completed. It will be useful for us to do setup work and quick general scikit learn experiments.

## Basic Setup & Installation

### Installing the `conda` environments

Firstly you need to [install Anaconda](https://www.continuum.io/downloads) on your computer

First fork and then clone the repo locally on your computer. Change directory to the repo folder.

To install the 3 conda environments just use the `Makefile` as shown below.

To create **all 3 environments** use:

```bash
make conda_all
```

Or to install them **individually** you can run the following commands separately

```bash
make conda_dev0
make conda_dev1
make conda_prod0
```

To confirm that the conda environments have installed correctly you can run the following in the terminal:
```bash
conda info -e
```

You should now see the 3 installed environments listed as required `sklearndev0`, `sklearndev1` and `sklearnprod0`.

To **activate** the conda environments simply use conda as usual i.e. `source activate sklearndev0`

### Quick info on each of the conda environments

`sklearndev0` : Is built to mirror [scikit learn dev requirements.txt](https://github.com/scikit-learn/scikit-learn/blob/master/build_tools/appveyor/requirements.txt). Useful for our scikit-learn development.

`sklearndev1` : Same as `sklearndev0` but includes pandas and jupyter notebook for additional interactive testing

`sklearnprod0`: This contains the latest conda **production** `scikit-learn` build and is useful for current production testing of `sklearn`

## Git-Github Workflow

We can use the following workflow for our `pull-commit-review-merge` cycle:

### Git-Github - First time only setup

1. **First time only** Fork this repo through github i.e. this will create https://github.com/**your-username-here**/scikit-learn-sandbox
2. On your local computer projects directory clone your fork i.e. `git clone git@github.com:**your-username-here**/scikit-learn-sandbox.git`
3. Access the cloned repo: `cd scikit-learn-sandbox`
4. Set the upstream remote: `git remote add upstream https://github.com/Yu-Group/scikit-learn-sandbox`
5. Check the remotes: `git remote -v`. You should see:

```bash
origin	git@github.com:**your-username-here**/scikit-learn-sandbox.git (fetch)
origin	git@github.com:**your-username-here**/scikit-learn-sandbox.git (push)
upstream	git@github.com:Yu-Group/scikit-learn-sandbox.git (fetch)
upstream	git@github.com:Yu-Group/scikit-learn-sandbox.git (push)
```

### Git-Github - Typical Workflow

With the one-off setup complete we are ready to start coding! The main rule to remember is:

*Never commit/ merge new code **directly to master**!*

The typical workflow is as follows:

1. [Create](https://github.com/Yu-Group/scikit-learn-sandbox/issues) a **github issue** for every task e.g. [example](https://github.com/Yu-Group/scikit-learn-sandbox/issues/19).

* Add as many helpful links and details as possible.
* This is our main form of assigned work documentation - so use judgement on what to include.
* Typically I begin each issue with "FIX:"

e.g. **FIX: Explore Binary Tree Traversal** *Insert Issue details here*

2. Assign a person to the issue e.g. shamindras, kkumbier etc
3. Locally update master i.e.:

```bash
git checkout master
git pull upstream master  # update local master
git push -f origin master # update origin master
```

5. Create a **new branch** for the issue: `git checkout -b issue-**issue-number**-**short-description**`

e.g. `issue-17-binary-tree-traverse` is one such typical branch name. Note you are now checked into the new branch and ready to go!
6. Do your great coding here :). Commit regularly with helpful messages e.g.:

*FIX: Issue #15, create first draft of all features in decision path to the leaf node predicted leaf node values and depth of the leaf node*

7. Once you are ready to send a pull request you just commit and then run:

```bash
git checkout master
git pull upstream master  # update local master
git checkout issue-**issue-number**-**short-description** # go back to working branch
git rebase master # sync branch with upstream master
```

8. Fix any merge conflicts and then commit branch
9. You can now commit the branch as `git push -f origin issue-**issue-number**-**short-description**`
10. In the [upstream github page](https://github.com/Yu-Group/scikit-learn-sandbox) you will see the pull request.
11. Ask the reviewer to review as required. For all code review changes requested, just repeat **step 7** onwards until reviewer is satisfied
12. Once all changes are put in - the reviewer can merge the changes in upstream master
13. Then start a new issue from **Step 1** onwards!