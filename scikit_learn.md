# Migrating to scikit-learn
The following code will be used when we migrate our
well tested `sandbox` code to our `scikit-learn` fork.

## One-time-only Setup
1. Goto: [here](https://github.com/Yu-Group/scikit-learn-sandbox/blob/master/README.md#git-github-workflow) to fork the scikit-learn-sandbox repo
2. Goto: [here](https://github.com/Yu-Group/scikit-learn-sandbox/blob/master/README.md#basic-setup--installation) to create the conda environments:
  - In particular you should run `make conda_dev2`
  - This will create  `sklearndev2` which is the main environment
    required to run the scikit-learn fork version of iRF
1. Goto: https://github.com/Yu-Group/scikit-learn
2. Fork the repo to your personal github i.e. it will create:
  - https://github.com/**your-gh-username**/scikit-learn
3. `git clone git@github.com:**your-gh-username**/scikit-learn.git`
  - clone the forked repo on your local machine
4. `cd scikit-learn`
  - Now we are ready to set up the remote repos of our team
5. `git remote add upstream git@github.com:Yu-Group/scikit-learn.git`
  - **Yu Group** fork of scikit-learn
5. `git remote add stefanv git@github.com:stefanv/scikit-learn.git`
  - **Stefan's** fork of scikit-learn
5. `git remote add shamindras git@github.com:shamindras/scikit-learn.git`
  - **Shamindra's** fork of scikit-learn
5. `git remote add Runjing-Liu120 git@github.com:Runjing-Liu120/scikit-learn.git`
  - **Bryan's** fork of scikit-learn
7. `git remote add prod https://github.com/scikit-learn/scikit-learn`
  - **Main** scikit-learn repo
8. `git remote -v`
  - This should display the following in the terminal:
  ```
  Runjing-Liu120	git@github.com:Runjing-Liu120/scikit-learn.git (fetch)
  Runjing-Liu120	git@github.com:Runjing-Liu120/scikit-learn.git (push)
  origin	git@github.com:shamindras/scikit-learn.git (fetch)
  origin	git@github.com:shamindras/scikit-learn.git (push)
  prod	https://github.com/scikit-learn/scikit-learn (fetch)
  prod	https://github.com/scikit-learn/scikit-learn (push)
  shamindras	git@github.com:shamindras/scikit-learn.git (fetch)
  shamindras	git@github.com:shamindras/scikit-learn.git (push)
  shifwang	git@github.com:shifwang/scikit-learn.git (fetch)
  shifwang	git@github.com:shifwang/scikit-learn.git (push)
  stefanv	git@github.com:stefanv/scikit-learn.git (fetch)
  stefanv	git@github.com:stefanv/scikit-learn.git (push)
  ```
9. `git fetch shamindras`
  - get all of shamindras committed branches
10. `git checkout feature_weight`
  - Checkout the `feature_weight` that `shamindras` recently committed     (includes commits from `stefanv` and `shifwang`), which we
    just pulled down to our local machine
10. `git merge shamindras/feature_weight`
  - MRG the `feature_weight` that `shamindras` committed, which we
    just pulled down to our local machine

## Regular commit Cycle (in your `scikit-learn` fork)
Assuming you have completed **steps 1-8** above:
1. `source activate sklearndev2`
  - Activate the development conda environment
2. `cd path to/scikit-learn`
  - This should be the directory where your local clone of the `scikit-learn` forked repo is
  - This clone was done in the steps above
2. `git checkout feature_weight`
  - Checkout branch with shamindras changes implemented
2. `pip install -e .`
3. `make cython`
4. Make changes and commit to branch, you can also run `jupyter` notebooks in this mode
5. `make cython`
6. `pip install -e .`
6. `git push -f **your-gh-username** feature_weight`
  - This will create the required pull request (PR)
7. Wait for code review on your PR and then make changes and repeat steps **4-6** onwards
8. You can also `cd ../scikit-learn-sandbox` and run jupyter notebooks now!
  - The `irf_utils` are now in scikit-learn
