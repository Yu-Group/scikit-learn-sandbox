# Migrating to scikit-learn
The following code will be used when we migrate our
well tested `sandbox` code to our `scikit-learn` fork.

## One-time-only Setup
1. Goto: https://github.com/Yu-Group/scikit-learn
2. Fork the repo to your personal github i.e. it will create:
  - https://github.com/**your-gh-username**/scikit-learn
3. `git clone git@github.com:shamindras/scikit-learn.git`
  - clone the forked repo in 2
4. `cd scikit-learn`
5. `git remote add upstream git@github.com:Yu-Group/scikit-learn.git`
6. `git remote add stefanv git@github.com:stefanv/scikit-learn.git`
7. `git remote add prod https://github.com/scikit-learn/scikit-learn`
8. `git remote -v`
  - This should display the following in the terminal:
  ```
origin	git@github.com:shamindras/scikit-learn.git (fetch)
origin	git@github.com:shamindras/scikit-learn.git (push)
prod	https://github.com/scikit-learn/scikit-learn (fetch)
prod	https://github.com/scikit-learn/scikit-learn (push)
stefanv	git@github.com:stefanv/scikit-learn.git (push)
stefanv	git@github.com:stefanv/scikit-learn.git (fetch)
upstream	git@github.com:Yu-Group/scikit-learn.git (fetch)
upstream	git@github.com:Yu-Group/scikit-learn.git (push)
  ```
9. `git fetch stefanv`
10. `git checkout make cython`
11. `make conda_dev0`

## Regular commit Cycle
Assuming you have completed **steps 1-8** above:
1. `source activate sklearndev0`
  - Activate the development conda environment
2. `git checkout git checkout feature_weighted_split`
  - Checkout branch with stefanv's changes implemented
2. `pip install -e .`
3. make cython
4. Make changes and commit to branch
3. make cython
5. `git push -f origin feature_weighted_split`
  - This will create the required pull request (PR)
6. Wait for code review on your PR and then make changes and repeat steps **4** onwards
