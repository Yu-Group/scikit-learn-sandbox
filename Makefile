.PHONY : conda_all conda_dev0 conda_dev1 conda_prod0 clean

# Create the various conda environments
conda_all:
	conda env create -f=./conda_envs/sklearndev0/environment.yml
	conda env create -f=./conda_envs/sklearndev1/environment.yml
	conda env create -f=./conda_envs/sklearnprod0/environment.yml

conda_dev0:
	conda env create -f=./conda_envs/sklearndev0/environment.yml

conda_dev1:
	conda env create -f=./conda_envs/sklearndev1/environment.yml

conda_prod0:
	conda env create -f=./conda_envs/sklearnprod0/environment.yml

conda_rem_reinst_prod0:
	conda remove --name sklearnprod0 --all
	conda_prod0

clean:
	rm -rf "#README.md#"
	rm -rf ".#README.md"
	rm -rf "#Makefile#"
	rm -rf "#.Makefile"
