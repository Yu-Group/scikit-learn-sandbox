import py_irf_benchmarks
import numpy as np
import argparse
import os

# Global variables
PRINT_SEP = ': '
CSV_DELIMETER = ','
MODEL_SPECS_DIR = './specs'
DATA_DIR = './data'
OUTPUT_DIR = './output'

# Validate that user is running in the benchmarks folderc
py_irf_benchmarks.check_path_exists(DATA_DIR)
py_irf_benchmarks.check_path_exists(OUTPUT_DIR)

# Get the model specs parameters file reference
parser = argparse.ArgumentParser()
parser.add_argument("fname_yaml_specs",
                    help="the yaml with specs to input")
args = parser.parse_args()
print('fname_yaml_specs', args.fname_yaml_specs, sep=PRINT_SEP)

# load specs
inp_fpath_yaml_specs = os.path.join(MODEL_SPECS_DIR,
                                    args.fname_yaml_specs + '.yaml')
py_irf_benchmarks.check_path_exists(inp_fpath_yaml_specs)
print('inp_fpath_yaml_specs', inp_fpath_yaml_specs, sep=PRINT_SEP)

# Read the model specs into a Python dictionary
inp_specs = py_irf_benchmarks.yaml_to_dict(inp_yaml=inp_fpath_yaml_specs)

# Get the raw data name from the model specs
# We will use this to name the output results yaml file later
inp_dsname = inp_specs['inp_dsname']
print('inp_dsname', inp_dsname, sep=PRINT_SEP)

# Get the raw features and responses data
features_csv = os.path.join(DATA_DIR, inp_dsname + "_features.csv")
responses_csv = os.path.join(DATA_DIR, inp_dsname + "_responses.csv")
py_irf_benchmarks.check_path_exists(features_csv)
py_irf_benchmarks.check_path_exists(responses_csv)
print('features_csv', features_csv, sep=PRINT_SEP)
print('responses_csv', responses_csv, sep=PRINT_SEP)

# load data
features = np.loadtxt(features_csv, delimiter=CSV_DELIMETER)
responses = np.loadtxt(responses_csv, delimiter=CSV_DELIMETER)
print(features[0:10])
print(responses[0:10])

# output name of the dataset
output_name = args.fname_yaml_specs + "_out.yaml"
print(output_name)

#rf_bm = py_irf_benchmarks.consolidate_bm_iRF(\
    #    features, responses, specs, seed = None)

#py_irf_benchmarks.dict_to_yaml(inp_dict=rf_bm,
#out_yaml_dir= OUTPUT_DIR,
#out_yaml_name=args.output_name)
