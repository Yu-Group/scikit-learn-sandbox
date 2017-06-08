from utils import iRF_benchmarks_lib
from utils import irf_jupyter_utils
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("yaml_file", help="the yaml with specs to input")
parser.add_argument("features", help="csv file with features")
parser.add_argument("responses", help="csv files with responses")
args = parser.parse_args()

# load specs
specs = irf_jupyter_utils.yaml_to_dict(inp_yaml=args.yaml_file)

# load data
features = np.loadtxt(args.features, delimiter=',')
responses = np.loadtxt(args.responses, delimiter=',')

rf_bm = iRF_benchmarks_lib.consolidate_bm_RF(\
            features, responses, specs, seed = None)

iRF_bm = iRF_benchmarks_lib.consolidate_bm_iRF\
    (features, responses, iRF_specs, seed = None)

dict_to_yaml(inp_dict=out_test_dict,
             out_yaml_dir="./model/output",
             out_yaml_name='out_test_new')
