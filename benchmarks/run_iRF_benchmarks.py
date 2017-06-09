import iRF_benchmarks_lib
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("yaml_file", help="the yaml with specs to input")
parser.add_argument("features", help="csv file with features")
parser.add_argument("responses", help="csv files with responses")
parser.add_argument("output_dir", help = "specify output directory")
parser.add_argument("output_name", help = "specify output name")

args = parser.parse_args()

# load specs
specs = iRF_benchmarks_lib.yaml_to_dict(inp_yaml=args.yaml_file)

# load data
features = np.loadtxt(args.features, delimiter=',')
responses = np.loadtxt(args.responses, delimiter=',')

rf_bm = iRF_benchmarks_lib.consolidate_bm_iRF(\
            features, responses, specs, seed = None)

iRF_benchmarks_lib.dict_to_yaml(inp_dict=rf_bm,
             out_yaml_dir= args.output_dir,
             out_yaml_name=args.output_name)
