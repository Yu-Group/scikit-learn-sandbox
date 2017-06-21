#!/usr/bin/env Rscript

library(yaml)
source('R_irf_benchmarks_utils.R')

# Global variables
PRINT_SEP = ': '
CSV_DELIMETER = ','
MODEL_SPECS_DIR = './specs'
DATA_DIR = './data'
OUTPUT_DIR = './output'

if(!dir.exists(DATA_DIR) | !dir.exists(OUTPUT_DIR)){
  stop('Please check you are running this from the benchmarks directory')
  }

# input name of yaml file with model specs
specs_name <- commandArgs(trailingOnly = TRUE)

# input specs
# yaml.load_file('./specs/iRF_mod01.yaml')
spec_path <- paste(MODEL_SPECS_DIR, '/', specs_name, '.yaml', sep = '')
specs <- yaml.load_file(spec_path)
print(spec_path)
#print(specs)

# Get the raw features and responses data
inp_dsname <- specs$inp_dsname
features_path <- paste(DATA_DIR, '/', inp_dsname, "_features.csv", sep = '')
responses_path <- paste(DATA_DIR, '/', inp_dsname, "_responses.csv", sep = '')
print(features_path)
print(responses_path)

# load data
features <- read.csv(features_path, header = FALSE)
responses <- read.csv(responses_path, header = FALSE)
features <- as.matrix(features)
responses <- as.factor(responses$V1)

# make columns 0-indexed to match with python
colnames(features) <- paste('X', 1:dim(features)[2] - 1, sep = '')

# output name of the dataset
irf_bm <- consolidate_bm_iRF(features, responses, specs,
                seed_classifier = 2001, seed_data_split = 24)

# save output
# output name of the dataset
output_path <- paste(OUTPUT_DIR, '/', specs_name, "_out.RData", sep = '')
print(output_path)
save(irf_bm, file=output_path)
