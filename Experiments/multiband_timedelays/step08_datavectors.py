import os
import sys
import json
import numpy as np
from importlib import import_module

dirname = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(dirname, 'InferenceRuns'))
sys.path.insert(0, os.path.join(dirname, '../..'))
from Experiments.multiband_timedelays.DataVectors.prep_data_vectors import create_static_data_vectors
import tdc_sampler
import time
import argparse

parser = argparse.ArgumentParser(description="Run model with specific configurations.")
# TODO: feed in config name
parser.add_argument('--config',help="Name of config, stored in InferenceRuns/") # ex: exp0_1_config

parser.add_argument('--use_td_measurements', 
                    help="Use time delay measurements in likelihood calculation.", 
                    action='store_true')                  

args = parser.parse_args()
config_name = args.config
config_module = import_module(config_name)
use_td_measurements = args.use_td_measurements

np.random.seed(config_module.RANDOM_SEED)

td_measurements_file = config_module.td_measurements_file
if use_td_measurements and td_measurements_file is None:
    raise ValueError("If --use_td_measurements is set, --td_measurements_file must be provided.")

static_dv_filepath = config_module.static_dv_file 
likelihood_configs = config_module.likelihood_configs

# Check if static data vectors already exist
if os.path.exists(static_dv_filepath):
    print(f"File {static_dv_filepath} already exists, exiting.")

else:
    print(f"Writing new static data vectors file: {static_dv_filepath}")

    # loop thru each subsample and create static data vectors
    data_vector_dict_list = []
    for subsamp in likelihood_configs.keys():
        #print('Processing ', subsamp)
        input_dict = likelihood_configs[subsamp]
        input_dict['td_measurements_file'] = td_measurements_file
        input_dict['use_td_measurements'] = use_td_measurements

        # TODO: replace args with **input_dict (check all params are there)
        data_vector_dict = create_static_data_vectors(**input_dict)

        # switch numpy arrays to lists for writing to .json
        for key in data_vector_dict.keys():
            if isinstance(data_vector_dict[key], np.ndarray):
                data_vector_dict[key] = data_vector_dict[key].tolist()

        # append to list (one for each likelihood object)
        data_vector_dict_list.append(data_vector_dict)
    print('data vector list length: ', len(data_vector_dict_list))

    # Write list of static data vectors to a JSON file
    if not os.path.exists(os.path.dirname(static_dv_filepath)):
            os.makedirs(os.path.dirname(static_dv_filepath))
            
    with open(static_dv_filepath, 'w') as file:
        json.dump(data_vector_dict_list, file, indent=4)

