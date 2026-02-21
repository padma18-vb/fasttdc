import os
import sys
import json
import numpy as np
from importlib import import_module

dirname = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(dirname, 'InferenceRuns'))
sys.path.insert(0, os.path.join(dirname, '../..'))
from Experiments.roman_forecast.DataVectors.prep_data_vectors import create_static_data_vectors
import tdc_sampler
import time
import argparse

parser = argparse.ArgumentParser(description="Run model with specific configurations.")
# TODO: feed in config name
parser.add_argument('--config',help="Name of config, stored in InferenceRuns/") # ex: exp0_1_config
args = parser.parse_args()
config_name = args.config
config_module = import_module(config_name)

np.random.seed(config_module.RANDOM_SEED)

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
        print('Processing ', subsamp)
        input_dict = likelihood_configs[subsamp]

        # TODO: replace args with **input_dict (check all params are there)
        data_vector_dict = create_static_data_vectors(**input_dict)

        # switch numpy arrays to lists for writing to .json
        for key in data_vector_dict.keys():
            if isinstance(data_vector_dict[key], np.ndarray):
                data_vector_dict[key] = data_vector_dict[key].tolist()

        # append to list (one for each likelihood object)
        data_vector_dict_list.append(data_vector_dict)

    # Write list of static data vectors to a JSON file
    with open(static_dv_filepath, 'w') as file:
        json.dump(data_vector_dict_list, file, indent=4)



['l01507' 'l01429' 'l01063' 'l00899' 'l01777' 'l01809' 'l02258' 'l02261'
 'l02379' 'l01886' 'l01847' 'l01411' 'l00475' 'l01980' 'l02113' 'l02570'
 'l01466' 'l01084' 'l02265' 'l00632' 'l00080' 'l00636' 'l02420' 'l01558'
 'l00113' 'l00991' 'l01954' 'l00223' 'l00814' 'l01324' 'l00877' 'l01195'
 'l02642' 'l02045' 'l01548' 'l02124' 'l01895' 'l01006' 'l00649' 'l01138'
 'l00656' 'l02544' 'l00634' 'l01511' 'l00521' 'l02416' 'l02273' 'l00698'
 'l01649' 'l01762' 'l02599' 'l02481' 'l02084' 'l00809' 'l02342' 'l01353'
 'l01826' 'l02565' 'l01668' 'l02279' 'l01859' 'l00398' 'l02391' 'l02067'
 'l00242' 'l00197' 'l01916' 'l00181' 'l01550' 'l01801' 'l01033' 'l00484'
 'l00911' 'l00214' 'l02246' 'l02062' 'l00102' 'l00851' 'l01296' 'l02567'
 'l01719' 'l01994' 'l02211' 'l00573' 'l02525' 'l01989' 'l00163' 'l01005'
 'l00689' 'l00500' 'l01873' 'l01908' 'l01155' 'l00123' 'l01747' 'l00846'
 'l01250' 'l01422' 'l00542' 'l00791' 'l01714' 'l01168' 'l00842' 'l02359'
 'l00043' 'l02510' 'l02515' 'l01927' 'l01935' 'l00345' 'l00651' 'l01092'
 'l00629' 'l00554' 'l00005' 'l01769' 'l01707' 'l01423' 'l01038' 'l01346'
 'l00785' 'l02466' 'l01554' 'l00292' 'l00887' 'l01239' 'l00103' 'l02578'
 'l02069' 'l01137' 'l00205' 'l02404' 'l00474' 'l02625' 'l02171' 'l00918'
 'l00891' 'l01807' 'l00603' 'l00567' 'l00959' 'l02562' 'l00339' 'l00107'
 'l00734' 'l00216' 'l02613' 'l02619' 'l01255' 'l00067' 'l01536' 'l01505'
 'l02244' 'l00270' 'l01044' 'l01638' 'l01393' 'l02210' 'l00420' 'l00537'
 'l02468' 'l00176' 'l02179' 'l01076' 'l02311' 'l01587' 'l01332' 'l02486'
 'l01455' 'l02100' 'l01644' 'l01116' 'l02047' 'l02245' 'l01122' 'l00327'
 'l00358' 'l01962' 'l01585' 'l00097' 'l01162' 'l02345' 'l00186' 'l02118'
 'l02247' 'l01532' 'l02569' 'l01448' 'l01586' 'l00934' 'l01216' 'l02553'
 'l00137' 'l00489' 'l01748' 'l01900' 'l01969' 'l00672' 'l00119' 'l00299'
 'l02046' 'l02545' 'l02142' 'l02490' 'l01310' 'l00609' 'l02444' 'l01613'
 'l01389' 'l00821' 'l00426' 'l02399' 'l00328' 'l00388' 'l02482' 'l00862'
 'l01042' 'l00237' 'l02533' 'l00616' 'l01313' 'l01416' 'l00972' 'l02417'
 'l00156' 'l02491' 'l00569' 'l01595' 'l01192' 'l01104' 'l00544' 'l00966'
 'l01953' 'l00175' 'l01202' 'l01447' 'l01426' 'l02114' 'l01941' 'l01077'
 'l01783' 'l01601' 'l01833' 'l02255' 'l02293' 'l01030' 'l00373' 'l02253'
 'l01775' 'l01818' 'l00406' 'l02602' 'l01047' 'l02074' 'l00243' 'l02483'
 'l00796' 'l00939' 'l02122' 'l00380' 'l02592' 'l01517' 'l02019' 'l00277'
 'l02301' 'l02622' 'l00166' 'l02096' 'l01343' 'l01617' 'l02335' 'l00857'
 'l01852' 'l00532' 'l01381' 'l01861' 'l01150' 'l01335' 'l01397' 'l02321'
 'l00021' 'l02097' 'l00154' 'l01188' 'l02035' 'l00811' 'l00946' 'l01018'
 'l02487' 'l02595' 'l00333' 'l00101' 'l02276' 'l00194' 'l02498' 'l01114'
 'l02266' 'l00852' 'l00597' 'l00034' 'l00274' 'l01696' 'l02647' 'l01610'
 'l01976' 'l00457' 'l00282' 'l02453' 'l01221' 'l01312' 'l00625' 'l00559'
 'l02456' 'l01321' 'l01013' 'l01703' 'l02329' 'l00399' 'l01646' 'l01760'
 'l02166' 'l01452' 'l02259' 'l01031' 'l00683' 'l01993' 'l01624' 'l01093'
 'l00121' 'l02010' 'l01740' 'l00509' 'l01677' 'l01727' 'l02192' 'l00712'
 'l00061' 'l02026' 'l01194' 'l00258' 'l00841' 'l02346' 'l01730' 'l01712'
 'l01632' 'l01100' 'l01118' 'l00180' 'l01570' 'l02607' 'l00548' 'l01616'
 'l00976' 'l00913' 'l00306' 'l00703' 'l02155' 'l01160' 'l00161' 'l02460']