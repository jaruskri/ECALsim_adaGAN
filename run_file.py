


# Imports
import os
import sys
import logging

# GPU = 1      # Or 0,1, 2, 3, etc.
# os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)


from adagan_v6 import *
from func_utils_v6 import *
from data_handler_v6 import EnergData

params = {}

# DATA PARAMS
params['dataset'] = 'energy'                   # 'energy' or 'angle'
params['data_dir'] = '/data/kjarusko/DataEnerg/unzipped/'
# params['data_dir'] = '/eos/home-k/kjarusko/Projects/DataEnerg/unzipped/'   # 20 files
# params['data_dir'] = '/eos/home-k/kjarusko/Projects/DataEnerg/trial/'  # 3 files only
params['data_load2mem'] = False                # load all data in the memory right away (if False, loading and processing file by file)
params['nb_train_files'] = 20                  # int or 'all'
params['keras_dformat'] = 'channels_last'              # 'channels_first' for GPU, 'channels_last' for CPU
params['test_data_portion'] = 0.1              # portion of data to use for testing
params['trial_run'] = True                     # if True, takes only first 20 % of data from each file (run quickly through code to see how it works)

# RESULTS PARAMS
# params['results_dir'] = '/data/kjarusko/adaGAN_gpu_test/run_g10'
params['results_dir'] = 'run_test_v6'

# ADAGAN PARAMS
params['is_bagging'] = False                   # True, False
params['beta_heur'] = 'uniform'                       # 'constant', 'uniform'
params['beta_constant'] = 0.5 # 
params['num_generators'] = 5
params['steps_made'] = 0    # 1

# GAN PARAMS
params['gen_arch'] = 'FloG1_leaky' # FloG1, FloG1_leaky... - start with LeakyReLU, then switch to ReLU
params['disc_arch'] = 'Flo_D'

params['lrate_g'] = 0.0005
params['lrate_d'] = 0.00010     #lr_d should be roughly the same as the ratio from g to d parameters
params['nb_epochs'] = 20
params['percent'] = 100         # how much data to use for validation

params['ReLU_epoch'] = 3                # at ReLU_epoch, switch to ReLU architecture - preferable value 3
params['latent_size'] = 200             # input vector into generator
params['batch_size'] = 128 # 128
params['wtf'] = 6.0                     # weight of true/false loss
params['wa'] = 0.2                      # weight of aux loss
params['we'] = 0.1                      # weight of ecal loss

params['crit_best'] = 'total'           # total, metricp, metrice

params['verbose'] = False

# MIXDISC PARAMS
params['DGAN_epochs'] = 20
params['DGAN_batch_size'] = 128
 

# doplnit dalsi parametry

# Create folder for saving weights, plots, etc.
create_resultdir(params)

logfile = os.path.join(params['results_dir'] + '/Info/adagan.log')
if params['steps_made'] == 0: 
    logmode = 'w'
else:
    logmode = 'a'
# logging.basicConfig(filename=logfile, level=logging.DEBUG)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(logfile, mode=logmode),
        logging.StreamHandler(sys.stdout)
    ]
)


# Initiate AdaGAN, import data, run training.
ada = Adagan(params)
ada.prepare_datainfo(params) # how many data we have, what are the dimensions
ada.train_ensemble(params) # train the ensemble


# if __name__ == '__main__':
#     main()