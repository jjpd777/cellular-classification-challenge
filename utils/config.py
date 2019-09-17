import json
import os
# define the paths to the images directory
IMAGES_PATH = "./clean_data/train/"

RESIZE = 128
NUM_CLASSES = 1108
NUM_CHANNELS = 6

EPOCHS = 80 
BATCH_SIZE = 32 
LEARNING_RATE = 0.01
POWER = 2.5
MOMENTUM = 0
DECAY = LEARNING_RATE/EPOCHS
FCH1 =64 
FCH2 = 128
NETWORK_REG = 0.008
STAGES = (1,1)
FILTERS = (64,64,64)

TRAIN_HDF5 = "./clean_data/hdf5/train.hdf5"
VAL_HDF5 = "./clean_data/hdf5/val.hdf5"
TEST_HDF5 = "./clean_data/hdf5/test.hdf5"
HDF5_FILES = [TRAIN_HDF5,VAL_HDF5,TEST_HDF5]
PARAMS = "parameters.txt"
EXP_NUM = "experiment-4/"
EXPERIMENT_NAME = "./output/" + EXP_NUM
LOG_NAME = EXPERIMENT_NAME + "console.log" 
CHECKPOINTS = EXPERIMENT_NAME + "checkpoints"
PARAMS_FILE = EXPERIMENT_NAME + PARAMS
MODEL_PATH = EXPERIMENT_NAME + "resnet.model"
OUTPUT_PATH = EXPERIMENT_NAME

def make_experiment():
    os.mkdir(EXPERIMENT_NAME)
    os.mkdir(CHECKPOINTS)
    os.mknod(LOG_NAME)

def store_params():
    data= {}
    data['hyperparameters'] = []
    data['hyperparameters'].append({
        'image_size' : RESIZE,
        'epochs' : EPOCHS,
        'batch_size' : BATCH_SIZE,
        'learning_rate' : LEARNING_RATE,
        'decay' : DECAY,
        'EXP_NUM' : 2.5,
        'network_reg': NETWORK_REG
        })
    with open(PARAMS_FILE,'w') as write:
        json.dump(data,write)
