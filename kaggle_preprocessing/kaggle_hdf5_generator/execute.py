from utils.preprocess_utils import * 
from utils.format_labels import *
import pandas as pd

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p","--preprocess")
ap.add_argument("-f","--free")
ap.add_argument("-s","--split")
ap.add_argument("-b","--build")
args = vars(ap.parse_args())

BASE_PATH = "../"
PROCESSED_TRAIN_CSV = "../train_sirna_labels.csv" 
PROCESSED_TEST_CSV = "../test_sirna_labels.csv" 
DST_PROCESSED_TRAIN = "../clean_data/train/"
DST_PROCESSED_TEST = "../clean_data/predict/"
if args["preprocess"]:
    TRAIN = BASE_PATH + "new_train.csv"
    TEST = BASE_PATH + "new_test.csv"
    write_labels_to_csv(TRAIN,TEST)
if["free"]:
    #data = pd.read_csv(PROCESSED_TRAIN_CSV)
    #free_memory(data)
    print("free")

TRAIN_CSV = "./train_split.csv"
VAL_CSV= "./val_split.csv"
TEST_CSV = "./test_split.csv"
SPLITS_LIST = [TRAIN_CSV, VAL_CSV, TEST_CSV]

if args["split"]:
    HUVEC_VAL = 7200 
    HEPG2_VAL= 3400
    RPE_VAL= 3400
    U2OS_VAL= 1500
    VAL_DISTRIBUTION = [HUVEC_VAL,HEPG2_VAL,RPE_VAL,U2OS_VAL]
    TEST_DISTRIBUTION = [x/2 for x in VAL_DISTRIBUTION]
    SIRNA_LABELS= "./train_sirna_labels.csv"
    t = pd.read_csv(SIRNA_LABELS)
    print(t["sirna"].isna().sum())
    splits = split_data(TEST_DISTRIBUTION,VAL_DISTRIBUTION,SIRNA_LABELS,SPLITS_LIST)
if args["build"]:
    TRAIN_HDF5 = "../kernel_data/hdf5/train.hdf5"
    VAL_HDF5 = "../kernel_data/hdf5/val.hdf5"
    TEST_HDF5 = "../kernel_data/hdf5/test.hdf5"
    HDF5_OUTPUTS = [TRAIN_HDF5,VAL_HDF5,TEST_HDF5]
    train_paths, val_paths, test_paths = load_paths(TRAIN_CSV,VAL_CSV,                                                     TEST_CSV)
    paths = [train_paths,val_paths,test_paths]
    #train_labels, val_labels, test_labels= get_labels(paths)
    final_paths, final_labels = get_paths_and_labels(paths)
    #labels = [train_labels, val_labels, test_labels]
    #print(train_labels[0])
    BUILD_DIMS = 300 
    BUILD_CHANELS = 3 
    write_hdf5(final_paths,final_labels, BUILD_DIMS,BUILD_CHANELS,HDF5_OUTPUTS)
