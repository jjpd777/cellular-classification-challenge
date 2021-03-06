from utils import AspectAwarePreprocessor
from utils import HDF5DatasetWriter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import random
import progressbar
from tqdm import tqdm
import pandas as pd
import csv
import json
import cv2
import os
from glob import glob
from pathlib import Path


def select_sister_sirna(original_data,val_test_subset):
    sister_sirna = []
    [original_data.remove(x) for x in val_test_subset]
    for image in val_test_subset:
       substring = image[:-6]
       for og in original_data:
           if substring in og:
               sister_sirna.append(og)
    result_val_test = val_test_subset + sister_sirna 
    return (original_data, result_val_test)

def pick_subset(values_distribution, image_paths):
    HUVEC = values_distribution[0]
    HEPG2 = values_distribution[1]
    RPE = values_distribution[2]
    U2OS = values_distribution[3]
    image_paths = [str(x) for x in image_paths]
    subsets = [("HUVEC", HUVEC), ("HEPG2", HEPG2),
               ("RPE", RPE), ("U2OS", U2OS)]
    val_test_subset= []
    for(label, num) in tqdm(subsets):
        ptr = 0
        label_count = 0
        while(num//2>label_count):
            if(label in image_paths[ptr]):
                val_test_subset.append(image_paths[ptr])
                label_count+=1
                ptr+=1
            else:
                ptr+=1
        print("This is label",label,"with a total count of",label_count)
    print("This is the total subset prior to sister matching", len(val_test_subset))
    original_data, subset = select_sister_sirna(image_paths,val_test_subset)
    print("This is the total subset after sister matching", len(subset))


    return (original_data,val_test_subset) 

def write_splits(file_splits):
    for (dst, content) in file_splits:
        df = pd.DataFrame(content,columns = ["data"])
        df.to_csv(dst,index=False)

def split_data(test_distribution,val_distributions,clean_path,file_names):
    np.random.seed(107)
    read_csv_paths = pd.read_csv(clean_path)
    paths_list = list(read_csv_paths["path"]) 
    np.random.shuffle(paths_list)
    print("[INFO] Processing test data..")
    train_data, test_data= pick_subset(test_distribution,paths_list)
    print("Length of test data is",len(test_data))
    print("[INFO] Processing validation data..")
    train_data, val_data= pick_subset(val_distributions,train_data)
    print("Length of val data is",len(val_data))
    np.random.seed(223)
    np.random.shuffle(train_data)
    np.random.seed(101)
    np.random.shuffle(train_data)
    np.random.seed(7)
    np.random.shuffle(train_data)
    np.random.seed(29)
    np.random.shuffle(val_data)
    np.random.seed(499)
    np.random.shuffle(val_data)
    np.random.seed(757)
    np.random.shuffle(test_data)
    np.random.seed(11)
    np.random.shuffle(test_data)

    result = train_data + val_data + test_data
    train = len(train_data)
    val = len(val_data)
    test = len(test_data)
    train_file, val_file, test_file = file_names
    print("The total available data is ",len(result))
    print("There are {} training, {} validation and {} test images".format(train,val, test))
    file_splits = [ (train_file,train_data),(val_file,val_data),
                    (test_file,test_data)]
    write_splits(file_splits)
    return (train_data, val_data,test_data)

def load_paths(train,val,test):
    train = pd.read_csv(train)
    train = list(train["data"])
    val = pd.read_csv(val)
    val = list(val["data"])
    np.random.seed(757)
    np.random.shuffle(test_data)

    result = train_data + val_data + test_data
    train = len(train_data)
    val = len(val_data)
    test = len(test_data)
    train_file, val_file, test_file = file_names
    print("The total available data is ",len(result))
    print("There are {} training, {} validation and {} test images".format(train,val, test))
    file_splits = [ (train_file,train_data),(val_file,val_data),
                    (test_file,test_data)]
    write_splits(file_splits)
    return (train_data, val_data,test_data)

def load_paths(train,val,test):
    train = pd.read_csv(train)
    train = list(train["data"])
    val = pd.read_csv(val)
    val = list(val["data"])
    test = pd.read_csv(test)
    test = list(test["data"])
    return(train,val,test)

def get_paths_and_labels(splits):
    result = []
    paths = []
    for data in splits:
        buff = [x.split("/")[-1] for x in data]
        label = [int(x.split("?")[0]) for x in buff]
        buff1 = [str(x.split("?")[1]) for x in buff]
        path = ["../train/"+ x for x in buff1]
        result.append(label)
        paths.append(path)
    print("[INFO]There are {} training, {} validation and {} test labels".format(len(result[0]),len(result[1]), len(result[2])))
    labels = [ result[ind] for ind,x in enumerate(result)]
    return paths, labels

def write_hdf5(input_paths,input_labels,build_size,channels,output_hdf5s):
    train_hdf5,val_hdf5,test_hdf5 = output_hdf5s
    (train_paths, val_paths,test_paths ) = input_paths 
    (train_labels, val_labels,test_labels ) = input_labels 

    datasets = [
    	("train", train_paths, train_labels, train_hdf5),
    	("val", val_paths, val_labels, val_hdf5),
    	("test", test_paths, test_labels, test_hdf5)]
    aap = AspectAwarePreprocessor(build_size, build_size)

    # loop over the dataset tuples
    for (dType, paths, labels, outputPath) in datasets:
    	# create HDF5 writer
    	print("[INFO] building {}...".format(outputPath))
    	writer = HDF5DatasetWriter((len(paths), build_size, build_size, channels), outputPath)

    	# initialize the progress bar
    	widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
    		progressbar.Bar(), " ", progressbar.ETA()]
    	pbar = progressbar.ProgressBar(maxval=len(paths),
    		widgets=widgets).start()

    	# loop over the image paths
    	for (i, (path, label)) in enumerate(zip(paths, labels)):
    		# load the image and process it
    		#image = cv2.imread(path)
    		#image = aap.preprocess(image)
    		# add the image and label # to the HDF5 dataset
                image = cv2.imread(path)
                image = cv2.resize(image,(build_size,build_size),interpolation=cv2.INTER_AREA)
                writer.add([image],[label])
                #os.remove(path)
                pbar.update(i)

    	# close the HDF5 writer
    	pbar.finish()
    	writer.close()
