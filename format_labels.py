import pandas as pd
from rxrx_utils import load_site
import numpy as np


def write_labels_to_csv(train_path,test_path):
    train =pd.read_csv(train_path)
    test =pd.read_csv(test_path)
    print(train.head)

    total_data = train.shape[0]+test.shape[0]


    print("Train data:")
    print(train.shape)
    print("Test data:")
    print(test.shape)
    print("Total rows", total_data)


    train["path"] = train["experiment"] + "/Plate" + train["plate"].astype(str)+"/" +train["well"]
    write_train = train[["path","sirna"]]
    test["path"] = test["experiment"] + "/Plate" + test["plate"].astype(str)+"/" +test["well"]
    write_test = train[["path","sirna"]]

    write_train.to_csv(r'./train_sirna_labels.csv',index=False)
    write_test.to_csv(r'./test_sirna_labels.csv',index=False)

def save_array(path,arr):
    np.save(path,arr)


def rewrite_to_array(path,sirna):
    elements = path.split('/')
    experiment =elements[0]
    plate_number = int(elements[1][-1])
    well = elements[2]

    sites = [1,2]
    for site in sites:
        array_image = load_site('train',experiment,plate_number,well,site)
        tag = '-'.join(elements)+ '-'+str(site)
        destination = "./clean_data/"+str(sirna) + "-"+tag

        save_array(destination,array_image)

def dataframe_to_arrray(df):
    print("lool")
    print(df.head(5))
    rows = df.shape[0]
    print(df.shape)
    for i in range(rows):
        path = df.iloc[i,0]
        sirna = df.iloc[i,1]
        rewrite_to_array(path,sirna)