from utils.format_labels import write_labels_to_csv, dataframe_to_arrray, free_memory
import pandas as pd


BASE_PATH = "./input/"

TRAIN = BASE_PATH + "train.csv"
TEST = BASE_PATH + "test.csv"
# write_labels_to_csv(TRAIN,TEST)


LABELED = "./train_sirna_labels.csv"
data = pd.read_csv(LABELED)
#dataframe_to_arrray(data)
free_memory(data)
