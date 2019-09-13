import pandas as pd


BASE_PATH = "./input/"

TRAIN = BASE_PATH + "train.csv"
TEST = BASE_PATH + "test.csv"
# write_labels_to_csv(TRAIN,TEST)


LABELED = "./input/test_controls.csv"
data = pd.read_csv(LABELED)
print(data.shape)
print(data.head(10))
#dataframe_to_arrray(data)
#free_memory(data)
