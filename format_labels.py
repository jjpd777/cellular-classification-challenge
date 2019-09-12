import pandas as pd

BASE_PATH = "./input/"

TRAIN = BASE_PATH + "train.csv"
TEST = BASE_PATH + "test.csv"

train =pd.read_csv(TRAIN)
test =pd.read_csv(TEST)


total_data = train.shape[0]+test.shape[0]


print("Train data:")
print(train.shape)
print("Test data:")
print(test.shape)
print("Total rows", total_data)


train["path"] = train["experiment"] + "/Plate" + train["plate"].astype(str)+"/" +train["well"]
write = train[["path","sirna"]]
write.to_csv(r'./sirna_labels.csv',index=False)
