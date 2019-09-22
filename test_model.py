# USAGE
# import the necessary packages
from utils import config as config
from utils import HDF5DatasetGenerator
from utils import SimplePreprocessor 
from utils.ranked import rank5_accuracy
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import progressbar
import json
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m","--model")
args = vars(ap.parse_args())
TEST = args["model"]
sp = SimplePreprocessor(config.RESIZE,config.RESIZE)
# load the pretrained network
print("[INFO] loading model...")
model = load_model(TEST)
valaug = ImageDataGenerator(rescale= 1 / 255.0)
print("[INFO] loading model...")
# initialize the testing dataset generator, then make predictions on
# the testing data
print("[INFO] predicting on test data...")
testGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE,
                               classes=config.NUM_CLASSES,preprocessors=[sp])
predictions = model.predict_generator(testGen.generator(), 
	steps=testGen.numImages //config.BATCH_SIZE , max_queue_size=10)

# compute the rank-1 and rank-5 accuracies
(rank1, rank5) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))
testGen.close()
