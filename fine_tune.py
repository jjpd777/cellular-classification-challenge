#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
from utils.ranked import rank5_accuracy
# import the necessary packages
from utils import config as config
from utils import SimplePreprocessor
from utils import TrainingMonitor
from utils import HDF5DatasetGenerator
from utils import FCHeadNet
from utils import EpochCheckpoint 
from keras.applications import Xception
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.callbacks import LearningRateScheduler
import json
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,
        help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
        help="epoch to restart training at")
args = vars(ap.parse_args())
# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")
#
# initialize the image preprocessors
sp = SimplePreprocessor(config.RESIZE,config.RESIZE)

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, aug=aug,
	preprocessors=[sp], classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE,
	preprocessors=[sp], classes=config.NUM_CLASSES)

########
sp = SimplePreprocessor(config.RESIZE,config.RESIZE)


# load the pretrained network

# initialize the testing dataset generator, then make predictions on
# the testing data
print("[INFO] predicting on test data (no crops)...")
testGen = HDF5DatasetGenerator(config.TEST_HDF5, config.BATCH_SIZE,
	preprocessors=[sp], classes=config.NUM_CLASSES)

########
# initialize the optimizer
print("[INFO] compiling model...")
baseModel = Xception(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(config.RESIZE, config.RESIZE, 3)))
headModel = FCHeadNet.build(baseModel, config.NUM_CLASSES,config.FCH1,config.FCH2 )
model = Model(inputs=baseModel.input, outputs=headModel)

opt = Adam(lr= config.LEARNING_RATE)

for layer in baseModel.layers:
	layer.trainable = False
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# construct the set of callbacks
callbacks = [
        EpochCheckpoint(config.CHECKPOINTS, every=5,
                startAt=args["start_epoch"]),
        TrainingMonitor(config.EXPERIMENT_NAME+"monitor.png",
                jsonPath=config.EXPERIMENT_NAME+"monitor.json",
                startAt=args["start_epoch"])]
# train the network
model.fit_generator(
	trainGen.generator(),
	steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
	validation_data=valGen.generator(),
	validation_steps=valGen.numImages // config.BATCH_SIZE,
	epochs=10,
	max_queue_size=10,
	callbacks=callbacks, verbose=1)

for layer in baseModel.layers[120:]:
	layer.trainable = True

opt = Adam(lr= config.SECOND_LR)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
model.fit_generator(
	trainGen.generator(),
	steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
	validation_data=valGen.generator(),
	validation_steps=valGen.numImages // config.BATCH_SIZE,
	epochs=config.EPOCHS,
	max_queue_size=10,
	callbacks=callbacks, verbose=1)
# save the model to file
print("[INFO] serializing model...")

# close the HDF5 datasets
trainGen.close()
valGen.close()
