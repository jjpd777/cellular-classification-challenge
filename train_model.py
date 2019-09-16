# USAGE
# python resnet_cifar10.py --checkpoints output/checkpoints
# python resnet_cifar10.py --checkpoints output/checkpoints \
# 	--model output/checkpoints/epoch_50.hdf5 --start-epoch 50

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from utils import SimplePreprocessor
from utils import HDF5DatasetGenerator
from utils import config as config
from utils.config import store_params
from utils import ResNet
from utils import EpochCheckpoint
from utils import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.datasets import cifar10
from keras.models import load_model
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import numpy as np
import argparse
import json

store_params()

def poly_decay(epoch):
	max_epochs = config.EPOCHS
	baseLR = config.LEARNING_RATE
	power = config.POWER
	alpha = baseLR * (1- (epoch / float(max_epochs)))** power
	return alpha
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-c", "--checkpoints", required=True,
# 	help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str,
	help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
	help="epoch to restart training at")
args = vars(ap.parse_args())

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.05,
	width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05,
	horizontal_flip=True, fill_mode="nearest")
valaug = ImageDataGenerator()
# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, aug=aug,
 		classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE, aug=valaug,
 		classes=config.NUM_CLASSES)

# if there is no specific model checkpoint supplied, then initialize
# the network (ResNet-56) and compile the model
if args["model"] is None:
	print("[INFO] compiling model...")
	#opt = SGD(lr=config.LEARNING_RATE,nesterov=True,decay=config.DECAY)
	opt = Adam(lr=config.LEARNING_RATE)
	model = ResNet.build(config.RESIZE, config.RESIZE, config.NUM_CHANNELS, config.NUM_CLASSES, stages=config.STAGES,filters = config.FILTERS, reg=config.NETWORK_REG)
	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])

# otherwise, load the checkpoint from disk
else:
	print("[INFO] loading {}...".format(args["model"]))
	opt = Adam(lr=config.LEARNING_RATE)
	model = load_model(args["model"])

	# update the learning rate
	print("[INFO] old learning rate: {}".format(
		K.get_value(model.optimizer.lr)))
	K.set_value(model.optimizer.lr, 0.01)
	print("[INFO] new learning rate: {}".format(
		K.get_value(model.optimizer.lr)))

# construct the set of callbacks
callbacks = [
	EpochCheckpoint(config.CHECKPOINTS, every=10,
		startAt=args["start_epoch"]),
	TrainingMonitor(config.EXPERIMENT_NAME+"trial.png",
		jsonPath=config.EXPERIMENT_NAME+"resnet56_pneumonia.json",
		startAt=args["start_epoch"])]

# train the network
print("[INFO] training network...")
model.fit_generator(
	trainGen.generator(),
	steps_per_epoch=trainGen.numImages//config.BATCH_SIZE,
	validation_data=valGen.generator(),
	validation_steps=valGen.numImages//config.BATCH_SIZE,
	epochs=config.EPOCHS,
	max_queue_size=10,
	callbacks=callbacks, verbose=1)

trainGen.close()
valGen.close()
