from format_labels import *
from rxrx_utils import *
from PIL import Image
import cv2
import numpy as np
from skimage.io import imread,imsave
import matplotlib.pyplot as plt


BASE_PATH = "./input/"
EXAMPLE = "./input/train/HEPG2-01/Plate1/B02_s2_w2.png"

trial = "HEPG2-01/Plate1/B02"
sirna = 777
# read_sites(trial,sirna)

LABELED = "./ttt.csv"
data = pd.read_csv(LABELED)
dataframe_to_arrray(data)


# TRAIN = BASE_PATH + "train.csv"
# TEST = BASE_PATH + "test.csv"
# # write_labels_to_csv(TRAIN,TEST)
# t =load_site('train','HEPG2-01',1,'B02',2)
# print(t.shape)
# name = './output/attempt'
# np.save(name,t)

# x = convert_tensor_to_rgb(t)
# x = x.astype(np.uint8)
#
#
#
# imsave("finally.jpg",x)


# this = convert_tensor_to_rgb(site)
# print("This",this.shape)
# cv2.imwrite('fuckyes.jpg',this)
#
# site =load_site_as_rgb('train','HEPG2-01',1,'B02',2)
# print(site.shape)
# cv2.imwrite('bri.jpg',this)

# print(type(site))
# #
# np.savetxt('trial.txt',site)
