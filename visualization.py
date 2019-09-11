import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("./input"))
import sys
import matplotlib.pyplot as plt
import cv2

one_im = "./input/train/HEPG2-01/Plate1/D07_s1_w6.png"
#t = rio.load_site('train', 'RPE-05', 3, 'D19', 2)
#print(t.shape)
im = cv2.imread(one_im)
print(im.shape)
