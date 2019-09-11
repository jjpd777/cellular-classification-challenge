import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("./input"))
import sys
import matplotlib.pyplot as plt
import rxrx1_utils.rxrx.io as rio


t = rio.load_site('train', 'RPE-05', 3, 'D19', 2)
print(t.shape)
