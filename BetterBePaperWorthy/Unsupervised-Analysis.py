#%%
### Imports
# Python Package Imports
print('Initializing...')
import os
import socket
if socket.gethostname() == 'Desktop-CS1TBMI':
  os.environ['R_HOME'] = "C:/PROGRA~1/R/R-43~1.1"
else:
  os.environ['R_HOME'] = 'C:/Users/howey024/AppData/Local/Programs/R/R-4.3.2'
import glob

import preprocess
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.ipython.ggplot import image_png
from rpy2.robjects.packages import importr, data

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Dropout, GlobalAveragePooling1D, BatchNormalization, Input, Activation
from tensorflow.keras import Model
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

#%%
### Get analyzed data
DATA_DIR = '../data/'
analyzed_dataset = preprocess.preprocess().analyze_features_from_csvs(data_dir = DATA_DIR)

# %%
