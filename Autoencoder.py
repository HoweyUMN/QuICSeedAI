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
import tensorflow as tf

#%%
### R Function Imports
print('Importing R functions...')
base = importr('base')
utils = importr('utils')
stringr = importr('stringr')
readxl = importr('readxl')
tidyverse = importr('tidyverse')
QuICAnalysis = importr('QuICAnalysis')

#%%
### Data Import
print('Loading R data...')
# Get a list of folders to search for data
folders = next(os.walk('./data'))[1]
folders = ['./data/' + folder for folder in folders]

# Search folders for appropriately named excel files and store together in a list of dictionaries
dataset = []
for i,folder in enumerate(folders):

    if 'Format2' in folder:
       continue

    print(folder)
    dataset.append({})

    plate_path = glob.glob(folder + '/*plate*.xlsx')[0].replace('\\', '/')
    raw_path = glob.glob(folder + '/*raw*.xlsx')[0].replace('\\', '/')
    replicate_path = glob.glob(folder + '/*replicate*.xlsx')[0].replace('\\', '/')

    plate_data = readxl.read_xlsx(plate_path)
    raw_data = readxl.read_xlsx(raw_path)
    replicate_data = readxl.read_xlsx(replicate_path)

    # List of dictionaries mimics original R script
    dataset[i]['Plate'] = plate_data
    dataset[i]['Raw'] = raw_data
    dataset[i]['Replicate'] = replicate_data

# %%
### Data Analysis and Decomposition

# Analyzing dataframes and storing in lists
my_norm_analysis = []
my_analysis = []
for data_dict in dataset:
    robjects.globalenv["AlternativeTime"] = QuICAnalysis.GetTime(data_dict['Raw'])

    meta = QuICAnalysis.GetCleanMeta(data_dict['Raw'], data_dict['Plate'], data_dict['Replicate'])
    clean_raw = QuICAnalysis.GetCleanRaw(meta, data_dict['Raw'])

    analysis = QuICAnalysis.GetAnalysis(clean_raw, 10, 10, 4)
    meta_analysis = base.cbind(meta,analysis)

    analysis_norm = QuICAnalysis.NormAnalysis(metadata = meta, data = meta_analysis, control_name = 'pos')

    my_norm_analysis.append(analysis_norm)
    my_analysis.append(meta_analysis)

# %%
### Extract Raw Fluorescence
x_test = np.asarray(clean_raw).T

x_grouped = []
for i in range(int(len(x_test) / 8)):
   replicates = []
   for j in range(8):
      replicates.append(x_test[8*i + j])
   x_grouped.append(replicates)

x_test = np.array(x_grouped)[:, :, :193].transpose((0,2,1))

# %%
from tensorflow import keras
model = keras.models.load_model('./Outdated/Models/RT-QuicModel.h5')
# %%
y_pred = model.predict(x_test)
y_pred = np.array([arr.item() for arr in y_pred])
print(np.max(np.rint(y_pred)))
# %%
