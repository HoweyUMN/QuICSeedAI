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
from sklearn.preprocessing import StandardScaler
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
### Import Labeled Data
NUM_REPLICATES = 8
labeled_df = pd.read_csv('./data/Format2/Compiled_Data.csv')

dataset = {}
data_dict = {}
replicate = 0
for i,column in enumerate(labeled_df.keys()):
   if i == 0:
      max_time = labeled_df[column].iloc[-1]
      time_step = float(labeled_df[column].iloc[4]) - float(labeled_df[column].iloc[3])
   elif replicate == 0:
      data_dict = {}
      data_dict['Sample'] = labeled_df[column].iloc[0]
      data_dict['Label'] = 1 if 'Pos'in column else 0
      data_dict['Replicate' + str(replicate)] = np.asarray(labeled_df[column].iloc[4:], dtype='float64') # Slice off first number since it's unhelpful
      replicate += 1
   else:
      data_dict['Replicate' + str(replicate)] = np.asarray(labeled_df[column].iloc[4:], dtype='float64')
      replicate = (replicate + 1) % NUM_REPLICATES
      if replicate == 0:
         dataset[labeled_df[column].iloc[1]] = data_dict

#%%
# Create ML dataset
x = []
y = []
for key in dataset.keys():
   row = []
   for i in range(NUM_REPLICATES):
      data = dataset[key]['Replicate'+str(i)]
      if False in np.isfinite(data):
         temp = []
         for i,val in enumerate(data):
            if np.isnan(val):
               if i == 0:
                  temp.append(0)
                  print('Erroneous Sample 0d out')
               else:
                  temp.append(temp[len(temp) - 1])
            else:
               temp.append(val)
         data = np.array(temp)
      row.append(data)
      y.append(dataset[key]['Label'])
   x.append(row)
   # y.append(dataset[key]['Label'])
x = np.array(x)
y = tf.keras.utils.to_categorical(np.array(y))

# Set x values to uniform 0
for i,row in enumerate(x):
   x[i] = row - np.min(row)
# Normalize the dataset
x = x / np.max(x)

x = np.concatenate(x, axis=0)

print(x.shape)
print(y.shape)

x_pos = x[y[:,0] == 0]
x_neg = x[y[:,0] == 1]
y_pos = y[y[:,0] == 0]
y_neg = y[y[:,0] == 1]

#%%
### Create training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_neg, y_neg, test_size=0.2, shuffle=True)

x_test = np.concatenate((x_pos, x_test))
y_test = np.concatenate((y_pos, y_test))

# %%
### Define an Autoencoder
np.object = object
np.int = int
np.float = float
np.bool = bool
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model

NDIM = x.shape[1]

# Inputs
input_layer = Input(shape=NDIM)

# Encoding Layer 1
encoder1 = Dense(NDIM / 2, activation = 'relu')(input_layer)

# Encoding Layer 2
encoder2 = Dense(NDIM / 4, activation = 'relu')(encoder1)

# Encoding Layer 3
encoder3 = Dense(NDIM / 8, activation = 'relu')(encoder2)

# Decoding Layer 1
decoder1 = Dense(NDIM / 4, activation = 'relu')(encoder3)

# Decoding Layer 2
decoder2 = Dense(NDIM / 2, activation = 'relu')(decoder1)

# Output
output = Dense(NDIM, activation='sigmoid')(decoder2)

# Model Definition
model = Model(input_layer, output)
model.summary()

# %%
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-9), loss = 'mse')

history = model.fit(
   x=x_train,
   y=x_train,
   epochs = 1000,
   batch_size = 64
)
# %%
preds = model.predict(x_test)

print(preds[-1])
# %%
