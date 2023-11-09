#%%
### Imports
# Python Package Imports
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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#%%
### R Function Imports
base = importr('base')
utils = importr('utils')
stringr = importr('stringr')
readxl = importr('readxl')
tidyverse = importr('tidyverse')
QuICAnalysis = importr('QuICAnalysis')

#%%
### Data Import

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
   x.append(row)
   y.append(dataset[key]['Label'])
x = np.array(x)
y = tf.keras.utils.to_categorical(np.array(y))

print(x.shape)
print(y.shape)

x_pos = x[y[:,0] == 0]
x_neg = x[y[:,0] == 1]
y_pos = y[y[:,0] == 0]
y_neg = y[y[:,0] == 1]

#%%
### Create training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_pos, y_pos, test_size=0.2, shuffle=True)

x_test = np.concatenate((x_neg, x_test))
y_test = np.concatenate((y_neg, y_test))

# %%
### Define an Autoencoder
np.object = object
np.int = int
np.float = float
np.bool = bool
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Dropout, GlobalAveragePooling1D, BatchNormalization, Input, Activation
from tensorflow.keras import Model

input_layer = Input(shape = x.shape[1:])

conv1 = Conv1D(filters=64, kernel_size=5, kernel_initializer=tf.keras.initializers.HeNormal(7))(input_layer)
conv1 = BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1)

gap = GlobalAveragePooling1D()(conv1)

flat = Flatten()(gap)

encoder1 = Dense(8*x.shape[1], activation= 'relu')(flat)

encoder2 = Dense(4*x.shape[1], activation='relu')(encoder1)

encoder3 = Dense(2*x.shape[1], activation='relu')(encoder2)

decoder1 = Dense(4*x.shape[1], activation='relu')(encoder3)

decoder2 = Dense(8*x.shape[1], activation = 'relu')(decoder1)

output_layer = Dense(8*x.shape[1], activation='sigmoid')(decoder1)

model = Model(inputs = input_layer, outputs = output_layer)
model.summary()

# %%
### Train the model
import keras.backend as K
import tensorflow as tf

epochs = 100
batch_size = 128

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

model.compile(
   optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
   loss = 'binary_crossentropy',
   metrics = ['accuracy', f1_m]
)

history = model.fit(
   x_train,
   x_train.reshape(len(x_train),8*len(x_train[1])),
   batch_size = batch_size,
   epochs = epochs,
   verbose = 1,
   validation_split = 0.1,
   callbacks = [callback]
)

# %%
### Evaluating the Model
preds = model.predict(x_test)

