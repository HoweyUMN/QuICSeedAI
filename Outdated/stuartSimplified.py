#%%
### Imports
import numpy as np
import pandas as pd
np.object = object
np.float = float
np.bool = bool
np.int = int
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import f1_score, average_precision_score
import random
import string
import pickle

import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

#%%
### Setting Global Variables
SEED = 2022
TIME_INCREMENT = 0.25
WORKING_DIRECTORY = '../Format2/'

#%%
### Read in Data (Compatibility with Stuart data)

# Set file names
files =  ["Compiled_Data.csv", "stuart_TRno307.CSV", "stuart_TRno309.CSV", "stuart_TRno310.CSV"]

# Read in data
data = {}
for filename in files:
    data[filename] = pd.read_csv(WORKING_DIRECTORY + filename, header=None)

# Append filename to replicate ID in each file
def append_filename(x, suffix):
    if pd.isnull(x):
        return x
    else:
        return x + '|' + suffix

for filename,df in data.items():
    # Print replicate IDs before
    print("Before: ", end="")
    print(df.iloc[2,:10].tolist())
    # Append filename to replicate ID
    df.iloc[2,:] = df.iloc[2,:].apply(lambda x: append_filename(x, suffix=filename.split('.')[0]))
    # Print replicate IDs after
    print("After: ", end="")
    print(df.iloc[2,:10].tolist())

# %%
### Ensuring Data Consistency

# VALIDATION: Left column equal up to just before initial time value
for filename,df in data.items():
    assert(np.all(np.array(df.iloc[:4,0].tolist()) == np.array(data[files[0]].iloc[:4,0].tolist())))

# Extract minimum time for each file
initial_times = []
for filename,df in data.items():
    print(filename + ": ")
    print(df.iloc[:5,0].tolist())
    print("initial_time=%s\n" % (str(df.iloc[4,0])))
    initial_times.append(float(df.iloc[4,0]))

## Enforce each file starts at 0
files_to_correct = []
for i, initial_time in enumerate(initial_times):
    if initial_time != 0:
        files_to_correct.append(i)

def stringify(t):
    if int(t) == float(t):
        return str(int(t))
    else:
        return str(t)

for i in files_to_correct:
    filename = files[i]
    print("Padding with empty strings to time 0 in %s" % (filename,))
    initial_time = initial_times[i]
    df = data[filename]
    # Compute times to fill in
    times = [stringify(i) for i in np.arange(0,initial_time, TIME_INCREMENT)]
    # Generate block of new rows to fill in
    data_to_insert = np.full([len(times), df.shape[1]-1], '', dtype='<U16')
    data_to_insert[:,0] = times
    # Insert block of new rows
    df = pd.concat([df.iloc[:4,:], pd.DataFrame(data_to_insert), df.iloc[4:,:]]).reset_index(drop=True)
    data[filename] = df

## Enforce same end time for all files
lengths = []
for filename in files:
    lengths.append(len(data[filename]))

min_length = min(lengths)
# VALIDATION: Left column equal up to minimum length
for filename,df in data.items():
    assert(np.all(np.array(df.iloc[:min_length,0].tolist()) == np.array(data[files[0]].iloc[:min_length,0].tolist())))

# Extract max time for each file
final_times = []
for i in range(len(files)):
    filename = files[i]
    df = data[filename]

    print(filename + ": \n...", end="")
    print(df.iloc[len(df)-4:len(df),0].tolist())
    print("final_time=%s\n" % (str(df.iloc[len(df)-1,0])))
    final_times.append(float(df.iloc[len(df)-1,0]))

# Determine latest final time
latest_time = max(final_times)

# Enforce each file goes to final time
files_to_correct = []
for i, final_time in enumerate(final_times):
    if final_time != latest_time:
        files_to_correct.append(i)

def stringify(t):
    if int(t) == float(t):
        return str(int(t))
    else:
        return str(t)

for i in files_to_correct:
    filename = files[i]
    print("Padding with empty strings to %s hours in %s" % (str(latest_time), filename))
    final_time = final_times[i]
    df = data[filename]
    # Compute times to fill in
    times = [stringify(i) for i in np.arange(final_time + TIME_INCREMENT, latest_time + TIME_INCREMENT, TIME_INCREMENT)]
    # Generate block of new rows to fill in
    data_to_insert = np.full([len(times), df.shape[1]-1], '', dtype='<U16')
    data_to_insert[:,0] = times
    # Insert block of new rows
    df = pd.concat([df.iloc[:len(df),:], pd.DataFrame(data_to_insert)]).reset_index(drop=True)
    data[filename] = df

## Double Checking Work
for filename,df in data.items():
    assert(np.all(np.array(df.iloc[:,0].tolist()) == np.array(data[files[0]].iloc[:,0].tolist())))

for filename,df in data.items():
    print(filename + ": \t", end='')
    print(df.shape)

# %%
### Combine Data
dfs_to_combine = []
for filename,df in data.items():
    if filename == files[0]:
        dfs_to_combine.append(df)
    else:
        dfs_to_combine.append(df.iloc[:, 1:])  # Remove first column from subsequent dfs when combining

combined_data = pd.concat(dfs_to_combine, axis=1, ignore_index=True).reset_index(drop=True)

print("Shape of combined data: ", end="")
print(combined_data.shape)

# %% 
### Data Conversion to Numpy
nrows, ncols = combined_data.shape

# Collect timepoints from first column
timepoints_h = combined_data.iloc[4:, 0].astype(float).tolist()

# Pull data instances from each column
X = []
y = []
sample = []
sample_id = []
well_name = []
col_idx = []

for j in range(1, ncols):
    # Parse out column
    column = combined_data.iloc[:, j].tolist()
    # Extract label
    if column[0] == 'Pos':
        y.append(1)
    elif column[0] == 'Neg':
        y.append(0)
    else:
        raise ValueError("Label was not 'Pos' or 'Neg'")
    # Extract sample note
    sample.append(column[1])
    # Extract sample identifier
    sample_id.append(column[2])
    # Extract well name
    well_name.append(column[3])
    # Extract RT-QuIC curve
    curve = column[4:]
    curve = list(pd.to_numeric(curve, errors='coerce'))
    X.append(curve)
    # Extract column index
    col_idx.append(j)

# Convert data to numpy
X = np.array(X)
y = np.array(y)

# %%
### Grouping Samples Together

## Integrity Checks
assert(len(X) == len(y))
assert(len(y) == len(sample))
assert(len(sample) == len(sample_id))
assert(len(sample_id) == len(well_name))
assert(len(well_name) == len(col_idx))

# Pass 1: create samples dictionary
samples = {}

for i in range(len(y)):
    key = sample_id[i]
    if key not in samples:
        samples[key] = {}
        samples[key]['X'] = []
        samples[key]['y_list'] = []
        samples[key]['sample_list'] = []
        samples[key]['well_name_list'] = []
        samples[key]['col_idx_list'] = []
    
    samples[key]['X'].append(X[i].copy())
    samples[key]['y_list'].append(y[i].copy())
    samples[key]['sample_list'].append(sample[i])
    samples[key]['well_name_list'].append(well_name[i])
    samples[key]['col_idx_list'].append(col_idx[i])

# Pass 2: integrity check on samples
keys_to_delete = set()

for key,val in samples.items():
    # Extract data for sample
    X_sample = samples[key]['X']
    y_list = samples[key]['y_list']
    sample_list = samples[key]['sample_list']
    well_name_list = samples[key]['well_name_list']
    col_idx_list = samples[key]['col_idx_list']

    # Integrity check - all lists need to be the same length
    assert(len(X_sample) == len(y_list))
    assert(len(y_list) == len(sample_list))
    assert(len(sample_list) == len(well_name_list))
    assert(len(well_name_list) == len(col_idx_list))

    # Add count
    samples[key]['well_count'] = len(y_list)

    # Check all labels are the same
    if sum(y_list)!=0 and sum(y_list)!=len(y_list):
        print(y_list)
        keys_to_delete.add(key)
    else:
        samples[key]['y'] = y_list[0]
    
    # Convert grouped sample data to array
    samples[key]['X'] = np.array(X_sample)

    # Check if all samples are the same
    sample_list = np.array(sample_list)
    if not np.all(sample_list == sample_list[0]):
        print(sample_list)
        keys_to_delete.add(key)
    else:
        samples[key]['sample'] = sample_list[0]

# Delete samples that failed integrity check
for key in keys_to_delete:
    samples.pop(key)

print("Deleted %i sample(s) due to label inconsistency" % (len(keys_to_delete),))

# Get grouped data - only take samples consisting of 8 replicates
X_grouped = []
y_grouped = []
sample_type = []
keys = []

for key,val in samples.items():
    x = val['X']
    replicate_size = val['well_count']
    if replicate_size == 8:
        X_grouped.append(x)
        y_grouped.append(val['y'])
        sample_type.append(val['sample'])
        keys.append(key)
    else:
        print("only %i replicates in %s" % (replicate_size, key))

X_grouped = np.array(X_grouped)
y_grouped = np.array(y_grouped)
sample_type = np.array(sample_type)
keys = np.array(keys)

#%%
### Remove nans from data
# Pad ends
for i in range(X_grouped.shape[0]):
    # Pull item
    x_group = X_grouped[i].copy()
    for j in range(x_group.shape[0]):
        x = x_group[j].copy()
        # Find first nonnan index
        first_valid_index = np.where(np.isfinite(x) == True)[0][0]
        # Find last nonnan index
        last_valid_index = (~np.isnan(x)).cumsum(0).argmax(0)
        # Num starting nans
        starting_nans = first_valid_index
        # Num ending nans
        ending_nans = len(x) - last_valid_index - 1
        # Truncate to valid indices
        x = x[first_valid_index:last_valid_index+1]
        # Pad if needed with last value
        if len(x) < len(timepoints_h):
            X_grouped[i][j] = np.pad(x, (starting_nans, ending_nans), 'edge')
        # Enforce correct length
        assert(len(X_grouped[i][j]) == len(timepoints_h))

# Check for nans
assert(not np.any(np.isnan(X_grouped)))

# %% 
### Preparing Data for Training
X_grouped = np.swapaxes(X_grouped,1,2)
X_train, X_test, y_train, y_test, sample_train, sample_test, keys_train, keys_test = train_test_split(X_grouped, y_grouped, sample_type, keys, test_size=0.25, stratify=y_grouped, random_state=42)

# %%
### Model Definition
input_layer = tf.keras.layers.Input(X_train[0].shape)

conv1 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding="same", 
                               kernel_initializer = tf.keras.initializers.HeNormal, 
                               kernel_regularizer = tf.keras.regularizers.L1L2(l1 = 1e-5, l2 = 1e-5))(input_layer)
conv1 = tf.keras.layers.BatchNormalization()(conv1)
conv1 = tf.keras.layers.ReLU()(conv1)

conv2 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding="same",
                               kernel_initializer = tf.keras.initializers.HeNormal, 
                               kernel_regularizer = tf.keras.regularizers.L1L2(l1 = 1e-5, l2 = 1e-5))(conv1)
conv2 = tf.keras.layers.BatchNormalization()(conv2)
conv2 = tf.keras.layers.ReLU()(conv2)

conv3 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding="same",
                               kernel_initializer = tf.keras.initializers.HeNormal, 
                               kernel_regularizer = tf.keras.regularizers.L1L2(l1 = 1e-5, l2 = 1e-5))(conv2)
conv3 = tf.keras.layers.BatchNormalization()(conv3)
conv3 = tf.keras.layers.ReLU()(conv3)

gap = tf.keras.layers.GlobalAveragePooling1D()(conv3)

dense1 = tf.keras.layers.Dense(128, activation = 'relu')(gap)
dense1 = tf.keras.layers.Dropout(0.5)(dense1)

output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(dense1)

model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
model.summary()
# %%
### Model Training
epochs = 500
batch_size = 128

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_split = 0.1,
    callbacks=[callback]
)

#%%
### Model Evaluation
y_pred = model.predict(X_test)
y_pred = np.array([arr.item() for arr in y_pred])

display = PrecisionRecallDisplay.from_predictions(y_test, y_pred, name="CNN")
_ = display.ax_.set_title("2-class Precision-Recall curve")

preds_neg = []
preds_pos = []

for i in range(len(y_test)):
    y_predicted = y_pred[i]
    y_real = y_test[i]
    if y_real == 1:
        preds_pos.append(y_predicted)
    else:
        preds_neg.append(y_predicted)

sns.violinplot(data=[preds_neg, preds_pos], orient='h', inner='stick', cut=0)

plt.clf()
plt.hist([preds_neg, preds_pos], range=(0,1), bins=20, label=['Neg samples', 'Pos samples'])
plt.xlabel("Model prediction")
plt.ylabel("Count")
plt.legend(loc='upper right')
plt.show()

y_pred_binarized = (y_pred >= 0.5)
print(classification_report(y_test, y_pred_binarized, target_names=["neg", "pos"]))
f1_score(y_test, y_pred_binarized)

model.save('./Models/RT-QuicModel.h5')
# %%
