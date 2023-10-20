# %%
### Imports
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# %% 
### Import Data
dataset = np.loadtxt("generatedData.csv", delimiter = ",", dtype=float)
labels = np.loadtxt("generatedLabels.csv", delimiter = ",", dtype=int)

# Shuffle for better randomness
indices = np.arange(len(labels))
np.random.shuffle(indices)
dataset = dataset[indices]
labels = labels[indices]

# %%
### MLP Approach
scaler = StandardScaler().fit(dataset)
scaledDS = scaler.transform(dataset)

trainDS, testDS, trainLabels, testLabels = train_test_split(scaledDS, labels, test_size=0.2, random_state=7)

NDIM = len(trainDS[0,:])
inputs = Input(shape = (NDIM,), name = 'input')
dense1 = Dense(65, activation = 'relu')(inputs)
dense2 = Dense(65, activation = 'relu')(dense1)
dense3 = Dense(65, activation = 'relu')(dense2)
outputs = Dense(2, name = 'output', kernel_initializer='normal', activation='softmax')(dense3)

model = Model(inputs = inputs, outputs = outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

from keras.callbacks import ModelCheckpoint
model_checkpoint = ModelCheckpoint('MLP.h5', monitor='val_loss', 
                                   verbose=0, save_best_only=True, 
                                   save_weights_only=False, mode='auto', 
                                   save_freq=200)

newTrainLabels = np.zeros((len(trainLabels), 2))
for i,label in enumerate(trainLabels):
    newTrainLabels[i, 0] = 1 - label
    newTrainLabels[i, 1] = label

model.fit(trainDS, 
          newTrainLabels, 
          epochs = 1000, 
          batch_size=200, 
          verbose = 1, 
          callbacks=[early_stopping, model_checkpoint], 
          validation_split=0.1)


# %%
predMLP = model.predict(testDS).argmax(axis = -1)
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(predMLP, testLabels, normalize = 'true')
# %%
