# %%
### Imports
import tensorflow as tf
import time
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
metadata = np.loadtxt("Generated Data/metadata.csv", delimiter=",", dtype = int)
datasetScalingFactor = metadata[0]
numPlates = metadata[2]
plateSizes = metadata[3:3+numPlates]
dataset = np.loadtxt("Generated Data/generatedData.csv", delimiter = ",", dtype = float)
labels = np.loadtxt("Generated Data/generatedLabels.csv", delimiter=",", dtype = int)

realIndices = np.arange(0, len(dataset), datasetScalingFactor)

#%%
### Normalizing Dataset
normDataset = np.zeros((len(dataset), len(dataset[0])))
for i, data in enumerate(dataset):
    tot = np.sum(data)
    for j in range(len(dataset[0])):
        if(np.isnan(dataset[i,j] / tot)): continue
        normDataset[i,j] = dataset[i,j] / tot

dataset = normDataset
datasetTrain = np.concatenate((dataset[0:48*datasetScalingFactor], dataset[134*datasetScalingFactor:]))
labelsTrain = np.concatenate((labels[:48*datasetScalingFactor], labels[134*datasetScalingFactor:]))
 # labelsTest, labelsTrain = np.split(labels, 2)

# Potentially add normalization
#%%
### Data preprocessing
indices = np.ma.make_mask(labelsTrain)
positives = datasetTrain[indices]
inverseIndices = np.where(indices == False, True, False)
negatives = datasetTrain[inverseIndices]
trainDS, testDS, trainLabels, testLabels = train_test_split(positives, labelsTrain[indices], test_size=0.2, random_state=7)
NDIM = len(trainDS[0,:])
statNDIM = len(trainDS[0,:])

testDS = np.concatenate((testDS, negatives))
testLabels = np.concatenate((testLabels, labelsTrain[inverseIndices]))
# Shuffle to avoid mearningless patterns
indices = np.arange(len(testLabels)),
np.random.shuffle(indices)
testDS = testDS[indices]
testLabels = testLabels[indices]

indices = np.arange(len(trainLabels))
np.random.shuffle(indices)
trainDS = trainDS[indices]
trainLabels = trainLabels[indices]

#%%
### Build Autoencoder
Input_Layer=Input(shape=NDIM)
encoder = Dense(NDIM, activation="tanh")(Input_Layer)
encoder = Dense(NDIM/2, activation='relu')(encoder)
encoder = Dense(NDIM/4, activation='relu')(encoder)
decoder = Dense(NDIM/2, activation="relu")(encoder)
decoder = Dense(NDIM, activation='relu')(encoder)
Output_Layer = Dense(NDIM, activation="tanh")(decoder)

autoencoder = Model(inputs = Input_Layer, outputs = Output_Layer)
autoencoder.summary()

#%%
### Training the Autoencoder
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=20,
                               min_delta=1e-7,
                               restore_best_weights=True)

from keras.callbacks import ModelCheckpoint
model_checkpoint = ModelCheckpoint('MLP.h5', monitor='val_loss', 
                                   verbose=0, save_best_only=True, 
                                   save_weights_only=False, mode='auto', 
                                   save_freq=5)
autoencoder.compile(loss='mse', optimizer='adam')
autoencoder.fit(trainDS,
                trainDS,
                epochs = 100,
                batch_size=1000,
                shuffle=True,
                validation_split=0.1,
                verbose=0,
                callbacks=[early_stopping, model_checkpoint])

#%%
### Evaluating the Autoencoder and getting Scores
reconstructions = autoencoder.predict(testDS)

mse = np.zeros(len(reconstructions))
predictions = np.zeros(len(reconstructions))
for i,reconstruction in enumerate(reconstructions):
    rawError = np.zeros(len(reconstruction))
    squaredError = np.zeros(len(reconstruction))
    for j in range (len(reconstruction)):
        rawError[j] = reconstruction[j] - testDS[i,j]
        squaredError[j] = rawError[j] ** 2
    mse[i] = np.mean(squaredError, axis=-1)
    predictions[i] = 0 if mse[i] > np.mean(mse) + (np.mean(mse) / 20) else 1

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(testLabels, predictions, normalize='true')

#%% 
### Training an MLP on the autoencoder output
trainMLP = np.concatenate((testDS, reconstructions), axis=-1)
scaler = StandardScaler().fit(trainMLP)
trainMLP = scaler.transform(trainMLP)

NDIM = len(trainMLP[0,:])
inputs = Input(shape = (NDIM,), name = 'input')
dense1 = Dense(NDIM, activation = 'relu')(inputs)
outputs = Dense(2, name = 'output', kernel_initializer='normal', activation='softmax')(dense1)

model = Model(inputs = inputs, outputs = outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.Precision()])
model.summary()

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

from keras.callbacks import ModelCheckpoint
model_checkpoint = ModelCheckpoint('MLP.h5', monitor='val_loss', 
                                   verbose=0, save_best_only=True, 
                                   save_weights_only=False, mode='auto', 
                                   save_freq=5)

newTrainLabels = np.zeros((len(testLabels), 2))
for i,label in enumerate(testLabels):
    newTrainLabels[i, 0] = 1 - label
    newTrainLabels[i, 1] = label

model.fit(trainMLP, 
          newTrainLabels, 
          epochs = 100, 
          batch_size=200, 
          verbose = 0, 
          callbacks=[early_stopping, model_checkpoint], 
          validation_split=0.1)


#%%
### Testing on entire dataset, including withheld data
reconstructions = autoencoder.predict(dataset)
testMLP = np.concatenate((dataset, reconstructions), axis=-1)
testMLP = scaler.transform(testMLP)
predMLP = model.predict(testMLP).argmax(axis=-1)

ConfusionMatrixDisplay.from_predictions(labels, predMLP, normalize = 'true')
# %%
### Plotting real data that was misclassified
for index in realIndices:
    if (predMLP[index] != labels[index]):
        sampleNum = index / datasetScalingFactor
        # Find location of misclassified data
        for i, plateSize in enumerate(plateSizes):
            if(sampleNum >= plateSize):
                sampleNum -= plateSize
            else:
                print('Misclassification of P{} location '.format(i) + chr(65 + int(sampleNum) % 8) +  str(1 + int(sampleNum / 8)))
                print('      Predicted: {}  Actual: {}'.format(predMLP[index], labels[index]))
                plt.plot(np.arange(NDIM / 2), reconstructions[index])
                plt.plot(np.arange(NDIM/2), dataset[index])
                plt.show()
# %%
