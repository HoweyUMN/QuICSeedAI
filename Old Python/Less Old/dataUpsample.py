#%%
### Imports
import random
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#%%
### Configure Options
datasetScalingFactor = 50 # Note, real data is included, so final dataset size will be this factor plus 1
samplesTaken = 64

#%%
### Dataset Import and Plotting
excludedFileNames = [
    # '4-11-23_Plate4_RT-QuIC.csv', 
    # '4-11-23_Plate4_RT-QuIC_labels.csv'
    ]
dataFileNames = []
for file in os.listdir('Old Data/CSV Data'):
    if (file.endswith(".csv") and not file in excludedFileNames):
        dataFileNames.append(file)

datasetParts = []
labelParts = []
for i in range(len(dataFileNames)):
    if ('labels' in dataFileNames[i]): continue # We will do labels and data files together to ensure consistency
    thisDataset = np.loadtxt('Old Data/CSV Data/' + dataFileNames[i], delimiter=',')
    thisLabels = np.loadtxt('Old Data/CSV Data/' + dataFileNames[i].replace('.csv', '_labels.csv'), delimiter=',')
    if (len(thisDataset[0]) > samplesTaken): 
        thisDataset = thisDataset[:, :samplesTaken]
    elif (len(thisDataset[0]) < samplesTaken):
        endBuffer = np.zeros((len(thisDataset), samplesTaken - len(thisDataset[0]))) 
        for j in range(len(thisDataset)):
            endBuffer[j] = np.full(samplesTaken - len(thisDataset[0]), thisDataset[j, len(thisDataset[j]) - 1])
        thisDataset = np.concatenate((thisDataset, endBuffer), axis=-1)

    datasetParts.append(thisDataset)
    labelParts.append(thisLabels)

datasetReal = np.concatenate(datasetParts)
labelsReal = np.concatenate(labelParts)
datasetSize = datasetScalingFactor * len(datasetReal)

fig, ax = plt.subplots(len(datasetParts),2)
for i in range(len(datasetParts)):
    ax[i, 0].set_title('Positives')
    ax[i, 1].set_title('Negatives')
    ax[i, 0].set_ylim(0, np.max(datasetReal))
    ax[i, 1].set_ylim(0, np.max(datasetReal))
    for j in range(len(datasetParts[i])):
        if labelParts[i][j] == 1:
            ax[i, 0].plot(np.arange(samplesTaken), datasetParts[i][j])
        else:
            ax[i, 1].plot(np.arange(samplesTaken), datasetParts[i][j])
    
plt.show()

mask = np.where(labelsReal == 1, True, False)
inverseMask = mask = np.where(labelsReal == 1, False, True)

positiveDS = datasetReal[mask]
negativeDS = datasetReal[inverseMask]
#%%
### Data Generation

## Defines a sigmoid curve to fit our data
def sigmoid(x, L, x0, k, b):
    y = (L / (1 + np.exp(-k*(x-x0))) + b)
    return y

dataset = np.zeros((datasetSize + len(datasetReal), samplesTaken))
labels = np.ones(datasetSize + len(datasetReal))

# fig, ax = plt.subplots(dataShape[0], dataShape[1])
for i,dataCell in enumerate(datasetReal):
    
    # Adding real data to dataset
    dataset[datasetScalingFactor * i] = dataCell
    labels[datasetScalingFactor * i] = labelsReal[i]

    # Fitting a sigmoid curve to the data
    p0 = [np.max(dataCell), samplesTaken / 2, 2, np.min(dataCell)] # Give an initial guess
    sigmoidParams, covariance = curve_fit(sigmoid, np.arange(samplesTaken), dataCell, p0, maxfev = 60000)
    fittedCurve = sigmoid(np.arange(samplesTaken), *sigmoidParams)
    # ax[int(i/dataShape[1]), int(i % dataShape[1])].set_ylim(0, np.max(datasetReal))
    # ax[int(i/dataShape[1]), int(i % dataShape[1])].plot(np.arange(samplesTaken), dataCell)

    # Randomly vary parameters and add noise
    for j in range(datasetScalingFactor):
        L = sigmoidParams[0] * random.uniform(0.9, 1.1)
        x0 = sigmoidParams[1] * random.uniform(0.9, 1.1)
        k = sigmoidParams[2] * random.uniform(0.99999, 1.00001)
        b = sigmoidParams[3] * random.uniform(0.9, 1.1)
        fittedCurveGen = sigmoid(np.arange(samplesTaken), L, x0, k, b)

        for k in range(len(fittedCurveGen)):
            fittedCurveGen[k] *= random.uniform(0.95, 1.05)
            # Correcting for random dataset blow up
            if(fittedCurveGen[k] > np.max(dataCell) * 2):
                fittedCurveGen[k] = np.max(dataCell) * 2
        
        # ax[int(i/dataShape[1]), int(i % dataShape[1])].plot(np.arange(samplesTaken), fittedCurveGen)
        dataset[datasetScalingFactor * i + j + 1] = fittedCurveGen
        labels[datasetScalingFactor * i + j + 1] = labelsReal[i]

print(np.max(dataset))

plt.show()
np.savetxt("Old Data/Generated Data/generatedData.csv", dataset, delimiter=",")
np.savetxt("Old Data/Generated Data/generatedLabels.csv", labels, delimiter=",")

# Saving metadata
metadata = [datasetScalingFactor, samplesTaken, len(datasetParts)]
for dataset in datasetParts:
    metadata.append(len(dataset))
np.savetxt("Old Data/Generated Data/metadata.csv", np.array(metadata), delimiter=',')
