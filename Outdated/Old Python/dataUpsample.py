#%%
### Imports
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#%%
### Configure Options
datasetScalingFactor = 520
samplesTaken = 47
timeInRows = False
dataShape = [6, 8]

#%%
### Dataset Import and Plotting
datasetReal = np.loadtxt("CSV Data/p3p2.csv", delimiter = ",", dtype=str)
datasetReal = datasetReal[:, 1:].astype(float) # First datapoint is always strange
if timeInRows: datasetReal = np.transpose(datasetReal)
labelsReal = np.loadtxt("CSV Data/labels.csv", delimiter = ",", dtype=str)
labelsReal[0] = 0
labelsReal = labelsReal.astype(int)
datasetSize = datasetScalingFactor * len(datasetReal)

fig, ax = plt.subplots(dataShape[0],dataShape[1])
for i in range(dataShape[0]):
    for j in range(dataShape[1]):
        ax[i, j].set_ylim(0, np.max(datasetReal))
        ax[i, j].plot(np.arange(samplesTaken), datasetReal[(dataShape[1]*i) + j])
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

dataset = np.zeros((datasetSize, samplesTaken))
labels = np.ones(datasetSize)

# fig, ax = plt.subplots(dataShape[0], dataShape[1])
for i,dataCell in enumerate(datasetReal):
    # Fitting a sigmoid curve to the data
    p0 = [np.max(dataCell), samplesTaken / 2, 2, np.min(dataCell)] # Give an initial guess
    sigmoidParams, covariance = curve_fit(sigmoid, np.arange(samplesTaken), dataCell, p0, maxfev = 60000)
    fittedCurve = sigmoid(np.arange(samplesTaken), *sigmoidParams)
    
    # ax[int(i/dataShape[1]), int(i % dataShape[1])].set_ylim(0, np.max(datasetReal))
    # ax[int(i/dataShape[1]), int(i % dataShape[1])].plot(np.arange(samplesTaken), dataCell)
    print(sigmoidParams)

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
        dataset[datasetScalingFactor * i + j] = fittedCurveGen
        labels[datasetScalingFactor * i + j] = labelsReal[i]

print(np.max(dataset))

plt.show()
np.savetxt("Generated Data/generatedData.csv", dataset, delimiter=",")
np.savetxt("Generated Data/generatedLabels.csv", labels, delimiter=",")

#%%
### Generating some sample plots
