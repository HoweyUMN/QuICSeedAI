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
samplesTaken = 64
timeInRows = True

#%%
### Dataset Import and Plotting
datasetReal = np.loadtxt("CSV Data/p3p2.csv", delimiter = ",", dtype=str)
if timeInRows: datasetReal = np.transpose(datasetReal)
datasetReal = datasetReal[:, 1:].astype(float) # First datapoint is always strange
labelsReal = np.loadtxt("CSV Data/labels3.csv", delimiter = ",", dtype=str)
labelsReal[0] = 0
labelsReal = labelsReal.astype(int)
datasetSize = datasetScalingFactor * len(datasetReal)

# fig, ax = plt.subplots(8,12)
# for i in range(8):
#     for j in range(12):
#         ax[i, j].set_ylim(0, np.max(datasetReal))
#         ax[i, j].plot(np.arange(samplesTaken), datasetReal[(12*i) + j])
# plt.show()

# mask = np.where(labelsReal == 1, True, False)
# inverseMask = mask = np.where(labelsReal == 1, False, True)

# positiveDS = datasetReal[mask]
# negativeDS = datasetReal[inverseMask]
#%%
### Data Generation

## Defines a sigmoid curve to fit our data
def sigmoid(x, L, x0, k, b):
    y = (L / (1 + np.exp(-k*(x-x0))) + b)
    return y

dataset = np.zeros((datasetSize, samplesTaken))
labels = np.ones(datasetSize)

# fig, ax = plt.subplots(8, 12)
for i,dataCell in enumerate(datasetReal):

    # Randomly vary parameters and add noise
    for j in range(datasetScalingFactor):
        fittedCurveGen = np.copy(dataCell)

        for k in range(len(fittedCurveGen)):
            fittedCurveGen[k] *= random.uniform(0.95, 1.05)
            # Correcting for random dataset blow up
            if(fittedCurveGen[k] > np.max(dataCell) * 2):
                fittedCurveGen[k] = np.max(dataCell) * 2
        
        # ax[int(i/12), int(i % 12)].plot(np.arange(samplesTaken), fittedCurveGen)
        dataset[datasetScalingFactor * i + j] = fittedCurveGen
        labels[datasetScalingFactor * i + j] = labelsReal[i]

print(np.max(dataset))

plt.show()
np.savetxt("Generated Data/generatedData.csv", dataset, delimiter=",")
np.savetxt("Generated Data/generatedLabels.csv", labels, delimiter=",")

#%%
### Generating some sample plots
