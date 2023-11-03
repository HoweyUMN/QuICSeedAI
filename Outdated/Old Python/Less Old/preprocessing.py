### This script organizes XLSX files for a 96 well RTQuic plate and converts it to a standardized format for
# training machine learning models. The script assumes A1, B1, etc. are the same tissue source and thus groups
# them together for easy division into a testing/training set. Supports up to 48 hour runtime at 45 minute increments
#%%
### Imports
import pandas as pd
import os
import numpy as np

#%%
### Loading Data
dataFileNames = []
for file in os.listdir('Raw Data'):
    if file.endswith(".xlsx"):
        dataFileNames.append(file)

dataFrames = []
for i in range(len(dataFileNames)):
    dataFrames.append(pd.read_excel("Raw Data/" + dataFileNames[i])) # Note, this will fail if the file is otherwise open

# %%
### Getting Data Labels and Shapes
labelSet = []
dataShapes = []
plateShape = []
for df in dataFrames:
    thisLabels = []
    cols = 0
    rows = 0
    thisPlateShape = np.full((12, 8) , -1)
    for i in range(12): 
        for j in range(8):
            if (not np.isnan(df[i + 1][j])):
                thisLabels.append(df[i + 1][j])
                thisPlateShape[i, j] = df[i + 1][j]
                if (cols < i + 1): cols = i + 1 # If this column has data, include it in our count
                if (i == 0): rows += 1 # Assuming it's a rectangle, find how many rows are occupied
    plateShape.append(thisPlateShape)
    labelSet.append(thisLabels)
    dataShapes.append([rows, cols])

# %%
### Defining Functions for Different XLSX Formats
def sampleInCol(dataFrame, index):
    timeSteps = int(np.max(df.iloc[11:77, 1]) / 0.75)
    dataset = np.zeros((dataShapes[index][0] * dataShapes[index][1], timeSteps))
    validSampleIndex = 0
    for i in range(12):
        for j in range(8):
            if(plateShape[index][i][j] == -1): continue
            dataset[validSampleIndex] = dataFrame.iloc[12:timeSteps + 12, 2 + (j * 12) + i]
            validSampleIndex += 1

    return dataset

def sampleInRow(dataFrame, index):
    timeSteps = int(np.max(df.iloc[10, 2:67]) / 0.75)
    dataset = np.zeros((dataShapes[index][0] * dataShapes[index][1], timeSteps))
    validSampleIndex = 0
    for i in range(12):
        for j in range(8):
            if(plateShape[index][i][j] == -1): continue
            dataset[validSampleIndex] = dataFrame.iloc[11 + (j * 12) + i, 3:timeSteps+3] # Chop off 0th timestamp as it's usually bad
            validSampleIndex += 1

    return dataset
#%%
### Importing Data into Consistent Format
for index, df in enumerate(dataFrames):
    if (df.iloc[11, 1] == 0 and df.iloc[12, 1] == 0.75):
        dataset = sampleInCol(df, index)
    else:
        dataset = sampleInRow(df, index)
    np.savetxt('CSV Data/' + dataFileNames[index].replace('.xlsx', '.csv'), dataset, delimiter=",")
    np.savetxt('CSV Data/' + dataFileNames[index].replace('.xlsx', '_labels.csv'), labelSet[index], delimiter=",")
