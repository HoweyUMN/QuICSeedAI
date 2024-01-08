#%%
### Import Packages
import importlib as imp
import ML_QuIC as ML_QuIC
import matplotlib.pyplot as plt
imp.reload(ML_QuIC)
import copy
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

### Import Data and Create Objects to Analyze
DATA_DIR = '../Data/BigAnalysis'
RANDOM_SEED = 7

# Load data
ml_quic = ML_QuIC.ML_QuIC()
ml_quic.import_dataset(data_dir=DATA_DIR)

# Get data for testing
x = ml_quic.get_numpy_dataset('raw')
y = ml_quic.get_numpy_dataset('labels')

# Separate TN from FP and TP
x_neg = x[y == 0]
x_poscurve = x[y != 0]
y_neg = y[y == 0]
y_poscurve = y[y != 0]

# Train initial model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

### MLP
### Unsupervised Learning - Raw
from Models import KMeansModel

# K-Means
x_train, x_testp, y_train, y_testp = train_test_split(x_poscurve, y_poscurve, test_size=0.2)

model2 = KMeansModel.KMeansModel(n_clusters = 2)
model2.fit(x = x_train, y = y_train)

preds = model2.predict(x_testp)

preds = np.where(preds >= 0.5, 1, 0)

y_testp = np.where(y_testp == 2, 1, 0)

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_testp, preds, normalize = 'true')
plt.show()