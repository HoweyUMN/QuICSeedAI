#%%
### Import Packages
import imp
import ML_QuIC
imp.reload(ML_QuIC)
import copy
import numpy as np

#%%
### Import Data and Create Objects to Analyze
DATA_DIR = './Data/'
RANDOM_SEED = 7

# Load data
data_importer = ML_QuIC.ML_QuIC()
data_importer.import_dataset(data_dir=DATA_DIR)

# Create clones for different model prep
supervised = copy.copy(data_importer)
unsupervised = copy.copy(data_importer)

# %%
### Unsupervised Learning - Raw
from Models import KMeansModel, AutoEncoder
imp.reload(KMeansModel)
from sklearn.model_selection import train_test_split

#%%
## K-Means
km_struct = copy.copy(unsupervised)
x = km_struct.get_numpy_dataset('raw')
km_struct.set_model(KMeansModel.KMeansModel(n_clusters = 2))
km_struct.train_model(x)
km_struct.get_model_scores()

#%%
## Autoencoder
imp.reload(AutoEncoder)

ae_struct = copy.copy(unsupervised)
x = ae_struct.get_numpy_dataset('raw')
y = ae_struct.get_numpy_dataset('labels')

# Get positive samples to train on 
x_train, x_test, y_train, y_test = train_test_split(x[y == 2], y[y == 2], test_size=0.2)
x_test = np.concatenate((x_test, x[y != 2]))

ae_struct.set_model(AutoEncoder.AutoEncoder(NDIM=x_test.shape[1]))
ae_struct.train_model(dataset=x_train, labels = y_train)
ae_struct.get_model_scores()

# %%
