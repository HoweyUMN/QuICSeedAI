#%%
### Import Packages
import importlib as imp
import ML_QuIC
imp.reload(ML_QuIC)
import copy
import numpy as np

#%%
### Import Data and Create Objects to Analyze
DATA_DIR = './Data/'
RANDOM_SEED = 7

# Load data
ml_quic = ML_QuIC.ML_QuIC()
ml_quic.import_dataset(data_dir=DATA_DIR);

# %%
### Unsupervised Learning - Raw
from Models import KMeansModel

# K-Means
ml_quic.add_model(KMeansModel.KMeansModel(n_clusters = 2), model_name='KMeans', tag='Unsupervised')
ml_quic.separate_train_test(model_names=['KMeans'], train_type=3)

#%%
# Autoencoder 
from sklearn.model_selection import train_test_split
from Models import AutoEncoder
imp.reload(AutoEncoder)

# Add model and prep data
ml_quic.add_model(AutoEncoder.AutoEncoder(NDIM=ml_quic.get_num_timesteps_raw()), model_name='AE', tag='Unsupervised')
ml_quic.separate_train_test(model_names=['AE'], train_type=1);

#%%
ml_quic.train_models(tags=['Unsupervised'])
# %%
ml_quic.get_model_scores(tags=['Unsupervised']);
#%%
ml_quic.get_model_plots(tags=['Unsupervised'])