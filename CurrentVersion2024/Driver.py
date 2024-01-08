#%%
### Import Packages
import importlib as imp
import ML_QuIC as ML_QuIC
imp.reload(ML_QuIC)
import copy
import numpy as np
import tensorflow as tf

#%%
### Import Data and Create Objects to Analyze
DATA_DIR = '../Data/BigAnalysis'
RANDOM_SEED = 7

# Load data
ml_quic = ML_QuIC.ML_QuIC()
ml_quic.import_dataset(data_dir=DATA_DIR);

#%%
### Unsupervised Learning - Raw
from Models import KMeansModel

# K-Means
ml_quic.add_model(KMeansModel.KMeansModel(n_clusters = 2), model_name='KMeans', data_type = 'raw', tag='Unsupervised')
ml_quic.separate_train_test(model_names=['KMeans'], train_type=3)

#%%
### Autoencoder 
from sklearn.model_selection import train_test_split
from Models import AutoEncoder
imp.reload(AutoEncoder)

# Add model and prep data
ml_quic.add_model(AutoEncoder.AutoEncoder(NDIM=ml_quic.get_num_timesteps_raw()), model_name='AE', data_type = 'raw', tag='Unsupervised')
ml_quic.separate_train_test(model_names=['AE'], train_type=1);

#%%
### Train Unsupervised Models
ml_quic.train_models(tags=['Unsupervised'])

#%%
### Get Unsupervised Scores and Plots
ml_quic.get_model_scores(tags=['Unsupervised']);
ml_quic.evaluate_fp_performance(tags=['Unsupervised'])
ml_quic.get_model_plots(tags=['Unsupervised'])

#%%
### MLP
from Models import MLP
imp.reload(MLP)

# Add MLP to list of supervised models
ml_quic.add_model(MLP.MLP(NDIM = ml_quic.get_num_timesteps_raw()), model_name = 'MLP', data_type = 'raw', tag='Supervised')

#%%
### SVM
from Models import SVM
imp.reload(SVM)

# Add SVM to list of supervised models
ml_quic.add_model(SVM.SVM(), model_name = 'SVM', data_type = 'raw', tag = 'Supervised')

#%%
### Hybrid AE and MLP
from Models import Hybrid
imp.reload(Hybrid)

# Add Hybrid model to list of supervised models
ml_quic.add_model(Hybrid.Hybrid(NDIM = ml_quic.get_num_timesteps_raw()), model_name = 'Hybrid', data_type = 'raw', tag='Supervised')

#%%
### Train Supervised Models
ml_quic.separate_train_test(tags=['Supervised'], train_type=0)
ml_quic.train_models(tags = ['Supervised'])

#%%
### Get Supervised Scores and Plots
ml_quic.get_model_scores(tags = ['Supervised'])
ml_quic.evaluate_fp_performance(tags=['Supervised'])
ml_quic.get_model_plots(tags=['Supervised'])

# %%
