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
ml_quic.get_dataset_statistics()

#%%
### Add KMeans to the list of models to test
from Models import KMeansModel
ml_quic.add_model(KMeansModel.KMeansModel(n_clusters = 2,
                                          file_path= '../SavedModels/Raw/', model_name='kmeans'
                                          ), model_name='KMeans Raw', data_type='raw', tag='Unsupervised')

#%%
### Add Spectral Clustering
from Models import SpectralClustering
ml_quic.add_model(SpectralClustering.SpectralClustering(n_clusters = 3), model_name='Spectral Raw', data_type='raw', tag='Unsupervised')

#%%
### MLP
from Models import MLP
imp.reload(MLP)

# Add MLP to list of supervised models
ml_quic.add_model(MLP.MLP(NDIM = ml_quic.get_num_timesteps_raw(), 
                          file_path='../SavedModels/Raw/', model_name='mlp'
                          ), model_name = 'MLP Raw', data_type = 'raw', tag='Supervised')

#%%
### SVM
from Models import SVM
imp.reload(SVM)

# Add SVM to list of supervised models
ml_quic.add_model(SVM.SVM(
    file_path='../SavedModels/Raw/', model_name='svm_raw'
    ), model_name = 'SVM Raw', data_type = 'raw', tag = 'Supervised')

#%%
### Train Supervised Models
ml_quic.separate_train_test(tags=['Supervised'], train_type=0)
ml_quic.train_models(tags = ['Supervised'])

#%%
### Add KMeans to the list of models to test
from Models import KMeansModel
ml_quic.add_model(KMeansModel.KMeansModel(n_clusters = 3,
                                          file_path= '../SavedModels/Analysis/', model_name='kmeans'
                                          ), model_name='KMeans Analysis', data_type='analysis', tag='Unsupervised')

#%%
### Add Spectral Clustering
from Models import SpectralClustering
ml_quic.add_model(SpectralClustering.SpectralClustering(n_clusters = 3), model_name='Spectral Analysis', data_type='analysis', tag='Unsupervised')
ml_quic.separate_train_test(tags=['Unsupervised'], train_type=3);

#%%
### Get Unsupervised Scores and Plots
# ml_quic.train_models(tags=['Unsupervised'])
# #%%
# ml_quic.get_model_scores(tags=['Unsupervised'])
# ml_quic.evaluate_fp_performance(tags=['Unsupervised'])
# ml_quic.get_model_plots(tags=['Unsupervised'])

#%%
### Get the group plots to use for side by side comparison
# ml_quic.get_group_plots_unsupervised(tags = ['Unsupervised'])

# %%
### SVM
from Models import SVM
imp.reload(SVM)

# Add SVM to list of supervised models
ml_quic.add_model(SVM.SVM(file_path='../SavedModels/Analysis/', model_name='svm_metrics'), model_name = 'SVM Analysis', data_type = 'analysis', tag = 'Supervised')

#%%
### Train Supervised Models
ml_quic.separate_train_test(tags=['Supervised'], train_type=0)
ml_quic.train_models(tags = ['Supervised'])

#%%
### Get Supervised Scores and Plots
ml_quic.get_model_scores(tags = ['Supervised'])
ml_quic.evaluate_fp_performance(tags=['Supervised'])
ml_quic.get_model_plots(tags=['Supervised'])
ml_quic.get_group_plots_supervised(tags = ['Supervised'])

#%%
### Test on G Wells
ml_quic = ml_quic = ML_QuIC.ML_QuIC()
ml_quic.import_dataset(data_dir='../Data/BigAnalysisGWells');
ml_quic.get_dataset_statistics()

ml_quic.add_model(SVM.SVM(
    file_path='../SavedModels/Raw/', model_name='svm_raw'
    ), model_name = 'SVM Raw', data_type = 'raw', tag = 'Supervised')

ml_quic.add_model(SVM.SVM(
    file_path='../SavedModels/Analysis/', model_name='svm_metrics'
    ), model_name = 'SVM Analysis', data_type = 'analysis', tag = 'Supervised')

ml_quic.add_model(MLP.MLP(NDIM = ml_quic.get_num_timesteps_raw(), 
                          file_path='../SavedModels/Raw/', model_name='mlp'
                          ), model_name = 'MLP Raw', data_type = 'raw', tag='Supervised')

ml_quic.separate_train_test(tags=['Supervised'], train_type=3)

#%%
### Get Supervised Scores and Plots
ml_quic.get_model_scores(tags = ['Supervised'])
ml_quic.get_model_plots(tags=['Supervised'])
ml_quic.get_group_plots_supervised(tags = ['Supervised'])
