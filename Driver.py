#%%
### Import Packages
import importlib as imp
import QuICSeedIF as QuICSeedIF
import copy
import numpy as np
import tensorflow as tf
from Models import KMeansModel, SVM, MLP

#%%
### Import Data and Create Objects to Analyze
DATA_DIR = './Data/GrinderClean'
RANDOM_SEED = 7

# Load dataAC
qsIF = QuICSeedIF.QuICSeedIF()
qsIF.import_dataset(data_dir=DATA_DIR);
qsIF.get_dataset_statistics()

#%%
### Add 2 KMeans Models

qsIF.add_model(KMeansModel.KMeansModel(n_clusters = 3,
                                          file_path= './SavedModels/Raw/', model_name='kmeans'
                                          ), model_name='KMeans Raw', data_type='raw', tag='Unsupervised')

qsIF.add_model(KMeansModel.KMeansModel(n_clusters = 3,
                                          file_path= './SavedModels/Analysis/', model_name='kmeans'
                                          ), model_name='KMeans Metrics', data_type='analysis', tag='Unsupervised')

#%%
### SVM
qsIF.add_model(SVM.SVM(
    file_path='./SavedModels/Raw/', model_name='svm'
    ), model_name = 'SVM Raw', data_type = 'raw', tag = 'Supervised')

qsIF.add_model(SVM.SVM(file_path='./SavedModels/Analysis/', model_name='svm'), model_name = 'SVM Metrics', data_type = 'analysis', tag = 'Supervised')

#%%
### MLP
qsIF.add_model(MLP.MLP(NDIM = qsIF.get_num_timesteps_raw(), 
                          file_path='./SavedModels/Raw/', model_name='mlp'
                          ), model_name = 'MLP Raw', data_type = 'raw', tag='Supervised')

#%%
### Train Models
qsIF.separate_train_test(tags=['Unsupervised', 'Supervised'], train_type=0, test_size=0.20)
qsIF.train_models(tags=['Unsupervised', 'Supervised'])

#%%
### Get Plots and Scores for Unsupervised Models
qsIF.get_model_scores(tags=['Unsupervised'])
qsIF.evaluate_fp_performance(tags=['Unsupervised'])
qsIF.get_model_plots(tags=['Unsupervised'])

#%%
### Get Supervised Scores and Plots
qsIF.get_model_scores(tags = ['Supervised'])
qsIF.evaluate_fp_performance(tags=['Supervised'])
qsIF.get_model_plots(tags=['Supervised'])

#%%
### Test on G Wells
qsIF = qsIF = QuICSeedIF.QuICSeedIF()
qsIF.import_dataset(data_dir='./Data/GrinderGWells')
qsIF.get_dataset_statistics()

qsIF.add_model(KMeansModel.KMeansModel(n_clusters = 3,
                                          file_path= './SavedModels/Analysis/', model_name='kmeans'
                                          ), model_name='KMeans Metrics', data_type='analysis', tag='Unsupervised')

qsIF.add_model(SVM.SVM(
    file_path='./SavedModels/Raw/', model_name='svm'
    ), model_name = 'SVM Raw', data_type = 'raw', tag = 'Supervised')

qsIF.add_model(SVM.SVM(
    file_path='./SavedModels/Analysis/', model_name='svm'
    ), model_name = 'SVM Metrics', data_type = 'analysis', tag = 'Supervised')

qsIF.add_model(MLP.MLP(NDIM = qsIF.get_num_timesteps_raw(), 
                          file_path='./SavedModels/Raw/', model_name='mlp'
                          ), model_name = 'MLP Raw', data_type = 'raw', tag='Supervised')

qsIF.separate_train_test(tags=['Supervised', 'Unsupervised'], train_type=3, file_loc='./TrainTest/GWells')

### Get Supervised Scores and Plots
qsIF.get_model_scores(tags = ['Supervised', 'Unsupervised'])

pred_km, sample_list_km = qsIF.evaluate_replicate_performance(model='KMeans Metrics')
pred_svm_r, sample_list_svmr = qsIF.evaluate_replicate_performance(model='SVM Raw')
pred_svm_m, sample_list_svmm = qsIF.evaluate_replicate_performance(model='SVM Metrics')
pred_mlp, sample_list_mlp = qsIF.evaluate_replicate_performance(model='MLP Raw')

print('Model Sample Predictions:')
print('\n{:20s} {:20s} {:20s} {:20s} {:20s}'.format('Sample:', 'KMeans Metrics:', 'SVM Raw:', 'SVM Metrics:', 'MLP Raw:'))
for i in range(len(pred_km)):
    if sample_list_km[i] != sample_list_mlp[i] or sample_list_mlp[i] != sample_list_svmm[i] or sample_list_svmm[i] != sample_list_svmr[i]:
        raise Exception('Sample order does not agree!')
    print('{:20s} {:20s} {:20s} {:20s} {:20s}'.format(sample_list_km[i], pred_km[i], pred_svm_r[i], pred_svm_m[i], pred_mlp[i]))


# %%
### Test on Unrelated Data
qsIF = qsIF = QuICSeedIF.QuICSeedIF()
qsIF.import_dataset(data_dir='./Data/ValidationData')
qsIF.get_dataset_statistics()

qsIF.add_model(KMeansModel.KMeansModel(n_clusters = 3,
                                          file_path= './SavedModels/Raw/', model_name='kmeans'
                                          ), model_name='KMeans Raw', data_type='raw', tag='Unsupervised')
qsIF.add_model(KMeansModel.KMeansModel(n_clusters = 3,
                                          file_path= './SavedModels/Analysis/', model_name='kmeans'
                                          ), model_name='KMeans Metrics', data_type='analysis', tag='Unsupervised')

qsIF.add_model(SVM.SVM(
    file_path='./SavedModels/Raw/', model_name='svm'
    ), model_name = 'SVM Raw', data_type = 'raw', tag = 'Supervised')

qsIF.add_model(SVM.SVM(
    file_path='./SavedModels/Analysis/', model_name='svm'
    ), model_name = 'SVM Metrics', data_type = 'analysis', tag = 'Supervised')

qsIF.add_model(MLP.MLP(NDIM = qsIF.get_num_timesteps_raw(), 
                          file_path='./SavedModels/Raw/', model_name='mlp'
                          ), model_name = 'MLP Raw', data_type = 'raw', tag='Supervised')

qsIF.separate_train_test(tags=['Supervised', 'Unsupervised'], train_type=3, file_loc='./TrainTest/Val_Data')

### Get Supervised Scores and Plots
qsIF.evaluate_fp_performance(tags=['Unsupervised', 'Supervised'], file_loc='./FiguresVal/')
qsIF.get_model_scores(tags = ['Unsupervised', 'Supervised'])
qsIF.get_model_plots(tags=['Unsupervised', 'Supervised'], file_loc='./FiguresVal/')
# %%
