"""The code herein contains all the methods and procedures necessary for importing and processing rds
data before training models on it. The data can be analyzed using R scripts or cleaned and processed as
raw fluorescence values"""
import os
import socket
if socket.gethostname() == 'Desktop-CS1TBMI':
  os.environ['R_HOME'] = "C:/PROGRA~1/R/R-43~1.1"
else:
  os.environ['R_HOME'] = 'C:/Users/howey024/AppData/Local/Programs/R/R-4.3.2'
import glob
# import rpy2
# import rpy2.robjects as robjects
# from rpy2.robjects import pandas2ri
# from rpy2.ipython.ggplot import image_png
# from rpy2.robjects.packages import importr, data
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
from sklearn.preprocessing import StandardScaler
import multiprocessing
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, roc_auc_score, roc_curve
import keras

# base = importr('base')
# utils = importr('utils')
# stringr = importr('stringr')
# readxl = importr('readxl')
# tidyverse = importr('tidyverse')
# QuICAnalysis = importr('QuICAnalysis')

class ML_QuIC:
  """This class contains a series of Python methods to processs and perform machine learning
  on RT-Quic data. R processing is currently unimplemented.

  \nAuthor - Kyle Howey
  \nVersion - January 11, 2024"""

  def __init__(self):
    # Reset backend session
    keras.backend.clear_session()

    self.raw_dataset = None
    """The raw fluorescence values as directly imported from the CSVs"""

    self.analysis_dataset = None
    """The engineered feature dataset as directly imported from the CSVs"""

    self.metadata = None
    """The metadata associated with the different samples"""

    self.replicate_data = None
    """Stores relationship between replicates and sample id"""

    self.labels = None
    """The extracted labels from the dataset"""

    self.training_indices = {}
    """The indices of data used for training for direct comparison of models on same data"""

    self.testing_indices = {}
    """The indices of data used for testing for direct comparison of models on same data"""

    self.models = {}
    """A dictionary of all the models"""

    self.model_dtype = {}
    """Model datatypes for training."""

    self.predictions = {}
    """A dictionary of predictions for a given model"""

    self.scores = {}
    """The scores returned by each of the models stored as a dictionary"""

    self.tags = {}
    """A dictionary of tags and their corresponding models for operating on groups rather than individual models"""

    self.plots = {}
    """A return of the plots for later use and examination stored as a dictionary for each model"""
    
    self.fp_plots = {}
    """Stores false positive plots for unified plotting"""
    
    self.max_fluorescence = 0
    "Store a maximum fluorescence value for use with plotting"
      
  def import_dataset(self, data_dir = './Data/', folders = None):
    """Takes in the directory data is stored in and the selected folder names 
      (automatically gotten if unspecified) and imports the data for training.\n
      Parameters:\n
      data_dir - path to directory where folders are stored\n
      folders - directories in data_dir where data is stored\n
      Returns:\n
      dataset - Imported dataframes for [raw, metadata, analysis]"""
      
    # If no folders are given, automatically search for all potential data folders
    if folders is None:
      folders = next(os.walk(data_dir))[1]
    folders = [data_dir + folder for folder in folders]
    folders.append(data_dir) # Search the data directory itself for data

    # Load and format data
    metadata = pd.DataFrame()
    raw_data = pd.DataFrame()
    analysis = pd.DataFrame()
    replicates = pd.DataFrame()
    for i, folder in enumerate(folders):
      print('Loading Data from {}'.format(folder))

      # Get Metadata
      meta_path = glob.glob(folder + '/*meta*.csv')[0].replace('\\', '/')
      this_md = pd.read_csv(meta_path)
      metadata = pd.concat((metadata, this_md))

      # Get Raw Data
      raw_path = glob.glob(folder + '/*raw*.csv')[0].replace('\\', '/')
      this_raw = pd.read_csv(raw_path)
      raw_data = pd.concat((raw_data, this_raw))

      # Get Analyzed Data
      analysis_path = glob.glob(folder + '/*analysis*.csv')[0].replace('\\', '/')
      this_analysis = pd.read_csv(analysis_path)
      analysis = pd.concat((analysis, this_analysis))

      # Get replicate metadata for true data cells
      replicate_path = glob.glob(folder + '/*replicate*.csv')[0].replace('\\', '/')
      this_replicate = pd.read_csv(replicate_path)
      replicates = pd.concat((replicates, this_replicate))

      # Eliminate Blank wells from dataset
      raw_data = raw_data[metadata['content'] != 'blank']
      analysis = analysis[metadata['content'] != 'blank']
      metadata = metadata[metadata['content'] != 'blank']

    # Error Checking
    if len(metadata) != len(raw_data) or len(metadata) != len(analysis):
      raise Exception('Dataset sizes are inconsistent - ensure all raw, meta, and analysis' +
                      ' data lines up!')

    # Take imported data and store with the object attributes
    self.raw_dataset = raw_data
    self.metadata = metadata
    self.analysis_dataset = analysis
    self.labels = metadata['final']
    self.replicate_data = replicates
    
    # Fit a standard scaler to the raw data for plots
    self.max_fluorescence = np.max(self.get_numpy_dataset())

    return [raw_data, metadata, analysis]
  
  def get_num_timesteps_raw(self):
    """Getter function for the number of features used in training"""
    return self.raw_dataset.shape[1] - 1 # -1 because we ignore labels
  
  def get_dataset_statistics(self, verbose = 1):
    """Getter function to get stats of the imported dataset"""

    # Ensure there is data to analyze
    if self.metadata is None:
      raise Exception("No data imported.")
    
    # Get the counts of different well types in the dataset
    negatives = len(self.metadata[self.metadata['final'] == 0])
    fps = len(self.metadata[self.metadata['final'] == 1])
    positives = len(self.metadata[self.metadata['final'] == 2])

    # blank_wells = len(self.metadata[self.metadata['content'] == 'blank'])
    control_wells = len(self.metadata[self.metadata['content'] == 'pos']) + len(self.metadata[self.metadata['content'] == 'neg'])
    data_wells = len(self.metadata) - control_wells 

    if verbose:
      print('---- Dataset Label Distribution ----')
      print('Negative Samples: {}'.format(negatives))
      print('False Positive Samples: {}'.format(fps))
      print('Positive Samples: {}'.format(positives))

      print('\n---- Well Content Distribution: ----')
      print('Data Wells: {}'.format(data_wells))
      # print('Blank Wells: {}'.format(blank_wells))
      print('Control Wells: {}'.format(control_wells))

    return [negatives, fps, positives, control_wells, data_wells]

  def separate_train_test(self, seed = 7, test_size = 0.1, train_type = 0, model_names = None, tags = None):
    """Separates imported data into a training set and a testing set.\n
    train_type: 0 - Mix of samples, 1 is positive samples only, 2 is negative samples only, 3 is all data is trained on. If only pos/neg is used, 
    test size is how many of the training sample type to withold\n
    Returns: [training_indices, testing_indices]"""
    np.random.seed = seed

    # If model names are not specified, assume we run on all models
    models = []
    if model_names is None:
      models = self.models.keys()
    else: models = model_names

    # Override previous import if tags are specified  
    if (not tags is None) and (model_names is None):
      models = []
      for tag in tags:
        models += self.tags[tag]

    # Train on only positive samples and test on mixed dataset
    if train_type == 1:
      pos_indices = np.array(np.where(self.labels == 2)[0])
      neg_indices = np.array(np.where(self.labels != 2)[0])

      train_indices = pos_indices[:int(test_size * len(pos_indices))]
      test_indices = np.concatenate((neg_indices, pos_indices[int(test_size * len(pos_indices)):]))
      np.random.shuffle(test_indices)
      np.random.shuffle(train_indices)

    # Train on only negative samples and test on mixed dataset
    elif train_type == 2:
      pos_indices = np.where(self.labels == 2)
      neg_indices = np.where(self.labels != 2)

      train_indices = neg_indices[:int(test_size * len(neg_indices))]
      test_indices = np.concatenate((pos_indices, neg_indices[int(test_size * len(neg_indices)):]))
      test_indices = np.random.shuffle(test_indices)
      train_indices = np.random.shuffle(train_indices)

    # Train and test on mixed dataset
    elif train_type == 0:
      # Shuffle the dataset to ensure randomness
      indices = np.random.permutation(len(self.raw_dataset))

      test_indices = indices[:int(test_size * len(self.raw_dataset))]
      train_indices = indices[int(test_size * len(self.raw_dataset)):]

    # Train and test on entire dataset (only appropriate for unsupervised learning)
    elif train_type == 3:
      train_indices = np.random.permutation(len(self.raw_dataset))
      test_indices = np.random.permutation(len(self.raw_dataset))

    else:
      raise Exception('Invalid argument, train_type must be 0, 1, 2, or 3!')

    # Ensure this is consitent for all models passed in
    for model in models:
      self.training_indices[model] = train_indices
      self.testing_indices[model] = test_indices

    return [self.training_indices, self.testing_indices]
  
  def add_model(self, model, data_type, model_name = '', tag = None):
    """Adds a specified model to the dictionary according to the specified name and tags"""

    if data_type != 'raw' and data_type != 'analysis':
      raise Exception('Datatype must be raw or analysis!')
    
    if model in self.models.keys():
      print('Model name already used, removing previous reference...')
      self.drop_model(model)

    self.models[model_name] = model
    self.model_dtype[model_name] = data_type

    if not tag is None:
      if tag in self.tags.keys():
        self.tags[tag].append(model_name)
      else:
        self.tags[tag] = [model_name]

  def drop_model(self, model_name = ''):
    """Removes a model from this datastructure"""
    del(self.models[model_name])
    del(self.model_dtype[model_name])
    for key, models in self.tags.items():
      if model_name in models:
        self.tags[key].remove(model_name)
        if len(self.tags[key]) == 0:
          del(self.tags[key])
        break

  def get_numpy_dataset(self, data_selection = 'raw'):
    """Converts a given portion of the dataset to a numpy array for training.\n
    Parameters:\n
    data_selection: Either 'raw', 'labels' or 'analysis'\n
    Returns:\n
    array: numpy array of given dataset"""

    if data_selection == 'raw':
      return np.array(self.raw_dataset.drop('content_replicate', axis=1))
    
    elif data_selection == 'labels':
      return np.array(self.labels)
    
    elif data_selection == 'analysis':
      return np.array(self.analysis_dataset.drop(columns = ['content_replicate']))
    
    raise Exception('Please select raw, labels, or analysis for option to get array of')

  def train_models(self, dataset_raw = None, labels = None, model_names = None, tags=None):
    """Calls the saved models fit method, getting the necessary data if applicable - model names overrides tags"""

    # Make sure we have trainable data, either specified or generated here
    if dataset_raw is None:
      dataset_raw = self.get_numpy_dataset('raw')
      dataset_analysis = self.get_numpy_dataset('analysis')
    if labels is None:
      labels = self.get_numpy_dataset('labels')

    # If model is not specified, train on all models
    models = []
    if model_names is None:
      models = self.models.keys()
    else: models = model_names

    # Override previous import if tags are specified and train for those tags
    if (not tags is None) and (model_names is None):
      models = []
      for tag in tags:
        models += self.tags[tag]

    # Train up each model according to its own fit function
    for model in models:
      if self.model_dtype[model] == 'raw':
        x_train = dataset_raw[self.training_indices[model]]
      else:
        x_train = dataset_analysis[self.training_indices[model]]
      y_train = labels[self.training_indices[model]]
      self.models[model].fit(x = x_train, y = y_train)
  
  def get_model_predictions(self, testing_indices = None, model_names = None):
    """When a model is stored in the ML_QuIC object, get predictions from the model on test data in set\n
    Parameters:\n
    testing_indices: The indices of the testing dataset - default is whats in dataset, full datset if
    not previously specified.
    \nReturns:
    \nAn array of the model's predictions 
    """
    # Use auto made indices
    if testing_indices is None:
      testing_indices = self.testing_indices

    # If model is unspecified, get predictions from all models
    models = []
    if model_names is None:
      models = self.models.keys()
    else: models = model_names

    # Get the predictions for each model
    predictions = {}
    for model in models:
      if self.model_dtype[model] == 'raw':
        x_test = self.get_numpy_dataset('raw')[testing_indices[model]]
      else:
        x_test = self.get_numpy_dataset('analysis')[testing_indices[model]]
      y_test = self.get_numpy_dataset('labels')[testing_indices[model]]
      predictions[model] = self.models[model].predict(x_test, y_test)
      self.predictions[model] = predictions[model]

    return predictions

  def get_model_scores(self, testing_indices = None, verbose = True, model_names = None, tags = None):
    """When a model is stored in the ML_QuIC object, get metrics from the model on test data in set\n
    Parameters:\n
    testing_indices: The indices of the testing dataset - default is whats in dataset, full datset if
    not previously specified.
    \nVerbose: True - print results, false - just return results
    \nReturns:
    \nclassification_report 
    """

    # Use auto made indices
    if testing_indices is None:
      testing_indices = self.testing_indices

    # If model is unspecified, get scores for all models
    models = []
    if model_names is None:
      models = self.models.keys()
    else: models = model_names

    # If tag is specified, only train on given tag
    if (not tags is None) and (model_names is None):
      models = []
      for tag in tags:
        models += self.tags[tag]

    # Get scores and store according to model names
    scores = {}
    for model in models:
      true = self.get_numpy_dataset('labels')[testing_indices[model]]
      if self.model_dtype[model] == 'raw':
        x_test = self.get_numpy_dataset('raw')[testing_indices[model]]
      else:
        x_test = self.get_numpy_dataset('analysis')[testing_indices[model]]
      scores[model] = self.models[model].get_scores(x_test, true)

      if verbose:
        print(model + ':')
        print(scores[model])

    return scores
  
  def evaluate_replicate_performance(self, replicate_data = None, test_indices = None, test_data = None, test_labels = None, model = None, dilutions = ['', 'x01', 'x02', 'x03']):
    """Evaluates the performance of the selected model on the replicates and attempts to predict the class of samples. Requires that 
    
    Notes:\n
    test_data takes precendence over test_indices and must include test_labels or will be ignored"""
    
    # Get the dataset of samples to evaluate replicate performance if none is provided
    if test_data is None or test_labels is None:
      if test_indices is None:
        test_indices = self.testing_indices[model]
      test_labels = self.labels.iloc[test_indices]
      test_data = self.raw_dataset.iloc[test_indices] if self.model_dtype[model] == 'raw' else self.analysis_dataset.iloc[test_indices]
    
    if replicate_data is None: replicate_data = self.replicate_data
    
    # Stack each replicate for a given sample into a multi-dimensional list (not array to allow for differing numbers of replicates)
    predictions = []
    sample_list = []
    for sample in replicate_data['Sample']:
      for dilution in dilutions:
        sample_frames = test_data[test_data['content_replicate'].str.contains(sample + dilution + '_')]
        
        # Skip non-GWell items
        if len(sample_frames) == 0:
          continue
        
        sample_labels = test_labels[test_data['content_replicate'].str.contains(sample + dilution + '_')]
        dilution_replicates = np.array(sample_frames.drop(columns = ['content_replicate']))
        sample_wells = len(dilution_replicates)
        sample_predictions = np.sum(self.models[model].predict(dilution_replicates, labels = np.array(sample_labels), binary=True))
        predictions.append('{}/{}'.format(sample_predictions, sample_wells))
        sample_list.append(sample + dilution)
      
    return predictions, sample_list
        
  def evaluate_fp_performance(self, test_indices_dict = None, model_names = None, tags = None):
    """Evaluates the performance of a model on known false positives in detail. Works for 2 class and 3 class models."""

    # If tag is specified, only run for given tag
    if (not tags is None) and (model_names is None):
      models = []
      for tag in tags:
        models += self.tags[tag]
    else:
      models = model_names
 
    # Perform analysis on all models and return incorrect indices
    incorrect_indices = {}
    for model in models:
      print('-------- Results on False Positives for {} --------'.format(model))

      # Get the test set to examine
      if test_indices_dict == None:
        test_indices = self.testing_indices[model]
      else:
        test_indices = test_indices_dict[model]

      # Get the test data for analysis
      if self.model_dtype[model] == 'raw':
        x_test = self.get_numpy_dataset('raw')[test_indices]
      else:
        x_test = self.get_numpy_dataset('analysis')[test_indices]
      y_test = self.get_numpy_dataset('labels')[test_indices]

      # Get model predictions
      preds = self.models[model].predict(x_test, y_test, binary=True)
      y_test_binary = np.array(y_test == 2)
    
      preds_fp = preds[y_test == 1]

      # Case where classifier has a nonbinary negative or positive output
      if np.max(preds) > 1:
        preds_fp = np.array(preds_fp >= 1.5)
      
      # Get basic metrics on FP data
      correct_preds_fp = len(preds_fp[preds_fp < 0.5])
      print('Accuracy on False Positives: {}'.format(correct_preds_fp / len(preds_fp)))

      # Determine FP contribution to misclass rate
      incorrect_preds = len(y_test_binary[preds != y_test_binary])
      incorrect_preds_fp = len(preds_fp) - correct_preds_fp
      print('False Positives Account for {:.2f}% of total misclassifications.'.format(100 * (incorrect_preds_fp / incorrect_preds)))

      ## Evaluate nature of FPs that were misclassified
      ttt = np.array(self.analysis_dataset['TimeToThreshold'])[test_indices][y_test == 1]
      raf = np.array(self.analysis_dataset['RAF'])[test_indices][y_test == 1]
      mpr = np.array(self.analysis_dataset['MPR'])[test_indices][y_test == 1]
      ms = np.array(self.analysis_dataset['MS'])[test_indices][y_test == 1]

      # Separate statistics by prediction
      ttt_mis = np.mean(ttt[preds_fp >= 0.5])
      ttt_cor = np.mean(ttt[preds_fp < 0.5])
      raf_mis = np.mean(raf[preds_fp >= 0.5])
      raf_cor = np.mean(raf[preds_fp < 0.5])
      mpr_mis = np.mean(mpr[preds_fp >= 0.5])
      mpr_cor = np.mean(mpr[preds_fp < 0.5])
      ms_mis = np.mean(ms[preds_fp >= 0.5])
      ms_cor = np.mean(ms[preds_fp < 0.5])

      # Output statistics to user for evaluation for each model
      print('Misclassified FP Characteristics:')
      print('Average Time to Threshold: {}'.format(ttt_mis))
      print('Average RAF: {}'.format(raf_mis))
      print('Average MPR: {}'.format(mpr_mis))
      print('Average MS: {}'.format(ms_mis))

      print('Correctly Classified FP Characteristics:')
      print('Average Time to Threshold: {}'.format(ttt_cor))
      print('Average RAF: {}'.format(raf_cor))
      print('Average MPR: {}'.format(mpr_cor))
      print('Average MS: {}'.format(ms_cor))
      
      incorrect_indices[model] = test_indices[preds != y_test_binary]
      
       # Plot some examples
      fig, ax = plt.subplots(1, 1)
      
      missed_fp_indices = test_indices[y_test == 1]
      missed_fp_indices = missed_fp_indices[preds_fp == 1] # FP classified as Pos
      
      if len(missed_fp_indices) < 1: 
        print('Insufficient missed false positives to plot, attempting replacement with missed positive sample')
        missed_fp_indices = test_indices[np.logical_and(y_test_binary == 1, preds == 0)]
      ax.set_title('Misclassified FP - ' + model, fontsize = 18)
      fp_to_plot = self.get_numpy_dataset('raw')[missed_fp_indices[0]]
      ax.plot(np.arange(self.get_num_timesteps_raw()) * .75, fp_to_plot, c = 'k')
      ax.set_xlabel('Time (Hours)')
      ax.set_ylabel('Fluorescence (A.U.)')
      ax.set_ylim([0, self.max_fluorescence])
      plt.savefig('Figures/' + model + '_' + self.model_dtype[model] + '_FPs.png', transparent = False, bbox_inches = 'tight', dpi = 500)
      plt.show()
      self.fp_plots[model] = self.get_numpy_dataset('raw')[missed_fp_indices[np.random.randint(0, len(missed_fp_indices))]]

    # Print positive curve statistics to compare to false positives
    print('-------- Positive Characteristics for Reference --------')
    pos = self.analysis_dataset.iloc[(self.get_numpy_dataset('labels') == 2)] # Positive analysis dataset
    print('Time To Threshold:')
    print('\tMin: {}, Average: {}, Max: {}'.format(pos.loc[:, 'TimeToThreshold'].min(), pos.loc[:, 'TimeToThreshold'].mean(), pos.loc[:, 'TimeToThreshold'].max()))
    print('RAF:')
    print('\tMin: {}, Average: {}, Max: {}'.format(pos.loc[:, 'RAF'].min(), pos.loc[:, 'RAF'].mean(), pos.loc[:, 'RAF'].max()))
    print('MPR:')
    print('\tMin: {}, Average: {}, Max: {}'.format(pos.loc[:, 'MPR'].min(), pos.loc[:, 'MPR'].mean(), pos.loc[:, 'MPR'].max()))
    print('MS:')
    print('\tMin: {}, Average: {}, Max: {}'.format(pos.loc[:, 'MS'].min(), pos.loc[:, 'MS'].mean(), pos.loc[:, 'MS'].max()))
    
    return incorrect_indices
    
  def get_model_plots(self, model_names = None, tags = None):
    """Get plots for models according to their kinds and tags"""

    # If unspecified, get plots for all models
    models = []
    if model_names is None:
      models = self.models.keys()
    else: models = model_names

    # Only get plots for specified model groups by tag
    if (not tags is None) and (model_names is None):
      models = []
      for tag in tags:
        models += self.tags[tag]

    # Predictions for all models in list
    preds = self.get_model_predictions(model_names=models)

    # Universal color map for plots
    color_map = ['b', 'k', 'g', 'r'] 

    # Define plot categories by the type of model tag input
    for model in models:
      for key, tag_list in self.tags.items():
        if model in tag_list:
          plot_category = key.lower()
          break
      
      # Select predictions and true labels for this model
      y_pred = preds[model]
      y_true = self.get_numpy_dataset('labels')[self.testing_indices[model]]

      if np.max(y_pred) <= 1:
        y_true = np.asarray(y_true == 2, dtype=int)
      
      # Unsupervised block handled differently
      if plot_category == 'Unsupervised'.lower():
        fig, ax = self._unsupervised_plots(y_true, y_pred, model, color_map)

      # Supervised block handled differently
      elif plot_category == 'Supervised'.lower():
        fig, ax = self._supervised_plots(y_pred, y_true, model, color_map = ['b', 'k', 'g', 'r'])

      # Save plots that were generated for this model
      plt.savefig('./Figures/' + model + '.png', bbox_inches = 'tight', transparent = False, dpi = 500)

  def _supervised_plots(self, y_pred, y_true, model, color_map = ['b', 'k', 'g', 'r']):
    """Plot model outcomes for generic supervised models"""
    # Set up figure for plotting
    fig, ax = plt.subplots(2, 2)
    fig.suptitle('Classification Results for ' + model)

    # Case of binary +/- classifier
    if np.max(y_true) == 1:
      # Got positive and negative predictions for violin plot
      preds_neg = []
      preds_pos = [] 
      for i in range(len(y_true)):
        y_predicted = y_pred[i]
        y_real = y_true[i]
        if y_real == 1:
            preds_pos.append(y_predicted)
        else:
            preds_neg.append(y_predicted)

      # Plot confusion matrix
      ax[0, 0].set_title(model + ' Confusion Matrix')
      ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=(y_pred >= 0.5), ax=ax[0, 0], normalize='true', display_labels=['Negative', 'Positive'])
      
      cm_fig = ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=(y_pred >= 0.5), normalize='true', display_labels=['Negative', 'Positive']).figure_
      cm_fig.suptitle(model + ' Confusion Matrix')
      cm_fig.savefig('Figures/' + model + ' Confusion Matrix.png', bbox_inches='tight', dpi=500)

      # Violin plot on the classificatoin distributions
      ax[0, 1].set_title('Classification Distribution')
      sns.violinplot(data=[preds_neg, preds_pos], orient='h', inner='stick', cut=0, ax=ax[0, 1])

      # Plot an ROC curve for supervised learning methods
      fpr, tpr, thresh = roc_curve(y_true, y_pred)
      auc = roc_auc_score(y_true, y_pred)
      ax[1, 0].plot(fpr,tpr,label=model + ", auc=%.3f" % auc)
      ax[1, 0].legend(loc=0, fontsize = 18)

      # Obtain the testing data for reference
      ds = self.get_numpy_dataset()[self.testing_indices[model]]

      # Plot Incorrectly Classed Data
      ax[1, 1].set_title('Incorrectly Classified')
      for i,data in enumerate(ds):
        if y_pred[i] != y_true[i]:
          ax[1, 1].plot(np.arange(self.get_num_timesteps_raw()) * .75, data, c = color_map[y_true[i]])

    # Case 3 class classifier (-/FP/+)
    else:
      # Got positive and negative predictions for violin plot
      preds_neg = []
      preds_pos = [] 
      preds_fp = []
      for i in range(len(y_true)):
        y_predicted = y_pred[i]
        y_real = y_true[i]
        if y_real == 2:
          preds_pos.append(y_predicted)
        elif y_real == 1:
          preds_fp.append(y_predicted)
        else:
          preds_neg.append(y_predicted)

      y_binary = (y_true == 2)
      pred_binary = (y_pred >= 1.5)

      # Plot confusion matrix
      ConfusionMatrixDisplay.from_predictions(y_true=y_binary, y_pred=pred_binary, ax=ax[0, 0], normalize='true', display_labels=['Negative', 'Positive'])
      cm_fig = ConfusionMatrixDisplay.from_predictions(y_true=y_binary, y_pred=pred_binary, normalize='true', display_labels=['Negative', 'Positive'], colorbar='False').figure_
      cm_fig.suptitle(model + ' Confusion Matrix')
      cm_fig.savefig('Figures/' + model + ' Confusion Matrix.png', bbox_inches='tight', dpi=500)

      # Violin plot on the classificatoin distributions
      ax[0, 1].set_title('Classification Distribution')
      ax[0, 1].violinplot(dataset=[preds_neg, preds_pos], showmeans=True, showextrema=True, vert=False)

      # Plot an ROC curve for supervised learning methods
      fpr, tpr, thresh = roc_curve(y_binary, pred_binary)
      auc = roc_auc_score(y_binary, pred_binary)
      ax[1, 0].set_title('ROC Curve for ' + model)
      ax[1, 0].plot(fpr,tpr,label=model + ", auc=%.3f" % auc)
      ax[1, 0].legend(loc=0, fontsize = 18)
      
      roc_fig, roc_ax = plt.subplots(1)
      roc_ax.plot(fpr,tpr,label=model + ", auc=%.3f" % auc)
      roc_ax.legend(loc=0, fontsize = 18)
      roc_ax.set_title('ROC Curve for ' + model)
      roc_fig.savefig('./Figures/' + model + ' ROC.png', dpi=500, bbox_inches = 'tight')

      # Obtain the testing data for reference
      ds = self.get_numpy_dataset()[self.testing_indices[model]]

      # Plot Incorrectly Classed Data
      ax[1, 1].set_title('Incorrectly Classified')
      for i,data in enumerate(ds):
        if pred_binary[i] != y_binary[i]:
          ax[1, 1].plot(np.arange(self.get_num_timesteps_raw()) * .75, data, c = color_map[y_true[i]])
      
    return fig, ax

  def _unsupervised_plots(self, y_true, y_pred, model, color_map = ['b', 'k', 'g', 'r']):
    """Generate plots for a generic unsupervised model. Takes predictions and the model name"""

    # Set up figure for plotting
    fig, ax = plt.subplots(2, 2)
    fig.suptitle('Classification Results for ' + model)

    # Obtain the testing data for reference
    ds = self.get_numpy_dataset()[self.testing_indices[model]]
    
    # Generate plot of data colored by class
    ax[1, 0].set_title('Classifications')
    for i,data in enumerate(ds):
      ax[1, 0].plot(np.arange(self.get_num_timesteps_raw()), data, c = color_map[y_pred[i]])

    # Confusion Matrix plot
    ax[0, 1].set_title(model + ' Confusion Matrix')
    if np.max(y_pred) > 1:
      y_cm = np.array(np.rint(y_true) == 2, dtype = int)
      y_cm_pred = np.array(np.rint(y_pred) == 2, dtype=int)
    else:
      y_cm_pred = y_pred 
      y_cm = y_true
    
    ConfusionMatrixDisplay.from_predictions(y_true=y_cm, y_pred=y_cm_pred, ax=ax[0, 1], normalize='true', display_labels=['Negative', 'Positive'])
    cm_fig = ConfusionMatrixDisplay.from_predictions(y_true=y_cm, y_pred=y_cm_pred, normalize='true', display_labels=['Negative', 'Positive']).figure_
    cm_fig.suptitle(model + ' ' + 'Confusion Matrix')
    cm_fig.savefig('Figures/' + model + ' Confusion Matrix.png', bbox_inches = 'tight', dpi=500)
    
    ax[0, 0].set_title('Classification Clusters')
    datapoints = self.get_numpy_dataset('analysis')
    pos_datapoints = datapoints[y_cm_pred >= 0.5]
    neg_datapoints = datapoints[y_cm_pred < 0.5]
    ax[0, 0].scatter(pos_datapoints[:, 0], pos_datapoints[:, 1], c = 'g')
    ax[0, 0].scatter(neg_datapoints[:, 0], neg_datapoints[:, 1], c = 'r')

    # Plot dataset according to correct or incorrect classification
    ax[1, 1].set_title('Incorrectly Classified')
    for i,data in enumerate(ds):
      if y_pred[i] != y_true[i]:
        ax[1, 1].plot(np.arange(self.get_num_timesteps_raw()) * .75, data, c = color_map[y_true[i]])

    return fig, ax
  
  def get_group_plots_unsupervised(self, model_names = None, tags = None):
      """Get plots for models according to their kinds and tags and groups together. Made for blocks of 4 unsupervised"""

      # If unspecified, get plots for all models
      models = []
      if model_names is None:
        models = self.models.keys()
      else: models = model_names

      # Only get plots for specified model groups by tag
      if (not tags is None) and (model_names is None):
        models = []
        for tag in tags:
          models += self.tags[tag]

      # Predictions for all models in list
      preds = self.get_model_predictions(model_names=models)
      
      # Define plot structure
      cm_fig, cm_ax = plt.subplots(2, 2)
      roc_fig, roc_ax = plt.subplots(1, 1)
      roc_ax.set_xlabel('False Positive Rate')
      roc_ax.set_ylabel('True Positive Rate')
    
      # Set titles for figures   
      roc_ax.set_title('ROC Curves for Unsupervised Models')
      cm_fig.suptitle('Unsupervised Models Confusion Matrices')  
      
      # Unsupervised block handled differently
      fp_plots_to_show = []
      pos_correct_mask = []
      neg_correct_mask = []
      for i, model in enumerate(models):
            
        # Select predictions and true labels for this model
        y_pred = preds[model]
        y_true = self.get_numpy_dataset('labels')[self.testing_indices[model]]
                    
        # Convert all labels to a binary format       
        y_cm = np.array(y_true == 2, dtype = int)
        if np.max(y_pred) > 1:
          y_cm_pred = np.array(y_pred == 2, dtype = int)
        else:
          y_cm_pred = y_pred
        
        # Get a mask of correctly classified datapoints for plotting
        this_pos_correct_mask = np.zeros(len(y_cm))
        this_neg_correct_mask = np.zeros(len(y_cm))
        temp_mask = np.zeros(len(y_cm))
        for j in range(len(y_cm)):
          if y_true[j] == 0:
            temp_mask[j] = 1
          if y_true[j] == 2 and y_cm_pred[j] == 1:
            this_pos_correct_mask[j] = 1
          elif y_true[j] == 0 and y_cm_pred[j] == 0:
            this_neg_correct_mask[j] = 1
            
        pos_correct_mask.append(this_pos_correct_mask.astype(bool))
        neg_correct_mask.append(this_neg_correct_mask.astype(bool))

        # Generate the ROC curve and add to axis
        fpr, tpr, thresh = roc_curve(y_cm, y_cm_pred)
        auc = roc_auc_score(y_cm, y_cm_pred)
        roc_ax.plot(fpr,tpr,label=model + ", auc=%.3f" % auc)
        roc_ax.legend(loc=0, fontsize = 12)
        
        cm_ax[i % 2, int(i / 2)].set_title(model)
        ConfusionMatrixDisplay.from_predictions(y_true=y_cm, y_pred=y_cm_pred, ax=cm_ax[i % 2, int(i / 2)], normalize='true', display_labels=['Negative', 'Positive'], colorbar=False)

        # Get data for plotting fps and comparison with positive sample
        fp_plots_to_show.append(self.fp_plots[model])
      
      cm_fig.savefig('Figures/Unsupervised CMs.png', bbox_inches = 'tight', dpi=500)
      roc_fig.savefig('Figures/Unsupervised ROC.png', bbox_inches = 'tight', dpi=500)
      
      # Using last version of model here, should be the same indices ideally TODO - Enforce this?
      samples = self.get_numpy_dataset('raw')[self.testing_indices[model]]
      
      # Create plot and collect axes
      fig, ax = plt.subplots(2, 3)
      fig.suptitle('False Positive Classification Comparisons')
      for i, fp_ax in enumerate(fp_plots_to_show):
        ax[i%2, 1 + int(i/2)].set_title(models[i])
        ax[i%2, 1 + int(i/2)].plot(np.arange(self.get_num_timesteps_raw()) * .75, fp_ax, c = 'k')
        ax[i%2, 1 + int(i/2)].set_xlabel('Time (Hours)')
        ax[i%2, 1 + int(i/2)].set_ylabel('Fluorescence (A.U.)')
        ax[i%2, 1 + int(i/2)].set_ylim([0, self.max_fluorescence])
      
      # Find a universal positive reference
      pos_sample = None
      for i in range(len(samples)):
        if pos_correct_mask[0][i] and pos_correct_mask[1][i] and pos_correct_mask[2][i] and pos_correct_mask[3][i]:
          pos_sample = samples[i]
          break
      
      # Find a universal negative reference
      neg_sample = None
      for i in range(len(samples)):
        if neg_correct_mask[0][i] and neg_correct_mask[1][i] and neg_correct_mask[2][i] and neg_correct_mask[3][i]:
          neg_sample = samples[i]
          break
          
      # If we can't find one of the references, we stop early
      if pos_sample is None or neg_sample is None:
        sample_type = 'positive' if pos_sample is None else 'negative'
        raise Exception('Could not find ' + sample_type + ' reference sample with agreement between models!')
      
      ax[0, 0].set_title('Positive Reference Sample')
      ax[0, 0].plot(np.arange(self.get_num_timesteps_raw()) * .75, pos_sample, c = 'k')
      ax[0, 0].set_xlabel('Time (Hours)')
      ax[0, 0].set_ylabel('Fluorescence (A.U.)')
      ax[0, 0].set_ylim([0, self.max_fluorescence])
      ax[1, 0].set_title('Negative Reference Sample')
      ax[1, 0].plot(np.arange(self.get_num_timesteps_raw()) * .75, neg_sample, c = 'k')
      ax[1, 0].set_xlabel('Time (Hours)')
      ax[1, 0].set_ylabel('Fluorescence (A.U.)')
      ax[1, 0].set_ylim([0, self.max_fluorescence])
      fig.savefig('Figures/Unsupervised Samples.png', bbox_inches = 'tight', dpi=500)
      plt.show()
  
  def get_group_plots_supervised(self, model_names = None, tags = None):
    """Creates a plotting block of a positive/negative reference and 2 supervised models - someday this could be more generalized if helpful"""
    # If unspecified, get plots for all models
    models = []
    if model_names is None:
      models = self.models.keys()
    else: models = model_names

    # Only get plots for specified model groups by tag
    if (not tags is None) and (model_names is None):
      models = []
      for tag in tags:
        models += self.tags[tag]

    # Predictions for all models in list
    preds = self.get_model_predictions(model_names=models)
    
    # Define plot structure
    cm_fig, cm_ax = plt.subplots(1, 2)
    roc_fig, roc_ax = plt.subplots(1, 1)
    roc_ax.set_xlabel('False Positive Rate')
    roc_ax.set_ylabel('True Positive Rate')
  
    # Set titles for figures   
    roc_fig.suptitle('ROC Curves for Supervised Models')
    cm_fig.suptitle('Supervised Models Confusion Matrices')
    
    # Supervised block handled differently
    fp_plots_to_show = []
    pos_correct_mask = []
    neg_correct_mask = []
    for i, model in enumerate(models):
      if model == 'MLP Raw':
        continue
          
      # Select predictions and true labels for this model
      y_pred = preds[model]
      y_true = self.get_numpy_dataset('labels')[self.testing_indices[model]]
                  
      # Convert all labels to a binary format
      if np.max(y_pred) > 1:
        y_cm = np.array(np.rint(y_true) == 2, dtype = int)
        y_cm_pred = np.array(np.rint(y_pred) == 2, dtype=int)
      else:
        y_cm_pred = y_pred 
        y_cm = y_true = np.asarray(y_true == 2, dtype=int)

      # Generate the ROC curve and add to axis
      fpr, tpr, thresh = roc_curve(y_cm, y_cm_pred)
      auc = roc_auc_score(y_cm, y_cm_pred)
      roc_ax.plot(fpr,tpr,label=model + ", auc=%.3f" % auc)
      roc_ax.legend(loc=0, fontsize = 18)
      
      cm_ax[i % 2].set_title(model)
      ConfusionMatrixDisplay.from_predictions(y_true=y_cm, y_pred=y_cm_pred, ax=cm_ax[i % 2], normalize='true', display_labels=['Negative', 'Positive'], colorbar=False)
      
      # Get data for plotting fps and comparison with positive sample
      this_pos_correct_mask = np.zeros(len(y_cm))
      this_neg_correct_mask = np.zeros(len(y_cm))
      for j in range(len(y_cm)):
        if y_cm[j] == 1 and y_cm_pred[j] == 1:
          this_pos_correct_mask[j] = 1
        elif y_cm[j] == 0 and y_cm_pred[j] == 0:
          this_neg_correct_mask[j] = 1
          
      pos_correct_mask.append(this_pos_correct_mask.astype(bool))
      neg_correct_mask.append(this_neg_correct_mask.astype(bool))
      fp_plots_to_show.append(self.fp_plots[model])
      
    cm_fig.savefig('Figures/Supervised CMs.png', dpi=500, bbox_inches='tight')
    roc_fig.savefig('Figures/Supervised ROC.png', dpi=500, bbox_inches='tight')
      
    # Using last version of model here, should be the same indices ideally TODO - Enforce this?
    samples = self.get_numpy_dataset('raw')[self.testing_indices[model]]
    
    # Create plot and collect axes
    fig, ax = plt.subplots(2, 2)
    fig.suptitle('False Positive Classification Comparisons')
    for i, fp_ax in enumerate(fp_plots_to_show):
      ax[i, 1].set_title(models[i])
      ax[i, 1].plot(np.arange(self.get_num_timesteps_raw()) * .75, fp_ax, c = 'k')
      ax[i, 1].set_xlabel('Time (Hours)')
      ax[i, 1].set_ylabel('Fluorescence (A.U.)')
      ax[i, 1].set_ylim([0, self.max_fluorescence])
    
    # Find a universal positive reference
    pos_sample = None
    for i in range(len(samples)):
      if pos_correct_mask[0][i] and pos_correct_mask[1][i]:
        pos_sample = samples[i]
    
    # Find a universal negative reference
    for i in range(len(samples)):
      if neg_correct_mask[0][i] and neg_correct_mask[1][i]:
        neg_sample = samples[i]
        
    # If we can't find one of the references, we stop early
    if pos_sample is None or neg_sample is None:
      sample_type = 'positive' if pos_sample is None else 'negative'
      raise Exception('Could not find ' + sample_type + ' reference sample with agreement between models!')
    
    ax[0, 0].set_title('Positive Reference Sample')
    ax[0, 0].plot(np.arange(self.get_num_timesteps_raw()) / .75, pos_sample, c = 'k')
    ax[0, 0].set_xlabel('Time (Hours)')
    ax[0, 0].set_ylabel('Fluorescence (A.U.)')
    ax[0, 0].set_ylim([0, self.max_fluorescence])
    
    ax[1, 0].set_title('Negative Reference Sample')
    ax[1, 0].plot(np.arange(self.get_num_timesteps_raw()) / .75, neg_sample, c = 'k')
    ax[1, 0].set_xlabel('Time (Hours)')
    ax[1, 0].set_ylabel('Fluorescence (A.U.)')
    ax[1, 0].set_ylim([0, self.max_fluorescence])
    fig.savefig('Figures/Supervised Samples.png', bbox_inches='tight', dpi=500)
    plt.show()