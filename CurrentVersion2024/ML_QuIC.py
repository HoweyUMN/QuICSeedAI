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
from sklearn.decomposition import KernelPCA
import multiprocessing
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, roc_auc_score
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
      predictions[model] = self.models[model].predict(x_test)
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
  
  def evaluate_replicate_performance(self, test_indices_dict = None, model_names = None, tags = None):
    # If tag is specified, only run given tag
    if (not tags is None) and (model_names is None):
      models = []
      for tag in tags:
        models += self.tags[tag]
    else:
      models = model_names

    # Perform analysis for each model
    for model in models:

      # Get the test set to examine
      if test_indices_dict == None:
        test_indices = self.testing_indices[model]
      else:
        test_indices = test_indices_dict[model]
        
      # Get testing data
      if self.model_dtype[model] == 'raw':
        test_df = self.raw_dataset.iloc[test_indices]
      else:
        test_df = self.analysis_dataset.iloc[test_indices]
        
      # Append to dataset for easier function
      preds_data = self.models[model].predict(self.get_numpy_dataset('raw')[test_indices], binary = True)
      test_df['Predictions'] = preds_data
      test_data = self.get_numpy_dataset('labels')[test_indices] # Required to avoid a useless error
      test_df['True'] = test_data
      test_df = test_df.iloc[test_indices]
      
      # Remove controls from dataset for better performance
      test_df = test_df[~test_df['content_replicate'].str.contains('pos', na = False)]
      test_df = test_df[~test_df['content_replicate'].str.contains('neg', na = False)]
      
      replicates = [] # list of dataframes only containing replicates
      for sample in self.replicate_data['Sample']:
        replicate_df = test_df[test_df.content_replicate.str.contains('^' + sample + 'x')]
        if len(replicate_df) < 3: continue # The replicate was broken up, so we won't get useful information
        replicates.append(replicate_df)

      max_num_replicates = len(self.replicate_data.columns) - 3 # Remove metadata from count

      # Extract data about each replicate
      correct_preds = 0
      correct_by_replicate = np.zeros(max_num_replicates)
      for replicate_df in replicates:
        preds = []
        for i in range(max_num_replicates):
          if i != 0:
            # Extract predictions
            replicate_id = '%02d' % i
            preds.append(test_df[test_df['content_replicate'].str.contains(replicate_id)]['Predictions'].iloc[0])
          else:
            preds.append(test_df[~test_df['content_replicate'].str.contains('x')]['Predictions'].iloc[0])
        
        # Check how predictions compare
        sum = 0
        count = 0
        for i,pred in enumerate(preds):
          if pred == -1: continue # Not in dataframe

          count += 1
          # If prediction is correct, add to sum
          if (pred == 1 and replicate_df['True'].iloc[0] == 2) or (pred == 0 and replicate_df['True'].iloc[0] < 2):
            sum += 1
            correct_by_replicate[i] += 1
        
        if round(sum / count) == 1:
          if replicate_df['True'].iloc[0] == 2:
            correct_preds += 1
        else:
          if replicate_df['True'].iloc[0] < 2:
            correct_preds += 1
            
      print(correct_by_replicate)
      print(correct_preds / len(replicates))
            
        
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
      preds = self.models[model].predict(x_test, binary=True)
      preds_fp = preds[y_test == 1]

      # Case where classifier has a binary negative or positive output
      if np.max(preds) > 1:
        preds_fp = np.array(preds_fp >= 1.5)
      y_test_binary = np.array(y_test == 2)
      
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
      plt.savefig('./Figures/' + model + '_' + self.model_dtype[model] + '.png', bbox_inches = 'tight')

  def _supervised_plots(self, y_pred, y_true, model, color_map = ['b', 'k', 'g', 'r']):
    """Plot model outcomes for generic supervised models"""
    # Set up figure for plotting
    fig, ax = plt.subplots(2, 2)
    fig.tight_layout(pad=3.0)
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
      dtype = 'Raw' if self.model_dtype[model] == 'raw' else 'Feature Extracted'
      
      ax[0, 0].set_title(model + ' ' + dtype + ' Confusion Matrix')
      ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=(y_pred >= 0.5), ax=ax[0, 0], normalize='true', display_labels=['Negative', 'Positive'])

      # Violin plot on the classificatoin distributions
      ax[0, 1].set_title('Classification Distribution')
      sns.violinplot(data=[preds_neg, preds_pos], orient='h', inner='stick', cut=0, ax=ax[0, 1])

      # Plot an ROC curve for supervised learning methods
      ax[1, 0].set_title('ROC Curve, AUC: {0:.2f}'.format(roc_auc_score(y_true, y_pred)))
      RocCurveDisplay.from_predictions(y_true, y_pred, ax=ax[1, 0])

      # Obtain the testing data for reference
      ds = self.get_numpy_dataset()[self.testing_indices[model]]

      # Plot Incorrectly Classed Data
      ax[1, 1].set_title('Incorrectly Classified')
      for i,data in enumerate(ds):
        if y_pred[i] != y_true[i]:
          ax[1, 1].plot(np.arange(self.get_num_timesteps_raw()), data, c = color_map[y_true[i]])

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
      ax[0, 0].set_title('Binary Confusion Matrix')
      ConfusionMatrixDisplay.from_predictions(y_true=y_binary, y_pred=pred_binary, ax=ax[0, 0], normalize='true', display_labels=['Negative', 'Positive'])

      # Violin plot on the classificatoin distributions
      ax[0, 1].set_title('Classification Distribution')
      ax[0, 1].violinplot(dataset=[preds_neg, preds_pos], showmeans=True, showextrema=True, vert=False)

      # Plot an ROC curve for supervised learning methods
      ax[1, 0].set_title('ROC Curve, AUC: {0:.2f}'.format(roc_auc_score(y_binary, pred_binary)))
      RocCurveDisplay.from_predictions(y_binary, pred_binary, ax=ax[1, 0])

      # Obtain the testing data for reference
      ds = self.get_numpy_dataset()[self.testing_indices[model]]

      # Plot Incorrectly Classed Data
      ax[1, 1].set_title('Incorrectly Classified')
      for i,data in enumerate(ds):
        if pred_binary[i] != y_binary[i]:
          ax[1, 1].plot(np.arange(self.get_num_timesteps_raw()), data, c = color_map[y_true[i]])
      
    return fig, ax

  def _unsupervised_plots(self, y_true, y_pred, model, color_map = ['b', 'k', 'g', 'r']):
    """Generate plots for a generic unsupervised model. Takes predictions and the model name"""

    # Set up figure for plotting
    fig, ax = plt.subplots(2, 2)
    fig.tight_layout(pad=3.0)
    fig.suptitle('Classification Results for ' + model)

    # For unsupervised, sometimes labels are swapped. This corrects for this fact
    if np.sum(np.where(y_pred == y_true, 1, 0)) < 0.5 * len(y_true):
      y_pred = 1 - y_pred

    # Obtain the testing data for reference
    ds = self.get_numpy_dataset()[self.testing_indices[model]]
    
    # Generate plot of data colored by class
    ax[1, 0].set_title('Classifications')
    for i,data in enumerate(ds):
      ax[1, 0].plot(np.arange(self.get_num_timesteps_raw()), data, c = color_map[y_pred[i]])

    # Ensures labels picked by unsupervised model are in appropriate ranges
    y_pred = y_pred % 3
    # Confusion Matrix plot
    dtype = 'Raw' if self.model_dtype[model] == 'raw' else 'Feature Extracted'
      
    ax[0, 1].set_title(model + ' ' + dtype + ' Confusion Matrix')
    if np.max(y_pred) > 1:
      y_cm = np.array(np.rint(y_true) == 2, dtype = int)
      y_cm_pred = np.array(np.rint(y_pred) == 2, dtype=int)
    else:
      y_cm_pred = y_pred 
      y_cm = y_true
    ConfusionMatrixDisplay.from_predictions(y_true=y_cm, y_pred=y_cm_pred, ax=ax[0, 1], normalize='true', display_labels=['Negative', 'Positive'])
    
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
        ax[1, 1].plot(np.arange(self.get_num_timesteps_raw()), data, c = color_map[y_true[i]])

    return fig, ax