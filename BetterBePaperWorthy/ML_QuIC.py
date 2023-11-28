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
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.ipython.ggplot import image_png
from rpy2.robjects.packages import importr, data
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score

base = importr('base')
utils = importr('utils')
stringr = importr('stringr')
readxl = importr('readxl')
tidyverse = importr('tidyverse')
QuICAnalysis = importr('QuICAnalysis')

class ML_QuIC:
  """This class contains a series of Python methods to processs and perform machine learning
  on RT-Quic data. R processing is currently unimplemented.

  \nAuthor - Kyle Howey
  \nVersion - November 27, 2023"""

  def __init__(self):
    self.raw_dataset = None
    self.analysis_dataset = None
    self.metadata = None
    self.labels = None
    self.training_indices = None
    self.testing_indices = None
    self.model = None
      
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

    # Error Checking
    if len(metadata) != len(raw_data) or len(metadata) != len(analysis):
      raise Exception('Dataset sizes are inconsistent - ensure all raw, meta, and analysis' +
                      ' data lines up!')

    self.raw_dataset = raw_data
    self.metadata = metadata
    self.analysis_dataset = analysis
    self.labels = metadata['final']

    return [raw_data, metadata, analysis]
  
  def separate_train_test(self, seed = 7, test_size = 0.1):
    """Separates imported data into a training set and a testing set.\n
    Returns: [training_indices, testing_indices]"""
    np.random.seed = seed

    # Shuffle the dataset to ensure randomness
    indices = np.random.permutation(len(self.raw_dataset))

    # Separate Training and Testing
    test_indices = indices[:int(test_size * len(self.raw_dataset))]
    train_indices = indices[int(test_size * len(self.raw_dataset)):]

    self.training_indices = train_indices
    self.testing_indices = test_indices

    return [train_indices, test_indices]
  
  def set_model(self, model):
    """Sets the model stored in this structure to the one specified."""
    self.model = model

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
      return np.array(self.analysis_dataset.drop('content_replicate', axis=1))
    
    raise Exception('Please select raw, labels, or analysis for option to get array of')

  def train_model(self, dataset = None, labels = None):
    if dataset is None:
      dataset = self.get_numpy_dataset('raw')
    
    if labels is None:
      labels = self.get_numpy_dataset('labels')

    self.model.fit(x = dataset, y = labels)
  
  def get_model_predictions(self, testing_indices = None):
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

    # Use entire dataset
    if testing_indices is None:
      testing_indices = np.arange(len(self.raw_dataset))

    return self.model.predict(self.get_numpy_dataset('raw')[testing_indices])

  def get_model_scores(self, testing_indices = None, verbose = True):
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

    # Use entire dataset
    if testing_indices is None:
      testing_indices = np.arange(len(self.raw_dataset))

    # Get true/pred class
    true = self.get_numpy_dataset('labels')[testing_indices]
    data = self.get_numpy_dataset('raw')[testing_indices]
    true = (true >= 2)

    scores = self.model.get_scores(data, true)
    if verbose:
      print(scores)

    return scores
