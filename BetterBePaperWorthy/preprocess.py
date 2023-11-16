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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

base = importr('base')
utils = importr('utils')
stringr = importr('stringr')
readxl = importr('readxl')
tidyverse = importr('tidyverse')
QuICAnalysis = importr('QuICAnalysis')

class preprocess:
    """This class contains a series of Python methods which serve as an interface with rds data
    and for preprocessing the data using existing R scripts
    
    Author - Kyle Howey
    Version - November 14, 2023"""

    def __init__(self):
        """Constructor for the preprocess class. Initializes attributes and creates object"""
        self.data_dir = ''
        self.dataset = []
        self.norm_analysis = []
        self.analysis = []
        self.pandas_dataframe = pd.DataFrame()

    def import_data(self, data_dir = './', folders = None):
        """Takes in the directory data is stored in and the selected folder names 
        (automatically gotten if unspecified) and imports the data for futhre analysis.
        Note all csv files must be in group of 3 - one for plate, one for replicate, and 
        one for the raw data.\n
        Parameters:\n
        data_dir - path to directory where folders are stored\n
        folders - directories in data_dir where data is stored\n
        Returns:\n
        dataset - Imported data in dictionary form containing R dataframes"""
        self.data_dir = data_dir

        # If no folders are given, automatically search for all potential data folders
        if folders is None:
            folders = next(os.walk(data_dir))[1]
            folders = [data_dir + folder for folder in folders]
        
        # Search folders for appropriately named excel files and store together in a list of dictionaries
        dataset = []
        for i,folder in enumerate(folders):
            print(folder)
            dataset.append({})

            plate_path = glob.glob(folder + '/*plate*.xlsx')[0].replace('\\', '/')
            raw_path = glob.glob(folder + '/*raw*.xlsx')[0].replace('\\', '/')
            replicate_path = glob.glob(folder + '/*replicate*.xlsx')[0].replace('\\', '/')

            plate_data = readxl.read_xlsx(plate_path)
            raw_data = readxl.read_xlsx(raw_path)
            replicate_data = readxl.read_xlsx(replicate_path)

            # List of dictionaries mimics original R script
            dataset[i]['Plate'] = plate_data
            dataset[i]['Raw'] = raw_data
            dataset[i]['Replicate'] = replicate_data

        self.dataset = dataset
        return dataset
    
    def extract_engineered_features(self, dataset = None):
        """Performs QuicAnalysis package analysis and decomposition on specified segment of
        the dataset. This extracts engineered features including Time to Threshold, Rate
        of Amyloid Formation, Max Slope, and Max Point Ratio. See original R package for 
        more info. Anaysis data is also stored in attribute.\n
        Parameters:\n
        dataset - a dictionary of R dataframes in the format created by the import data function.
        Defaults to analyzing the entire dataset\n
        Returns:\n
        List of analysis data in format [analysis_norm, meta_analysis]"""

        # Analyzing dataframes and storing in lists
        if dataset == None:
            dataset = self.dataset

        my_norm_analysis = []
        my_analysis = []
        for data_dict in dataset:
            robjects.globalenv["AlternativeTime"] = QuICAnalysis.GetTime(data_dict['Raw'])

            meta = QuICAnalysis.GetCleanMeta(data_dict['Raw'], data_dict['Plate'], data_dict['Replicate'])
            clean_raw = QuICAnalysis.GetCleanRaw(meta, data_dict['Raw'])

            analysis = QuICAnalysis.GetAnalysis(clean_raw, 10, 10, 4)
            meta_analysis = base.cbind(meta,analysis)

            analysis_norm = QuICAnalysis.NormAnalysis(metadata = meta, data = meta_analysis, control_name = 'pos')

            my_norm_analysis.append(analysis_norm)
            my_analysis.append(meta_analysis)
        self.norm_analysis = my_norm_analysis
        self.analysis = my_analysis
        return [my_norm_analysis, my_analysis]
    
    def r2pandas(self, dataframes = None):
        """Takes a list of R dataframes for multiple (or one if passed as a one element list) 
        and converts them to a Pandas dataframe with all the data concatenated together. This
        dataframe is stored as an attribute.\n
        Parameters:\n
        dataframes - A list of dataframes to concatenate\n
        Returns:\n
        The concatenated Pandas dataframe"""
        
        my_df = pd.DataFrame()
        for sub_df in dataframes:
          with (robjects.default_converter + pandas2ri.converter).context():
            dataframe = robjects.conversion.get_conversion().rpy2py(sub_df)
          my_df = pd.concat((my_df, dataframe), axis=0)
        dataframe = my_df
        # TODO

        
