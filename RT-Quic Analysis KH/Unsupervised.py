#%%
### Imports
# Python Package Imports
import preprocess
import os
import glob
os.environ['R_HOME'] = 'C:\PROGRA~1\R\R-43~1.1'
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.ipython.ggplot import image_png
from rpy2.robjects.packages import importr, data
import numpy as np
import pandas as pd

# R Function Imports
base = importr('base')
utils = importr('utils')
stringr = importr('stringr')
readxl = importr('readxl')
tidyverse = importr('tidyverse')
QuICAnalysis = importr('QuICAnalysis')

# %%
### Data Import

# Get a list of folders to search for data
folders = next(os.walk('./data'))[1]
folders = ['./data/' + folder for folder in folders]

# Search folders for appropriately named excel files and store together in a list of dictionaries
dataset = []
for i,folder in enumerate(folders):
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

# %%
### Data Analysis and Decomposition

# Analyzing dataframes and storing in lists
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
# %%
### Convert R Dataframes to Pandas Dataframes
with (robjects.default_converter + pandas2ri.converter).context():
  dataframe = robjects.conversion.get_conversion().rpy2py(my_norm_analysis[0])
print(dataframe['TimeToThreshold'])

# %%
