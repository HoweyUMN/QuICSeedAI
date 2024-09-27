#%%
### Imports
import pandas as pd
import numpy as np
import re

#%%
### Import the data
DATA_IN = './Data/GrinderSource/'
DATA_OUT = './Data/GrinderClean/'
DATA_G = './Data/GrinderGWells/'

raw = pd.read_csv(DATA_IN + 'combined_raw.csv')
analysis = pd.read_csv(DATA_IN + 'combined_analysis.csv')
meta = pd.read_csv(DATA_IN + 'combined_meta.csv')
replicate = pd.read_csv(DATA_IN + 'combined_replicate.csv')

# Cut out KBC wells
raw = raw[raw['content_replicate'].str.contains('KBC') == False]
analysis = analysis[analysis['content_replicate'].str.contains('KBC') == False]
replicate = replicate[replicate['Sample'].str.contains('KBC') == False]
meta = meta[meta['content'].str.contains('KBC') == False]

#%%
### Extract samples based on selected criteria
# Import CSV of wells to exclude into unused samples
unused_samples = pd.read_csv(DATA_IN + 'BigAnalysis-Excluded.csv')

# Create a mask to identify where undesired samples exist
mask = np.zeros(len(analysis))
for index, sample in unused_samples.iterrows():
    search_sample = ('^' + sample['Sample'] + '(x|_).*$')
    for index, sample in analysis.iterrows():
        if re.match(search_sample, sample['content_replicate']):
            mask[index] = 1
            
# Mask is separately created for replicate which has fewer entries            
mask_replicate = np.zeros(replicate.shape[0])
for sample in range(replicate.shape[0]):
    mask_replicate[sample] = np.max(unused_samples['Sample'] == replicate['Sample'].iloc[sample])

# Error checking and type conversion
mask = mask.astype(bool)
if not (True in mask):
    raise Exception('Failed to find any samples to keep')

# Error checking and type conversion
mask_replicate = mask_replicate.astype(bool)
if not (True in mask_replicate):
    raise Exception('Failed to find any samples to keep')

#%%
### Perform cuts
# Separate wells into G (disagreement) and G/GC (Verified)
raw_separate = raw.iloc[mask]
analysis_separate = analysis.iloc[mask]
meta_separate = meta.iloc[mask]
replicate_separate = replicate.iloc[mask_replicate]

mask = np.logical_not(mask)
mask_replicate = np.logical_not(mask_replicate)

# Get GC and non G wells for train/test
raw = raw.iloc[mask]
analysis = analysis.iloc[mask]
meta = meta.iloc[mask]
replicate = replicate.iloc[mask_replicate]

#%%
### Export to Files
raw.to_csv(DATA_OUT + 'combined_raw.csv', index = False)
analysis.to_csv(DATA_OUT + 'combined_analysis.csv', index = False)
meta.to_csv(DATA_OUT + 'combined_meta.csv', index = False)
replicate.to_csv(DATA_OUT + 'combined_replicate.csv', index = False)

raw_separate.to_csv(DATA_G + 'combined_raw.csv', index = False)
analysis_separate.to_csv(DATA_G + 'combined_analysis.csv', index = False)
meta_separate.to_csv(DATA_G + 'combined_meta.csv', index = False)
replicate_separate.to_csv(DATA_G + 'combined_replicate.csv', index = False)