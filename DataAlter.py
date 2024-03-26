import pandas as pd
import numpy as np
import re

# Import the data for separation
raw = pd.read_csv('./Data/BigAnalysisSource/combined_raw.csv')
analysis = pd.read_csv('./Data/BigAnalysisSource/combined_analysis.csv')
meta = pd.read_csv('./Data/BigAnalysisSource/combined_meta.csv')
replicate = pd.read_csv('./Data/BigAnalysisSource/combined_replicate.csv')

# Import the list of samples to withhold
unused_samples = pd.read_csv('./Data/BigAnalysisSource/BigAnalysis-Excluded.csv')

mask = np.zeros(len(analysis))
for index, sample in unused_samples.iterrows():
    search_sample = ('^' + sample['Sample'] + '(x|_).*$')
    for index, sample in analysis.iterrows():
        if re.match(search_sample, sample['content_replicate']):
            mask[index] = 1
    
mask = mask.astype(bool)
if not (True in mask):
    raise Exception('AH')

mask_replicate = []
for sample in replicate.Sample:
    mask_replicate.append(unused_samples['Sample'].str.contains(sample))


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

# Export to Files
raw.to_csv('./Data/BigAnalysis/combined_raw.csv', index = False)
analysis.to_csv('./Data/BigAnalysis/combined_analysis.csv', index = False)
meta.to_csv('./Data/BigAnalysis/combined_meta.csv', index = False)
replicate.to_csv('./Data/BigAnalysis/combined_replicate.csv', index = False)

raw_separate.to_csv('./Data/BigAnalysisGWells/combined_raw.csv', index = False)
analysis_separate.to_csv('./Data/BigAnalysisGWells/combined_analysis.csv', index = False)
meta_separate.to_csv('./Data/BigAnalysisGWells/combined_meta.csv', index = False)
replicate_separate.to_csv('./Data/BigAnalysisGWells/combined_replicate.csv', index = False)