#%%
### Imports
# Python Package Imports
import os
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

#%%
### R Function Imports
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
my_df = pd.DataFrame()
for norm_analysis in my_norm_analysis:
  with (robjects.default_converter + pandas2ri.converter).context():
    dataframe = robjects.conversion.get_conversion().rpy2py(norm_analysis)
  my_df = pd.concat((my_df, dataframe), axis=0)
dataframe = my_df
print(dataframe)

# %%
# Separate into classes and create training data
positive_samples = dataframe.loc[dataframe['content'] == 'pos']
negative_samples = dataframe.loc[dataframe['content'] == 'neg']
unlabeled_samples = dataframe.loc[dataframe['content'] != 'neg']
unlabeled_samples = unlabeled_samples.loc[unlabeled_samples['content'] != 'pos']

# Select out features we want to inspect
x_pos = np.asarray(positive_samples[['TimeToThreshold', 'RAF', 'MPR', 'MS']])
x_neg = np.asarray(negative_samples[['TimeToThreshold', 'RAF', 'MPR', 'MS']])
x_unknown = np.asarray(unlabeled_samples[['TimeToThreshold', 'RAF', 'MPR', 'MS']])

# Create labels
y_pos = np.ones(len(x_pos))
y_neg = np.zeros(len(x_neg))
y_unknown = np.full(len(x_unknown), 2)

# Combine into unified labeled dataset
x = np.concatenate((x_pos, x_neg, x_unknown))
y = np.concatenate((y_pos, y_neg, y_unknown))

print('Dataset shape: ' + str(x.shape))
print('Labels shape: ' + str(y.shape))

# %%
### Plot the data to show separability
fig, ax = plt.subplots()
ax.scatter(x_unknown[:,0], x_unknown[:,1], c='b')
ax.scatter(x_pos[:,0], x_pos[:,1], c='g')
ax.scatter(x_neg[:,0], x_neg[:,1], c='r')
ax.legend(['Unknown', 'Positive', 'Negative'])
plt.xlabel('Time To Threshold')
plt.ylabel('Rate of Amyloid Formation')
plt.show()

fig, ax = plt.subplots()
ax.scatter(x_unknown[:,2], x_unknown[:,3], c='b')
ax.scatter(x_pos[:,2], x_pos[:,3], c='g')
ax.scatter(x_neg[:,2], x_neg[:,3], c='r')
ax.legend(['Unknown', 'Positive', 'Negative'])
plt.xlabel('Max Point Ratio')
plt.ylabel('Max Slope')
plt.show()

# %%
### Attempt an unsupervised learning approach
from sklearn.cluster import KMeans
from sklearn.metrics import ConfusionMatrixDisplay

kmc = KMeans(n_clusters=2, random_state=7).fit(x)
preds = kmc.predict(x)
centers = kmc.cluster_centers_
print(centers)

# Separate into positive and negative clusters
pred_pos = x[preds == 1]
pred_neg = x[preds == 0]

# misclassed = []
# for i,pred in enumerate(preds[:12]):
#    if pred != y[i]:
#       misclassed.append(x[i])
# misclassed = np.array(misclassed)

# Plot confusion matrix on labeled data - should be perfect
# ConfusionMatrixDisplay.from_predictions(np.concatenate((y_pos, y_neg)), preds[:12])

# Show how KMeans divided the dataset
fig, ax = plt.subplots()
ax.scatter(pred_pos[:, 0], pred_pos[:, 1], c='g')
ax.scatter(pred_neg[:, 0], pred_neg[:, 1], c='r')
# ax.scatter(misclassed[:, 0], misclassed[:, 1], c='b', marker='x')
ax.legend(['Positive', 'Negative'])
plt.xlabel('Time To Threshold')
plt.ylabel('Rate of Amyloid Formation')
plt.show()

fig, ax = plt.subplots()
ax.scatter(pred_pos[:, 2], pred_pos[:, 3], c='g')
ax.scatter(pred_neg[:, 2], pred_neg[:, 3], c='r')
# ax.scatter(misclassed[:, 2], misclassed[:, 3], c='b', marker='x')
ax.legend(['Positive', 'Negative'])
plt.xlabel('Max Point Ratio')
plt.ylabel('Max Slope')
plt.show()
# %%
### Try a small supervised approach
from sklearn.svm import LinearSVC

svc = LinearSVC()
svc = svc.fit(np.concatenate((x_pos, x_neg)), np.concatenate((y_pos, y_neg)))
preds = svc.predict(x)

# Separate into positive and negative clusters
pred_pos = x[preds == 1]
pred_neg = x[preds == 0]

# misclassed = []
# pred_known = np.concatenate((preds[y == 1], preds[y==0]))
# for i,pred in enumerate(preds_known):
#    if pred != y[i]:
#       misclassed.append(x[i])
# misclassed = np.array(misclassed)

# Plot confusion matrix on labeled data - should be perfect
# ConfusionMatrixDisplay.from_predictions(np.concatenate((y_pos, y_neg)), preds[:12])

# Show how KMeans divided the dataset
fig, ax = plt.subplots()
ax.scatter(pred_pos[:, 0], pred_pos[:, 1], c='g')
ax.scatter(pred_neg[:, 0], pred_neg[:, 1], c='r')
# if len(misclassed) > 0:
#   ax.scatter(misclassed[:, 0], misclassed[:, 1], c='b', marker='x')
ax.legend(['Positive', 'Negative'])
plt.xlabel('Time To Threshold')
plt.ylabel('Rate of Amyloid Formation')
plt.show()

fig, ax = plt.subplots()
ax.scatter(pred_pos[:, 2], pred_pos[:, 3], c='g')
ax.scatter(pred_neg[:, 2], pred_neg[:, 3], c='r')
# if(len(misclassed > 0)):
#   ax.scatter(misclassed[:, 2], misclassed[:, 3], c='b', marker='x')
ax.legend(['Positive', 'Negative'])
plt.xlabel('Max Point Ratio')
plt.ylabel('Max Slope')
plt.show()
# %%
