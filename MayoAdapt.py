#%%
### Imports
import pandas as pd
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

#%%
### Data Loading
df = pd.read_csv('./Data/Shahnawaz/Shahnawaz_raw.csv')

ds = np.array(df.drop(columns=['content_replicate']))
timesteps = np.array(df.drop(columns=['content_replicate']).columns, dtype=float)

#%%
### Interpolation
f = interpolate.interp1d(timesteps, ds, axis = - 1)
newTS = np.arange(0, 361, 360 / 64)
new_ds = f(newTS)

# %%
# Plot some examples for verification
plt.plot(np.arange(0, 361, 360 / 15), ds[5], c = 'r')
plt.plot(np.arange(0, 361, 360 / 64), new_ds[5], c = 'k', linestyle = 'dashed')
plt.xlabel('Hours')
plt.ylabel('Fluorescence')
plt.title('Interpolated Data')

# %%
### Save new data
new_df = pd.DataFrame()
new_df['content_replicate'] = df['content_replicate']

for i,ts in enumerate(newTS.astype(str)):
    new_df[ts] = new_ds[:, i]

new_df.to_csv('./Data/FPCA_Interp/Shahnawaz_raw.csv', index = False)
# %%
