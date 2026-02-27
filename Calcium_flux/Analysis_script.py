#%%

import matplotlib.pyplot as plt
import numpy as np
import utils as utils
import os
import numpy as np
import pandas as pd
# %% STEP 1: extract metadata from file paths 
path = r'D:\Data\Chi_data\2. Ca flux'  # Root directory to scan
csv_dirs = []
#Look for CSV files 
for root, dirs, files in os.walk(path):
    for file in files:
        if file.lower().endswith('.csv'):
            csv_dirs.append(os.path.join(root, file))
del dirs, files, root, file
#Extract metadata from file paths
metadata = []
for path in csv_dirs:
    if 'tracks_unfiltered.csv' not in path:
        continue
    neg_control = False
    well = np.nan
    expression = np.nan
    surface = np.nan
    CAR = np.nan
    if '8well_chamber' in path: 
        well = '8well'
    elif 'selfmade_chamber' in path: 
        well = 'self'
    if '100xdiluted' in path: 
        surface = 'denseCD19'
    elif '1000xdiluted' in path or '1500xdiluted' in path: 
        surface = 'midCD19'
    elif '3000xdiluted' in path or '6000xdiluted' in path: 
        surface = 'sparseCD19'  
    if 'High' in path: 
        expression = 'high'
    elif 'Low' in path: 
        expression = 'low'
    if 'CART3' in path: 
        CAR = 'CART3'
    elif 'CART4' in path:
        CAR = 'CART4'
    elif 'Jurkat' in path: 
        CAR= 'jurkat'
    if 'SNAP' in path: 
        surface = 'SNAP'
        expression = ''
        CAR = ''
    if 'negative_ctrl' in path: 
        well = np.nan
        expression = np.nan
        surface = np.nan
        CAR = np.nan
        neg_control = True
    data = {'path': path, 'well': well, 'expression': expression, 'surface': surface, 'CAR': CAR, 'neg_control': neg_control}
    
    metadata.append(data)
    del well, expression, surface, CAR, neg_control, data, path
df2 = pd.DataFrame(metadata)
del metadata     
# Filter filepaths for analysis and create their categories. 
files_info = (
    df2[(df2.well == 'self') & (df2.CAR != 'jurkat') & (~df2['path'].str.contains('20251107')) ]
    .set_index('path')[['surface', 'CAR', 'expression']]
    .astype(str)
    .agg('_'.join, axis=1)
    .reset_index()            # bring 'path' back as a column
    .apply(tuple, axis=1)     # make each row a tuple (path, surface_CAR)
    .tolist()                 # convert to list of tuples
)
# %% STEP 2: define analysis parameters
framerate = 1/3  # Frame rate used in analysis ([1/s])
min_track_duration = 0.25  # Minimum track duration (Fraction of total duration)
outlier_percentile = 0.01  # Percentile for filtering out outliers
prominence_value_final = 0.35  # Prominence threshold for peak filtering
threshold_value_final = -0.005  # Slope value for trace classification
# %% STEP 3: Preparing Data
# Filter Tracks Shorter than Minimum Duration and the Outliers
df = utils.process_multiple_csv(files_info, min_track_duration=min_track_duration,
                                outlier_percentile=outlier_percentile, framerate=framerate)
# Processing CSV files, filtering, and smoothing intensity traces
df = utils.smooth_traces(
    df, intensity_col='NORM_MEAN_INTENSITY_CH1', window_length=11)
df = utils.calculate_derivative(
    df, intensity_col='SMOOTH_NORM_MEAN_INTENSITY_CH1')

# Calculating statistical measures for the processed data
df['STD_DIFF_SMOOTH_NORM_MEAN_INTENSITY_CH1'] = df.groupby(
    ['DATASET', 'TRACK_ID'])['DIFF_SMOOTH_NORM_MEAN_INTENSITY_CH1'].transform('std')
df['STD_NORM_MEAN_INTENSITY_CH1'] = df.groupby(['DATASET', 'TRACK_ID'])[
    'NORM_MEAN_INTENSITY_CH1'].transform('std')
df['AVG_NORM_MEAN_INTENSITY_CH1'] = df.groupby(['DATASET', 'TRACK_ID'])[
    'NORM_MEAN_INTENSITY_CH1'].transform('mean')
df['CV_NORM_MEAN_INTENSITY_CH1'] = df.groupby(['DATASET', 'TRACK_ID'])['NORM_MEAN_INTENSITY_CH1'].transform(
    lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else np.nan)

# Output the count of unique tracks by dataset
print(
    f"Number of calcium traces: {df.groupby('DATASET')['TRACK_ID'].nunique()}")


# %% STEP 4 - Peak Detection and Trace Classification
# Peak detection
# Peaks are found using scipy.signal.find_peaks, with a prominence value of 0.1. Since sometimes the peaks do not 
# completely go down to base level after the peak, we then exchange the prominence value to the difference between the intensity in 
# the peak and the base level. These new prominence values are then used to filter the peaks aginst the prominence threshold defined in STEP2.
peaks_df, df = utils.find_all_peaks(
    df, prominence_value=prominence_value_final, framerate=framerate)

total_peaks = peaks_df.groupby('DATASET')['TRACK_ID'].nunique().sum()
total_tracks = df.groupby('DATASET')['TRACK_ID'].nunique().sum()
print(
    f"Using Prominence Value {prominence_value_final:.3f}: Found Peaks in {total_peaks:d} of {total_tracks:d} Traces ({total_peaks/total_tracks:.2%})")

# Slope Threshold
# traces without peaks are then classified as decaying or resting
# Mark tracks as DECAYING using:
    #   - if decay between 80th and 20th percentile is larger than 0.2 (traces are normalized)  OR
    #   - if in a linear regression rule: R^2 > r2_thresh AND slope < slope_thresh
    #then the traces is marked as DECAYING. 
df = utils.tracks_split_by_regression(df,frame_col='FRAME_SYNC',ycol='SMOOTH_NORM_MEAN_INTENSITY_CH1', peak_col='PEAK', rel_drop_thresh=0.2, 
                               r2_thresh=0.5,slope_thresh=threshold_value_final)

df.to_hdf(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\Ca_flux.hdf', key = 'df')

# %% Final plots 

# % of cells in each category, and % of cells peaking. 
fig, ax = plt.subplots(figsize=(6, 4))
summary = utils.plot_cell_status_bars_percent(df, colors=('yellowgreen', 'wheat', 'gray'), ax=ax)
fig, summary = utils.plot_peaking_percentage_boxplot(df)
plt.show()


# %% Peak prominence 
fig, ax = plt.subplots(figsize=(6, 6))

# Prepare data grouped by dataset
datasets = peaks_df['DATASET'].unique()
data = [peaks_df.loc[peaks_df['DATASET'] == ds, 'Prominence'].dropna() for ds in datasets]

# Make boxplot
bp = ax.boxplot(data, labels=datasets, showfliers=False)

# Overlay individual points (optional)
for i, y in enumerate(data, start=1):
    x = np.random.normal(i, 0.05, size=len(y))  # jitter for visibility
    ax.scatter(x, y, alpha=0.6, s=10, edgecolor='k', linewidths=0.3)

# Format
ax.set_xlabel('Dataset')
ax.set_ylabel('Peak Prominence')
ax.set_title('Distribution of Peak Prominence per Dataset')
ax.set_ylim(0, 1)
ax.tick_params(axis='x', labelrotation=90)
ax.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()
# %% Time until first peak, in sec

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1) One FirstPeakTime per (DATASET, TRACK_ID)
#    Use the earliest FirstPeakTime per track (change to .max() if you prefer)
per_track = (
    peaks_df
    .dropna(subset=['FirstPeakTime'])
    .groupby(['DATASET', 'TRACK_ID'], as_index=False)['FirstPeakTime']
    .min()
)

# 2) Boxplot per DATASET + jittered points
fig, ax = plt.subplots(figsize=(6, 4))
datasets = per_track['DATASET'].unique()
data = [per_track.loc[per_track['DATASET'] == ds, 'FirstPeakTime'] for ds in datasets]

ax.boxplot(data, labels=datasets, showfliers=False)
for i, y in enumerate(data, start=1):
    x = np.random.normal(i, 0.05, size=len(y))  # jitter
    ax.scatter(x, y, s=12, alpha=0.7, edgecolor='k', linewidths=0.3)

ax.set_xlabel('Dataset')
ax.set_ylabel('FirstPeakTime')
ax.set_title('First peak time per track by dataset')
ax.set_ylim(0, 500)
ax.tick_params(axis='x', labelrotation=90)
ax.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()

plt.savefig(r'D:\Data\Chi_data\2. Ca flux\selfmade_chamber\time_first_peak_box.pdf', dpi = 600, bbox_inches='tight')
plt.show()

# %% Same but violin plot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Filter so each TRACK_ID appears once per DATASET ---
per_track = (
    peaks_df
    .dropna(subset=['FirstPeakTime'])
    .groupby(['DATASET', 'TRACK_ID'], as_index=False)['FirstPeakTime']
    .min()
)

# --- Violin plot ---
fig, ax = plt.subplots(figsize=(6, 4))
datasets = per_track['DATASET'].unique()
data = [per_track.loc[per_track['DATASET'] == ds, 'FirstPeakTime'] for ds in datasets]

parts = ax.violinplot(data, showmedians=True, showextrema=False)

# Optional: color violins
for pc in parts['bodies']:
    pc.set_facecolor('white')
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

# Overlay dots for individual tracks
for i, y in enumerate(data, start=1):
    x = np.random.normal(i, 0.05, size=len(y))  # jitter
    ax.scatter(x, y, s=10, alpha=0.6, edgecolor='k', linewidths=0.3)

# --- Labels and style ---
ax.set_xticks(np.arange(1, len(datasets) + 1))
ax.set_xticklabels(datasets, rotation=90)
ax.set_ylim(0, 600)
ax.set_xlabel('Dataset')
ax.set_ylabel('First Peak Time')
# ax.set_title('Distribution of FirstPeakTime per dataset')
ax.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()


plt.savefig(r'D:\Data\Chi_data\2. Ca flux\selfmade_chamber\time_first_peak_violin.pdf', dpi = 600, bbox_inches='tight')
plt.show()

# %% Final plot for paper
#With filtering and proper color code.
fig, summary = utils.plot_peaking_percentage_boxplot(df)#%%
color_scheme = {'SNAP':'#bfc1c3ff', 
                'denseCD19_CART3_low': '#A4D5D8', 
                'denseCD19_CART3_high': '#3FB2B2', 
                'denseCD19_CART4_low': '#E2A1E2', 
                'denseCD19_CART4_high': '#C361C6', 
                'midCD19_CART3_low': '#A4D5D8', 
                'midCD19_CART3_high': '#3FB2B2', 
                'midCD19_CART4_low': '#E2A1E2', 
                'midCD19_CART4_high': '#C361C6', 
                'sparseCD19_CART3_low': '#A4D5D8', 
                'sparseCD19_CART3_high': '#3FB2B2', 
                'sparseCD19_CART4_low': '#E2A1E2', 
                'sparseCD19_CART4_high': '#C361C6'}

summary2 = summary[summary.n_tracks_total >= 30].copy()   


# Make sure DATASET is string type
summary2['DATASET'] = summary2['DATASET'].astype(str)

# Replace any label containing 'SNAP' with 'SNAP'
summary2['DATASET'] = summary2['DATASET'].str.replace(r'^SNAP.*$', 'SNAP', regex=True)


filtered_summary = summary2[
    (summary2['DATASET'] == 'SNAP') | (summary2['DATASET'].str.startswith('dense') & (summary2['DATASET'].str.endswith('low')))
]

utils.plot_peaking_percentage_boxplot_from_summary2(filtered_summary, color_scheme= color_scheme)
plt.savefig(r"P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig3_Ca_peaking_percent.pdf", dpi = 600)
plt.savefig(r"P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig3_Ca_peaking_percent.png", dpi = 600)
plt.show()
