#%%

import matplotlib.pyplot as plt
import numpy as np
import utils as utils
import os
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from itertools import combinations

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
peaks_df.to_hdf(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\Ca_flux_peaks_df.hdf', key = 'df')


# %%#########PLOTTING###########
# Load hdf files to avoid rerunning the analysis 
df = pd.read_hdf(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\Ca_flux.hdf')
peaks_df = pd.read_hdf(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\Ca_flux_peaks_df.hdf')

# %% Time until first peak, in sec
per_track = (peaks_df.dropna(subset=['FirstPeakTime']).groupby(['DATASET', 'TRACK_ID'], as_index=False)['FirstPeakTime'].min())

fig, ax = plt.subplots(figsize=(6, 4))

datasets = per_track['DATASET'].unique()
data = [per_track.loc[per_track['DATASET'] == ds, 'FirstPeakTime'] for ds in datasets]

bp = ax.boxplot(
    data,
    labels=datasets,
    showfliers=False,
    patch_artist=True   # <-- important
)

colors = plt.cm.tab10.colors

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
    patch.set_edgecolor('black')

for element in ['whiskers', 'caps', 'medians']:
    for item in bp[element]:
        item.set(color='black')

for i, y in enumerate(data, start=1):
    x = np.random.normal(i, 0.05, size=len(y))  # jitter
    ax.scatter(
    x, y,
    s=5,
    facecolor='gray',   # fill color
    edgecolor='black',  # edge color
    linewidths=0.5,     # thickness of edge
    alpha=0.7,
    zorder=3
)

ax.set_xlabel('Dataset')
ax.set_ylabel('FirstPeakTime')
ax.set_title('First peak time per track by dataset')
ax.set_ylim(0, 500)
ax.tick_params(axis='x', labelrotation=90)
ax.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260409_new\4_time_first_peak_boxplot.pdf', dpi = 600, bbox_inches='tight')
plt.savefig(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260409_new\4_time_first_peak_boxplot.png', dpi = 600, bbox_inches='tight')
stats_per_box = []

stats_per_box = (
    per_track
    .groupby('DATASET')['FirstPeakTime']
    .agg(
        n_points='count',
        mean='mean',
        SD='std',        # added: standard deviation
        median='median'
    )
    .reset_index()
)


ref = "SNAP__"

#stats
stats_per_box = (
    per_track
    .groupby('DATASET')['FirstPeakTime']
    .agg(
        n_points='count',
        mean='mean',
        SD='std',        # added: standard deviation
        median='median'
    )
    .reset_index()
)

# Calculate SEM
stats_per_box['SEM'] = stats_per_box['SD'] / np.sqrt(stats_per_box['n_points'])
# Reorder columns
stats_per_box = stats_per_box[
    ['DATASET', 'n_points', 'mean', 'SD', 'SEM', 'median']
]

# Compute p-values vs SNAP__
pvals = []

g_ref = per_track.loc[per_track['DATASET'] == ref, 'FirstPeakTime'].values

for ds in stats_per_box['DATASET']:
    if ds == ref:
        p = np.nan
    else:
        g = per_track.loc[per_track['DATASET'] == ds, 'FirstPeakTime'].values

        if len(g_ref) < 2 or len(g) < 2:
            p = np.nan
        else:
            _, p = mannwhitneyu(g_ref, g, alternative='two-sided')

    pvals.append(p)

# Add p-values to table
stats_per_box['p_vs_SNAP__'] = pvals

stats_per_box.to_csv(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260409_new\4_first_peak_time_summary.csv',
    index=False
)
plt.show()

# %% incompleteness of peaks
completeness = (
    peaks_df
    .groupby(['DATASET', 'PATH'])['Complete']
    .agg(
        n_points='count',
        n_true=lambda x: (x == True).sum(),
        n_false=lambda x: (x == False).sum(),
    )
    .reset_index()
)

# Percentages
completeness['pct_complete'] = completeness['n_true'] / completeness['n_points'] * 100
completeness['pct_incomplete'] = completeness['n_false'] / completeness['n_points'] * 100

# Filter low-count replicates
completeness2 = completeness[completeness['n_points'] >= 25].copy()

# Filtering for dense data and removing bad replicate
completeness2 = completeness2[
    completeness2['DATASET'].str.startswith('SNAP')
    |
    (
    (completeness2['DATASET'].str.startswith('dense')) 
    & 
    (completeness2['PATH'] != r'D:\Data\Chi_data\2. Ca flux\selfmade_chamber\High expression CAR\100xdilutedCD19\20250711\CART4Hi\R2\tracks_unfiltered.csv')
    )].copy()

fig, ax = plt.subplots(figsize=(6, 4))
#boxplot
sns.boxplot(data=completeness2,
    x='DATASET',
    y='pct_incomplete',
    ax=ax,
    showfliers=False)

# Overlay dots
datasets = completeness2['DATASET'].unique()
data = [completeness2.loc[completeness2['DATASET'] == ds, 'pct_incomplete'].values for ds in datasets]

for i, y in enumerate(data):
    x = np.random.normal(i, 0.05, size=len(y))
    ax.scatter(
        x, y,
        facecolor='gray',
        edgecolor='black',
        linewidths=0.4,
        s=30,
        alpha=0.7,
        zorder=3
    )

# Formatting
ax.set_ylabel('Incomplete cells (%)')
ax.set_xlabel('Dataset')
ax.set_title('Percentage of incomplete cells per replicate')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260409_new\4_incomplete_cells_boxplot_3dots.pdf', dpi = 600, bbox_inches='tight')
plt.savefig(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260409_new\4_incomplete_cells_boxplot_3dots.png', dpi = 600, bbox_inches='tight')
plt.show()

#stats
summary_incomplete = (
    completeness2
    .groupby('DATASET')['pct_incomplete']
    .agg(
        n_replicates='count',
        mean_pct_incomplete='mean',
        median_pct_incomplete='median'
    )
    .reset_index()
)

ref = "SNAP__"

if ref not in summary_incomplete['DATASET'].values:
    raise ValueError("SNAP__ not found in DATASET")

ref_vals = completeness2.loc[completeness2['DATASET'] == ref, 'pct_incomplete'].values

pvals = []

for ds in summary_incomplete['DATASET']:
    if ds == ref:
        p = np.nan
    else:
        vals = completeness2.loc[completeness2['DATASET'] == ds,'pct_incomplete'].values

        if len(vals) < 2 or len(ref_vals) < 2:
            p = np.nan
        else:
            _, p = mannwhitneyu(ref_vals, vals, alternative='two-sided')

    pvals.append(p)

summary_incomplete['p_vs_SNAP__'] = pvals

# summary_incomplete.to_csv(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260409_new\incomplete_peaks_summary_3dots.csv', index=False)

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
    (summary2['DATASET'] == 'SNAP') | (summary2['DATASET'].str.startswith('dense') & (summary2['DATASET'].str.endswith('low')) 
                                       & (summary2['PATH'] != r'D:\Data\Chi_data\2. Ca flux\selfmade_chamber\High expression CAR\100xdilutedCD19\20250711\CART4Hi\R2\tracks_unfiltered.csv'))
]

fil_to_save = (
    filtered_summary
    .groupby('DATASET')
    .agg({
        'n_tracks_total': 'sum',
        'n_tracks_peaking': 'sum',
        'pct_peaking': ['mean', 'median']
    })
    .reset_index()
)
fil_to_save.columns = ['_'.join(col).strip() for col in fil_to_save.columns.values]
fil_to_save = fil_to_save.reset_index()
fil_to_save['pct_peaking_from_sum'] = fil_to_save['n_tracks_peaking_sum'] / fil_to_save['n_tracks_total_sum']*100
fil_to_save.to_csv(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260409_new\4_Ca_peaking_percentage_summary_High_exp.csv', index=False)
utils.plot_peaking_percentage_boxplot_from_summary2(filtered_summary, color_scheme= color_scheme)
plt.savefig(r"P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig3_Ca_peaking_percent_High_exp.pdf", dpi = 600)
plt.savefig(r"P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig3_Ca_peaking_percent_High_exp.png", dpi = 600)
plt.show()
