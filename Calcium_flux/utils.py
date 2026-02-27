import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import find_peaks, peak_widths
import seaborn as sns
from scipy.stats import linregress
from scipy.stats import mannwhitneyu
from itertools import combinations


def process_multiple_csv(files, min_track_duration, outlier_percentile, framerate):
    '''
    Processes multiple CSV files containing track data, applies normalization, and filters outliers.
    Normalization is done across the whole dataset.
    '''
    dataframes = []
    max_track_id = 0  # Initialize the maximum track_id found so far
    for file_path, dataset in files:
        df = process_data(file_path, min_track_duration, framerate, dataset)
        if df is None or df.empty:
            print(f"Skipping {file_path} — empty DataFrame")
            continue
        df['TRACK_ID_ORIGINAL'] = df['TRACK_ID']
        
        df['TRACK_ID'] = df['TRACK_ID'] + max_track_id

        # Update the maximum track_id for the next iteration
        max_track_id = df['TRACK_ID'].max()+1
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    # Normalize intensity by dividing each intensity value by the maximum intensity
    # found overall in the whole dataset
    max_value = combined_df.groupby(["DATASET", "TRACK_ID"]).MEAN_INTENSITY_CH1.transform('max').max()
    combined_df['NORM_MEAN_INTENSITY_CH1'] = combined_df['MEAN_INTENSITY_CH1'] / max_value
    

    # Calculate average intensity per trace
    combined_df['AVG_MEAN_INTENSITY_CH1'] = combined_df.groupby(
        ["DATASET", "TRACK_ID"])['NORM_MEAN_INTENSITY_CH1'].transform('mean')
    
    # Define a function to filter out tracks based on intensity quantiles, removing extremes
    def filter_outliers(group):
        lower_quantile = group['AVG_MEAN_INTENSITY_CH1'].quantile(
            outlier_percentile)
        upper_quantile = group['AVG_MEAN_INTENSITY_CH1'].quantile(
            1-outlier_percentile)
        return group[(group['AVG_MEAN_INTENSITY_CH1'] > lower_quantile) & (group['AVG_MEAN_INTENSITY_CH1'] < upper_quantile)]

    # Apply the filtering function to each dataset group
    filtered_df = combined_df.groupby('DATASET').apply(
        filter_outliers).reset_index(drop=True)

    return filtered_df


def process_data(csv_file, min_track_duration, framerate=1, datset=None):
    '''
    Loads and processes data from a single CSV file by dropping unnecessary rows and columns,
    converting data types, and aligning tracks based on frames.
    '''
    # Load data, omitting the three rows and the first column for a cleaner dataset
    # Read the CSV file and store its content in df
    df = pd.read_csv(csv_file, index_col=False, low_memory=False)
    df = df.drop(df.index[[0, 1, 2]])  # Drop the first three rows
    df = df.drop(df.columns[0], axis=1)  # Drop the first column
    

    # Convert remaining data in the DataFrame to float type
    df = df.astype(float)
    df['PATH'] = csv_file
    # Get Track length by counting number of TRACK_ID occurrences
    df['TRACK_LENGTH'] = df.TRACK_ID.map(df.TRACK_ID.value_counts())
    num_tracks = df['TRACK_ID'].nunique()
    # print(f"Number of tracks: {num_tracks}")
    # Filter out rows with missing TRACK_ID and short tracks
    df = df.dropna(subset=['TRACK_ID'])
    
    num_tracks = df['TRACK_ID'].nunique()
    # print(f"Number of tracks: {num_tracks}")

    df = df.loc[df.TRACK_LENGTH > df.FRAME.max() * min_track_duration]
    
    num_tracks = df['TRACK_ID'].nunique()
    # print(f"Number of tracks: {num_tracks}")


    # Create a column with aligned and filled frame numbers so all tracks start at 0
    df['FRAME_SYNC'] = df.groupby(
        'TRACK_ID')['FRAME'].transform(lambda x: x - x.min())
    df['TIME'] = df.groupby('TRACK_ID')[
        'FRAME'].transform(lambda x: x/framerate)
    df['TIME_SYNC'] = df.groupby(
        'TRACK_ID')['FRAME_SYNC'].transform(lambda x: x/framerate)

    # Add a new column to distinguish the data source
    df['DATASET'] = datset

    return df


def apply_savgol_filter(series, window_length, polyorder):
    """
    Function to apply Savitzky-Golay filter to a series
    """
    # Ensure window_length is odd and <= length of the series
    if window_length % 2 == 0:
        window_length += 1
    if window_length > len(series):
        window_length = len(series) if len(
            series) % 2 == 1 else len(series) - 1
    # Apply the Savitzky-Golay filter with the specified window length and polynomial order.
    return savgol_filter(series, window_length=window_length, polyorder=polyorder)


def smooth_traces(df, track_id_col='TRACK_ID', intensity_col='NORM_MEAN_INTENSITY_CH1', frame_col='FRAME', window_length=9, polyorder=2):
    """
    Applies Savitzky-Golay smoothing to an intensity column for each track.
    """
    # Ensure the DataFrame is sorted by the specified frame column
    df_sorted = df.sort_values(by=['DATASET', track_id_col, frame_col])

    # Apply the Savitzky-Golay filter to the specified intensity column for each track
    smoothed_col_name = 'SMOOTH_' + intensity_col
    df_sorted[smoothed_col_name] = df_sorted.groupby(['DATASET', track_id_col])[
        intensity_col].transform(lambda x: savgol_filter(x, window_length, polyorder))

    return df_sorted


def calculate_derivative(df, intensity_col, track_id_col='TRACK_ID'):
    """
    Calculates the derivative of the smoothed intensity values and computes absolute average and standard deviation.
    """
    # Calculate the derivative of the smoothed intensity values
    diff_col_name = 'DIFF_' + intensity_col
    df[diff_col_name] = df.groupby(['DATASET', track_id_col])[
        intensity_col].transform(lambda x: x.diff())

    return df


def pivot_intensity_data(df, sort_col='AVG_MEAN_INTENSITY_CH1', track_id_col='TRACK_ID', intensity_col='NORM_MEAN_INTENSITY_CH1', frame_col='FRAME'):
    '''
    Processes and pivots a DataFrame to organize intensity data into a matrix format, 
    facilitating further analysis such as heatmap visualization.

    '''
    # Sort the DataFrame by 'TRACK_ID' and 'FRAME'
    df_sorted = df.sort_values(by=[sort_col, track_id_col, frame_col])

    # Perform the pivot operation
    pivot_df = df_sorted.pivot(
        index=track_id_col, columns=frame_col, values=intensity_col)

    # Reindex the pivoted DataFrame to match the order of TRACK_IDs in df_sorted
    pivot_df = pivot_df.reindex(df_sorted[track_id_col].unique())

    # Fill missing values in the pivoted DataFrame with interpolation
    pivot_df = pivot_df.interpolate(limit_area='inside', axis=1)

    # Fill NAs with 0
    pivot_df_filled = pivot_df.fillna(0)

    return pivot_df_filled


def plot_intensity_heatmap(df, sort_col='AVG_MEAN_INTENSITY_CH1', track_id_col='TRACK_ID', intensity_col='NORM_MEAN_INTENSITY_CH1', frame_col='TIME_SYNC', cmap='viridis', vmin=None, vmax=None, ax=None):
    """
    Plots a heatmap of intensity over time for each cell.

    """
    df_pivoted = pivot_intensity_data(
        df, sort_col=sort_col, track_id_col=track_id_col, intensity_col=intensity_col, frame_col=frame_col)
    sns.heatmap(df_pivoted, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax)

    # Remove y-ticks and y-labels
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Set x-ticks to the min and max frame
    ax.set_xticks([df[frame_col].min(), df[frame_col].max()])
    ax.set_xticklabels([int(df[frame_col].min()), int(df[frame_col].max())])
    ax.set_xlabel('Frames')
    ax.set_ylabel('Cell (Track ID)')


def plot_histogram(df, column='AVG_MEAN_INTENSITY_CH1', bins='fd', ax=None):
    """
    Plots a histogram of average intensity values.
    """
    ax.hist(df[column], bins=bins, color='skyblue', edgecolor='black')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')


def find_trace_peaks(df, track_id, prominence, intensity_col='NORM_MEAN_INTENSITY_CH1', framerate=1):
    """
    Process peaks for a single track ID in the DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - track_id: The track ID to process.
    - prominence: The prominence value used for peak detection.
    - intensity_col: The name of the column containing intensity data.

    Returns:
    - A list of dictionaries, each representing data for a detected peak.
    """
    df_F = df[df.TRACK_ID == track_id]
    path = df_F.PATH.unique()

    # Find peaks
    peaks, properties = find_peaks(df_F[intensity_col], prominence=0.1, wlen=60, height= (0.2, None), width = (None, 50))
    prom = properties['prominences']
    # print(properties['peak_heights'])
    l= properties['left_bases']
    r = properties['right_bases']
    # Calculate peak widths at half prominence
    widths, width_heights, left_ips, right_ips = peak_widths(
        df_F[intensity_col].values, peaks, rel_height=0.5)

    # Shift peaks to correspond to actual FRAME values
    peaks_shifted = df_F['FRAME'].iloc[peaks].values
    l_s = df_F['FRAME'].iloc[l].values
    r_s = df_F['FRAME'].iloc[r].values
    peaks_time = df_F['FRAME'].iloc[peaks].transform(
        lambda x: x/framerate).values
    peak_values = df_F['NORM_MEAN_INTENSITY_CH1'].iloc[peaks].values
    peak_values_raw = df_F['MEDIAN_INTENSITY_CH1'].iloc[peaks].values
    diff_peak_values = df_F['DIFF_SMOOTH_NORM_MEAN_INTENSITY_CH1'].iloc[peaks].values
    first_peak_frame = df_F['FRAME'].iloc[peaks[0]] - \
        df_F['FRAME'].min() if peaks.size > 0 else None
    first_peak_time = first_peak_frame/framerate if peaks.size > 0 else None
    # Compile peak data into dictionaries
    peak_data = []
    for i in range(len(peaks)):
        p = df_F.loc[df_F.FRAME == int(peaks_shifted[i]), intensity_col].values[0]
        l= df_F.loc[df_F['FRAME'].between(int(l_s[i])-2, int(l_s[i])), intensity_col].median()

        r= df_F.loc[df_F['FRAME'].between(int(r_s[i]), int(r_s[i])+2), intensity_col].median()
        promi = max(p-l, p-r)
        diff = (r-l)/l
        
        # print(promi, 'VS', prom[i])
        if promi >= prominence:
            peak_data.append({
                'PATH': path[0],
                'TRACK_ID': track_id,
                'Frame': peaks_shifted[i],
                'L_frame': l_s[i], 
                'R_frame': r_s[i], 
                'Time': peaks_time[i],
                'DiffPeakValue': diff_peak_values[i],
                'PeakValue': peak_values[i],
                'PeakValueRaw': peak_values_raw[i],
                'PeakWidth': widths[i],
                'Prominence': promi,#prom[i],
                'Complete': abs(diff) < 1,
                'FirstPeakFrame': first_peak_frame,
                'FirstPeakTime': first_peak_time
            })

    return peak_data


def find_all_peaks(df, prominence_value, intensity_col='NORM_MEAN_INTENSITY_CH1', framerate=1):
    """
    Function to find peaks for each track ID in the DataFrame with a specified prominence.
    """
    peak_data_list = []  # List to accumulate all peak data

    # Initialize a 'PEAK' column in the main DataFrame to False
    df.loc[:, 'PEAK'] = False
    df.loc[:, 'PROMINENCE'] = np.nan

    # Group by 'DATASET' to process each group separately
    for dataset, group in df.groupby('DATASET', observed=True):
        # Iterate over each unique track ID within the current dataset group
        for track_id in group['TRACK_ID'].unique():
            # Extract data for this track_id
            track_data = group[group['TRACK_ID'] == track_id]

            # Assume find_trace_peaks returns a list of dictionaries with peak data for each track ID
            track_peaks = find_trace_peaks(
                track_data, track_id, prominence=prominence_value, intensity_col=intensity_col, framerate=framerate)

            # Check if peaks were found and update the 'PEAK' column
            if track_peaks:
                peak_indices = [peak['TRACK_ID'] for peak in track_peaks]
                # Set 'PEAK' to True for rows where peaks were found
                df.loc[(df['TRACK_ID'] == track_id) & (df['DATASET'] == dataset) & (
                    df['TRACK_ID'].isin(peak_indices)), 'PEAK'] = True
                

            # Append the dataset to each peak data dictionary and accumulate
            for peak in track_peaks:
                peak['DATASET'] = dataset
                peak_data_list.append(peak)

    # Convert the list of dictionaries to a DataFrame
    peaks_df = pd.DataFrame(peak_data_list)

    return peaks_df, df


def plot_traces(df, peaks_df, dataset=None, track_id=None, intensity_col='NORM_MEAN_INTENSITY_CH1', derivative_col='DIFF_SMOOTH_NORM_MEAN_INTENSITY_CH1', frame_col='TIME', ax=None):
    """
    Plot intensity and its derivative for a specific track, along with detected peaks.

    Parameters:
    - df: DataFrame containing the track data.
    - peaks_df: DataFrame containing the peaks data for all tracks.
    - track_id: The specific track ID to plot.
    - intensity_col: Column name for the intensity data.
    - derivative_col: Column name for the derivative of the intensity data.
    """
    # Filter the DataFrame for the current track_id and dataset
    df_F = df[(df['TRACK_ID'] == track_id) & (df['DATASET'] == dataset)]
    track_peaks_df = peaks_df[(peaks_df['TRACK_ID'] == track_id) & (
        peaks_df['DATASET'] == dataset)]

    # Determine plotting ranges
    plot_max = df[intensity_col].max()
    plot_min = df[derivative_col].min()
    frame_max = df['TIME'].max()

    # Create the plot
    ax.plot(df_F['TIME'], df_F[derivative_col], alpha=1,
            color='forestgreen', label='1st Derivative')
    ax.plot(df_F['TIME'], df_F[intensity_col],
            alpha=0.75, color='orange', label='Intensity')

    # Iterate over each peak in the track's peak DataFrame
    for _, peak_row in track_peaks_df.iterrows():
        peak_frame = peak_row['Time']
        # diff_peak_height = peak_row['DiffPeakValue']
        peak_height = peak_row['PeakValue']

        ax.plot(peak_frame, peak_height, 'x', color="red",
                label='Intensity Peak' if 'Intensity Peak' not in ax.get_legend_handles_labels()[1] else "")

    ax.set_title(
        f'Intensity Time Series for Track ID {track_id}, Dataset {dataset}')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Intensity / Derivative')
    ax.set_ylim([plot_min, plot_max])
    ax.set_xlim([0, frame_max])
    ax.legend()


def plot_last_positions(image_path, df, dimension, ax=None):
    """
    Plot the last positions of tracks from a DataFrame on top of an image.

    Parameters:
    - image_path: Path to the image file.
    - df: DataFrame containing the track data.
    - dimension: Dimension of the image in some unit of measurement.
    """
    # Read the image
    image_array = plt.imread(image_path)

    # Get the last positions of each track
    last_positions = df.groupby('TRACK_ID_ORIGINAL').last().reset_index()

    # Set up the plot
    ax.imshow(image_array, extent=[0, dimension, 0, dimension])

    # Create a scatter plot of the last positions
    ax.scatter(last_positions['POSITION_X'], dimension - last_positions['POSITION_Y'], s=last_positions['RADIUS'] * 50,
               marker='o', edgecolors='red', facecolors='none', alpha=0.9)

    # Annotate each point with the TRACK_ID
    for _, row in last_positions.iterrows():
        ax.text(row['POSITION_X'], dimension - row['POSITION_Y'], str(int(row['TRACK_ID_ORIGINAL'])),
                color='black', fontsize=9, ha='center', va='center')

    # Customize the plot
    ax.set_title('Last Cell Positions by Track ID')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    # Ensure the aspect ratio is equal to maintain the image scale
    ax.set_aspect('equal', 'box')


def plot_tracks_split_by_peaks(df, peaks_df, intensity_col='NORM_MEAN_INTENSITY_CH1', frame_col='FRAME_SYNC', axs=None):
    """
    Plots intensity tracks for peak detected tracks and remaining tracks, based on specified column.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the track data.
    - peaks_df (pd.DataFrame): DataFrame containing peak IDs.
    - column_name (str): Specifies which column to plot, 'NORM_MEAN_INTENSITY_CH1' or 'DIFF_SMOOTH_NORM_MEAN_INTENSITY_CH1'.
    - prominence (int): The prominence value used for peak detection, for display purposes.
    """

    # Create subplots
    # Plot for peak detected tracks
    df[df['PEAK']].sort_values(by='FRAME').groupby(['DATASET', 'TRACK_ID'], observed=True).plot(
        y=intensity_col, x=frame_col, ax=axs[0])

    dataset_names = ', '.join(df['DATASET'].unique())
    axs[0].set_title(
        f"Peak Detection with Prominence Value {df['PROMINENCE'].unique()[0]:.3f}, {dataset_names}")

    # Plot for remaining tracks
    df[~df['PEAK']].sort_values(by='FRAME').groupby(['DATASET', 'TRACK_ID'], observed=True).plot(
        y=intensity_col, x=frame_col, ax=axs[1])
    axs[1].set_title('Remaining Traces without Peaks')

    # Set labels and layout
    if intensity_col == 'DIFF_SMOOTH_NORM_MEAN_INTENSITY_CH1':
        ylim = max(abs(df[intensity_col].min()), df[intensity_col].max())
        set_ylim = [-ylim, ylim]
    else:
        set_ylim = [0, df[intensity_col].max()]

    for ax in axs:
        if ax.lines:
            ax.legend().set_visible(False)
        ax.set_ylim(set_ylim)
        ax.set_ylabel(intensity_col.replace('_', ' '))
        ax.set_xlabel('Time [s]')
    # plt.tight_layout()

    return axs


def tracks_split_by_threshold(df, frame_col='FRAME_SYNC', threshold_col='DIFF_SMOOTH_NORM_MEAN_INTENSITY_CH1', threshold=0):
    '''
    Identifies tracks that are "decaying" based on a specified threshold of mean intensity values
    after excluding the initial 20% of frames for each track. The function marks these tracks directly
    within the original DataFrame.
    '''

    df['DECAYING'] = False  # Initialize
    # Calculate the 20th percentile frame to start from for each track
    frame_cutoff = df.groupby(['DATASET', 'TRACK_ID'])[
        frame_col].transform(lambda x: x.quantile(0.2))

    # Filter the DataFrame to exclude the first 20% of frames for each track and exclude peaking traces
    filtered_df = df[(df[frame_col] > frame_cutoff) & (df['PEAK'] == False)]

    # Compute means on the filtered DataFrame
    track_means = filtered_df.groupby(['DATASET', 'TRACK_ID'])[
        threshold_col].mean()

    # Identify tracks below the threshold
    below_threshold_ids = track_means[track_means <= threshold].index

    # Update 'DECAYING' columns to df, correctly handling MultiIndex
    # Create a MultiIndex from df for comparison
    df_index = pd.MultiIndex.from_arrays(
        [df['DATASET'], df['TRACK_ID']], names=['DATASET', 'TRACK_ID'])
    df['DECAYING'] = df_index.isin(below_threshold_ids)
    df['THRESHOLD'] = threshold

    return df

def plot_tracks_split_by_threshold(df, intensity_col='NORM_MEAN_INTENSITY_CH1', frame_col='TIME_SYNC', threshold=0, axs=None):
    '''
    Plots tracks from a DataFrame, separating them into two categories based on the 'DECAYING' flag.
    Tracks not marked as 'DECAYING' are plotted on the first subplot, and tracks marked as 'DECAYING'
    are plotted on the second subplot. The function assumes the DataFrame has a 'PEAK' column and
    uses it to filter out peak tracks before plotting.
    '''

    df_filtered = df[df['PEAK'] == False]

    # Plotting tracks not marked as 'DECAYING'
    for track_id, track_data in df_filtered[~df_filtered['DECAYING']].groupby(['DATASET', 'TRACK_ID']):
        track_data.plot(y=intensity_col, x=frame_col, ax=axs[0], legend=False)

    axs[0].set_title(
        f"Tracks with Slope Threshold > {df_filtered.THRESHOLD.iloc[0]:.5f}, {df['DATASET'].unique()}")
    axs[0].set_ylabel('Normalized Mean Intensity')

    # Plotting tracks marked as 'DECAYING'
    for track_id, track_data in df_filtered[df_filtered['DECAYING']].groupby(['DATASET', 'TRACK_ID']):
        track_data.plot(y=intensity_col, x=frame_col, ax=axs[1], legend=False)

    dataset_names = ', '.join(df['DATASET'].unique())
    axs[1].set_title(
        f'Tracks with Slope Threshold < {df_filtered.THRESHOLD.iloc[0]:.5f}, {dataset_names}')
    axs[1].set_ylabel('Normalized Mean Intensity')

    # Setting the y-limit for both axes based on the maximum of the filtered data
    max_intensity = df_filtered[intensity_col].max()
    for ax in axs:
        ax.set_ylim([0, max_intensity])
        ax.set_ylabel(intensity_col.replace('_', ' '))
        ax.set_ylim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Time [s]')
    plt.tight_layout()

    return axs

def tracks_split_by_regression(df,frame_col='FRAME_SYNC',ycol='SMOOTH_NORM_MEAN_INTENSITY_CH1', peak_col='PEAK', rel_drop_thresh=0.2, 
                               r2_thresh=0.5,slope_thresh=-5e-4, min_points=8):
    """
    Mark tracks as DECAYING using:
      - robust start/end drop on first/last quarter (80th/20th pct)
      - AND/OR a linear regression rule: (R^2 > r2_thresh) & (slope < slope_thresh)
    Data handling:
      - Excludes first 20% of frames per track
      - Excludes rows with PEAK == True
    """
    df = df.copy()
    df['DECAYING'] = False

    decisions = {}

    for (dataset, track_id), g in df.groupby(['DATASET', 'TRACK_ID'], sort=False):
        # exclude first 20% of frames within this track
        g2 = g[(g[frame_col] > 0)]
        # exclude peaking rows
        if peak_col in g2.columns:
            g2 = g2[g2[peak_col] == False]

        # pull arrays and clean NaNs
        x = g2[frame_col].astype(float).to_numpy()
        y = g2[ycol].astype(float).to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]

        # need enough points
        if y.size < min_points:
            decisions[(dataset, track_id)] = False
            continue

        # regression
        res = linregress(x, y)
        R2 = res.rvalue**2

        # robust start/end stats over first/last quarter
        k = max(3, y.size // 4)
        start = np.percentile(y[:k], 80)
        end   = np.percentile(y[-k:], 20)

        # your absolute drop rule (not normalized)
        rel_drop = start - end

        decaying = (rel_drop > rel_drop_thresh) or ((R2 > r2_thresh) and (res.slope < slope_thresh))
        decisions[(dataset, track_id)] = bool(decaying)

    # map decisions back to all rows
    idx = pd.MultiIndex.from_arrays([df['DATASET'], df['TRACK_ID']], names=['DATASET','TRACK_ID'])
    df['DECAYING'] = idx.map(decisions).fillna(False).astype(bool)

    return df


def plot_cell_states_pie_chart(df, ax):
    '''
    Plots a pie chart representing the distribution of cell states (Peaking, Decaying, Constant) within a dataset.
    '''
    # Calculate unique counts of cells in different states
    cells_peaking = df[df['PEAK']].TRACK_ID.unique().shape[0]
    cells_decay = df[df['DECAYING']].TRACK_ID.unique().shape[0]
    cells_rest = df['TRACK_ID'].unique().shape[0] - cells_decay - cells_peaking

    # Define plot parameters
    labels = 'Peaking', 'Constant Signal', 'Decaying Signal'
    sizes = [cells_peaking, cells_rest, cells_decay]
    colors = ['yellowgreen', 'wheat', 'grey']
    explode = (0.1, 0.1, 0.1)

    # Plotting
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=140)
    ax.set_aspect('equal')
    dataset_names = ', '.join(df['DATASET'].unique())
    ax.set_title(
        f"Fraction of Peaking Cells \n Total cell number: {df['TRACK_ID'].unique().shape[0]}, {dataset_names}")


def plot_peaks_histogram(peaks_df, ax=None):
    '''
    Plots a histogram of the number of peaks per track in a dataset.
    '''
    # Calculate the number of peaks per track
    num_peaks_per_track = peaks_df['TRACK_ID'].value_counts()

    # Create bins such that each bar is centered on the integer
    min_peak_count = num_peaks_per_track.min() - 0.5
    max_peak_count = num_peaks_per_track.max() + 0.5
    bins = np.arange(min_peak_count, max_peak_count + 1, 1)

    # Plot the histogram
    ax.hist(num_peaks_per_track, bins=bins, alpha=1,
            color='yellowgreen', edgecolor='black')
    ax.set_title(
        f"Histogram of Number of Peaks per Track, {peaks_df['DATASET'].unique()}")
    ax.set_xlabel('Number of Peaks')
    ax.set_ylabel('Frequency of Tracks')


def plot_first_peak_histogram(peaks_df, bins_param='fd', ax=None):
    '''
    Plots a histogram of the duration until the first peak across all tracks in different datasets.
    '''
    # Group by 'DATASET' and 'TRACK_ID', and concatenate all FirstPeakTime values into a single array
    durations = np.concatenate(peaks_df.groupby(
        ['DATASET', 'TRACK_ID']).FirstPeakTime.agg(list).tolist())
     
    # Plot the histogram
    bins, _, _  = ax.hist(durations, bins=bins_param, alpha=1, color="orange", edgecolor='black')

    # Setting up the title using unique values from the 'DATASET' column
    dataset_names = ', '.join(peaks_df['DATASET'].unique())
    ax.set_title(f'Histogram of Duration until First Peak, {dataset_names}')

    # Setting x and y labels
    ax.set_xlabel('Duration [s]')
    ax.set_ylabel('Frequency')
    
    # Adding a text box inside the plot showing the median duration
    median_duration = np.median(durations)
    ax.text(0.4, 0.9, f'Median Duration: {median_duration} s',
            fontsize=12, color='black', transform=ax.transAxes)
    
    binsize = len(bins)
    return binsize

def plot_cell_status_bars(df, colors=('yellowgreen', 'wheat', 'gray'), ax=None, percent=False):
    ''' 
    Plots a bar chart of cell counts in different states (Peaking, Resting, Decaying) across multiple datasets.
    '''
    # Extract unique datasets
    datasets = df['DATASET'].unique()
    data_for_plotting = []

    # Calculate counts of peaking, decaying, and resting cells per dataset
    for dataset in datasets:
        subset = df[df['DATASET'] == dataset]
        cells_peaking = subset[subset['PEAK']].TRACK_ID.nunique()
        cells_decay = subset[subset['DECAYING']].TRACK_ID.nunique()
        cells_rest = subset['TRACK_ID'].nunique() - cells_decay - cells_peaking
        data_for_plotting.append((cells_peaking, cells_rest, cells_decay))

    # Unpack the plotting data
    peaking_counts, rest_counts, decaying_counts = zip(*data_for_plotting)
    summary_df = pd.DataFrame(
    data_for_plotting,
    index=datasets,
    columns=['Peaking', 'Resting', 'Decaying']
    )
    
    # Plotting the bar chart
    positions = range(len(datasets))
    width = 0.4

    # Create bars for peaking, decaying, and resting counts
    ax.bar(positions, peaking_counts, bottom=np.array(rest_counts) +
           np.array(decaying_counts), label='Peaking', color=colors[0], width=width)
    ax.bar(positions, rest_counts, label='Resting', bottom=np.array(decaying_counts),
           color=colors[1], width=width)
    ax.bar(positions, decaying_counts,
           label='Decaying', color=colors[2], width=width)

    # Set labels, title, and ticks
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Number of Cells')
    ax.set_xticks(positions)
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_xticklabels(datasets)
    ax.legend(title='Cell Status')
    return summary_df
def plot_cell_status_bars_percent(df, colors=('yellowgreen', 'wheat', 'gray'), ax=None):
    ''' 
    Plots a bar chart of cell counts in different states (Peaking, Resting, Decaying) across multiple datasets.
    '''
    # Extract unique datasets
    datasets = df['DATASET'].unique()
    data_for_plotting = []

    # Calculate counts of peaking, decaying, and resting cells per dataset
    for dataset in datasets:
        subset = df[df['DATASET'] == dataset]
        cells_peaking = subset[subset['PEAK']].TRACK_ID.nunique()
        cells_decay = subset[subset['DECAYING']].TRACK_ID.nunique()
        cells_rest = subset['TRACK_ID'].nunique() - cells_decay - cells_peaking
        data_for_plotting.append((cells_peaking, cells_rest, cells_decay))

    # Unpack the plotting data
    peaking_counts, rest_counts, decaying_counts = zip(*data_for_plotting)

    # --- convert to percentages ---
    import numpy as np
    peaking_counts = np.array(peaking_counts, dtype=float)
    rest_counts = np.array(rest_counts, dtype=float)
    decaying_counts = np.array(decaying_counts, dtype=float)

    totals = peaking_counts + rest_counts + decaying_counts
    totals[totals == 0] = 1  # avoid division by zero
    peaking_counts = 100 * peaking_counts / totals
    rest_counts = 100 * rest_counts / totals
    decaying_counts = 100 * decaying_counts / totals
    # ------------------------------

    # Plotting the bar chart
    positions = range(len(datasets))
    width = 0.4

    # Create bars for peaking, decaying, and resting counts
    ax.bar(positions, peaking_counts, bottom=np.array(rest_counts) +
           np.array(decaying_counts), label='Peaking', color=colors[0], width=width)
    ax.bar(positions, rest_counts, label='Resting', bottom=np.array(decaying_counts),
           color=colors[1], width=width)
    ax.bar(positions, decaying_counts,
           label='Decaying', color=colors[2], width=width)

    # Set labels, title, and ticks
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Percentage of Cells (%)')
    ax.set_xticks(positions)
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_xticklabels(datasets)
    ax.legend(title='Cell Status')

def plot_peaking_percentage_boxplot(df, ax=None):
    """
    Boxplot of % cells peaking per DATASET, with each dot = one replicate.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # --- build per-replicate (% peaking) table ---
    # total tracks per (DATASET, PATH)
    totals = (df.groupby(['DATASET', 'PATH'])['TRACK_ID']
                .nunique()
                .rename('n_tracks_total'))

    # peaking tracks per (DATASET, PATH)
    peakers = (df[df['PEAK']].groupby(['DATASET', 'PATH'])['TRACK_ID']
                .nunique()
                .rename('n_tracks_peaking'))

    summary = (totals.to_frame()
               .join(peakers, how='left')
               .fillna({'n_tracks_peaking': 0}))

    # percentage per replicate
    summary['pct_peaking'] = (summary['n_tracks_peaking'] /
                              summary['n_tracks_total'].replace(0, np.nan)) * 100
    summary = summary.dropna(subset=['pct_peaking']).reset_index()

    # --- plot ---
    datasets = summary['DATASET'].unique().tolist()
    data_by_ds = [summary.loc[summary['DATASET'] == ds, 'pct_peaking'].values
                  for ds in datasets]

    # boxplot
    bp = ax.boxplot(data_by_ds, labels=datasets, showfliers=False)
    ax.set_ylim(0, 100)

    # overlay dots (one per PATH) with small jitter
    for i, ys in enumerate(data_by_ds, start=1):
        if len(ys) == 0:
            continue
        xs = np.random.normal(loc=i, scale=0.05, size=len(ys))
        ax.scatter(xs, ys, alpha=0.8, edgecolor='k', linewidths=0.5)

    ax.set_ylabel('Cells peaking (%)')
    ax.set_xlabel('Dataset')
    ax.set_title('Percentage of cells peaking per replicate')
    ax.tick_params(axis='x', labelrotation=90)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    return fig, summary  # summary has one row per (DATASET, PATH)

def extract_date_str_from_path(path, idx=7):
    """
    Returns the folder at position `idx` (0-based) from PATH.
    Example:
      D:\\...\\SNAP\\20250718\\CART3Hi\\R1\\tracks_filtered.csv
                           ^ returns '20250718'
    """
    parts = str(path).split("\\")  # works fine for Windows-style paths
    return parts[idx] if len(parts) > idx else "unknown"


def plot_peaking_percentage_boxplot_hue_date(df, ax=None):
    """
    Boxplot of % cells peaking per DATASET (like your original),
    with replicate dots hued by date string extracted from PATH (index 7).
    Returns (ax, summary) where summary has one row per (DATASET, PATH).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))

    # --- per-replicate (% peaking) table ---
    totals = (df.groupby(['DATASET', 'PATH'])['TRACK_ID']
                .nunique()
                .rename('n_tracks_total'))

    peakers = (df[df['PEAK']].groupby(['DATASET', 'PATH'])['TRACK_ID']
                .nunique()
                .rename('n_tracks_peaking'))

    summary = (totals.to_frame()
               .join(peakers, how='left')
               .fillna({'n_tracks_peaking': 0})
               .reset_index())

    summary['pct_peaking'] = (summary['n_tracks_peaking'] /
                              summary['n_tracks_total'].replace(0, np.nan)) * 100
    summary = summary.dropna(subset=['pct_peaking']).copy()

    # --- add DATE_STR from PATH (folder index 7) ---
    summary['DATE_STR'] = summary['PATH'].map(lambda p: extract_date_str_from_path(p, idx=7))

    # --- plot (boxplots per DATASET, dots hued by DATE_STR) ---
    datasets = summary['DATASET'].unique().tolist()
    data_by_ds = [summary.loc[summary['DATASET'] == ds, 'pct_peaking'].values
                  for ds in datasets]

    # boxplot (overall per DATASET)
    ax.boxplot(data_by_ds, labels=datasets, showfliers=False)
    ax.set_ylim(0, 100)

    # colors per date string
    dates = sorted(summary['DATE_STR'].unique())
    cmap = plt.get_cmap('tab20')  # more distinct colors if many dates
    color_by_date = {d: cmap(i % cmap.N) for i, d in enumerate(dates)}

    # overlay dots (each replicate PATH), colored by DATE_STR
    legend_handles = {}
    for i, ds in enumerate(datasets, start=1):
        sub = summary[summary['DATASET'] == ds]
        ys = sub['pct_peaking'].to_numpy()
        xs = np.random.normal(loc=i, scale=0.05, size=len(ys))
        colors = sub['DATE_STR'].map(color_by_date).to_numpy()
        sc = ax.scatter(xs, ys, c=colors, alpha=0.9, edgecolor='k', linewidths=0.5, s=28)
        # collect one handle per date for legend
        for d in sub['DATE_STR'].unique():
            if d not in legend_handles:
                # dummy handle with correct color
                legend_handles[d] = ax.scatter([], [], c=[color_by_date[d]], edgecolor='k', linewidths=0.5, s=28)

    ax.set_ylabel('Cells peaking (%)')
    ax.set_xlabel('Dataset')
    ax.set_title('Percentage of cells peaking per replicate')
    ax.tick_params(axis='x', labelrotation=90)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    # legend by date string
    ax.legend(legend_handles.values(), legend_handles.keys(),
              title='PATH date (folder 7)', frameon=False,
              ncol=min(4, max(1, int(len(legend_handles)**0.5))))

    return fig, summary




def plot_peaking_percentage_boxplot_from_summary(summary, ax=None, color_scheme=None):
    """
    Boxplot of % cells peaking per DATASET, with each dot = one replicate (PATH),
    using a precomputed summary DataFrame.

    Expects columns: ['DATASET', 'PATH', 'pct_peaking'].
    Returns (fig, summary).
    """
    # Validate required columns
    required_cols = {'DATASET', 'PATH', 'pct_peaking'}
    missing = required_cols - set(summary.columns)
    if missing:
        raise ValueError(f"summary must contain columns: {required_cols}. Missing: {missing}")

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    # Prepare data
    datasets = summary['DATASET'].unique().tolist()
    data_by_ds = [summary.loc[summary['DATASET'] == ds, 'pct_peaking'].values
                  for ds in datasets]

     # Overlay jittered points with color scheme
    for i, (ds, ys) in enumerate(zip(datasets, data_by_ds), start=1):
         if len(ys) == 0:
             continue
         xs = np.random.normal(loc=i, scale=0.05, size=len(ys))
         dot_color = color_scheme.get(ds, '#000000') if color_scheme else '#000000'
         ax.scatter(xs, ys, alpha=0.8, edgecolor='k', linewidths=0.5, facecolor=dot_color, s = 50)
    # Boxplot with patch_artist=True for coloring
    bp = ax.boxplot(data_by_ds, labels=datasets, showfliers=False, patch_artist=True)
    ax.set_ylim(0, 100)
    
    
    # Make all boxes white
    for patch in bp['boxes']:
        patch.set_facecolor('none')

   

    # Labels and grid
    ax.set_ylabel('Cells peaking (%)')
    ax.set_xlabel('Dataset')
    ax.set_title('Percentage of cells peaking per replicate (PATH)')
    ax.tick_params(axis='x', labelrotation=90)
    ax.grid(axis='y', linestyle=':', alpha=0.5)


def plot_peaking_percentage_boxplot_from_summary2(summary, ax=None, color_scheme=None, dot_size=60):
    """
    Boxplot of % cells peaking per DATASET, with each dot = one replicate (PATH),
    using a precomputed summary DataFrame.

    Boxes: colored according to color_scheme
    Dots: black
    Adds: pairwise Mann–Whitney U tests between all datasets with brackets + numeric p-values
    """
    required_cols = {'DATASET', 'PATH', 'pct_peaking'}
    missing = required_cols - set(summary.columns)
    if missing:
        raise ValueError(f"summary must contain columns: {required_cols}. Missing: {missing}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    # Prepare data
    datasets = summary['DATASET'].unique().tolist()
    data_by_ds = [summary.loc[summary['DATASET'] == ds, 'pct_peaking'].values for ds in datasets]

    # Boxplot
    bp = ax.boxplot(data_by_ds, labels=datasets, showfliers=False, patch_artist=True)

    # Color boxes
    if color_scheme:
        for patch, ds in zip(bp['boxes'], datasets):
            patch.set_facecolor(color_scheme.get(ds, '#FFFFFF'))
            patch.set_edgecolor('black')

    # Overlay jittered dots
    for i, ys in enumerate(data_by_ds, start=1):
        xs = np.random.normal(loc=i, scale=0.05, size=len(ys))
        ax.scatter(xs, ys,
                   alpha=0.9,
                   edgecolor='k',
                   linewidths=0.5,
                   facecolor='black',
                   s=dot_size,
                   zorder=3)

    # ---------------------------
    # Pairwise Mann–Whitney tests
    # ---------------------------
    # Base height just above the max data point
    y_max = max(max(d) for d in data_by_ds if len(d) > 0)
    height_step = 4.0     # vertical spacing between brackets
    bracket_h = 1.0       # bracket "cap" height

    # To reduce crossings/overlap, draw shorter-distance pairs first
    pairs = list(combinations(range(len(datasets)), 2))
    pairs.sort(key=lambda ij: abs(ij[1] - ij[0]))

    # Track how many brackets already span each x-position -> stack neatly
    span_counts = [0] * (len(datasets) + 1)  # 1-based plotting positions

    max_bracket_y = y_max

    for i, j in pairs:
        data_i = data_by_ds[i]
        data_j = data_by_ds[j]

        # Skip if either group has no data
        if len(data_i) == 0 or len(data_j) == 0:
            continue

        stat, p = mannwhitneyu(data_i, data_j, alternative='two-sided')

        x1, x2 = i + 1, j + 1

        # Determine stack level based on spans already used between x1..x2
        level = max(span_counts[x1:x2+1])
        y = y_max + 3 + level * height_step

        # Update span usage
        for k in range(x1, x2 + 1):
            span_counts[k] = max(span_counts[k], level + 1)

        # Draw bracket
        ax.plot([x1, x1, x2, x2],
                [y, y + bracket_h, y + bracket_h, y],
                lw=1, c='k')

        # Numeric p-value label
        ax.text((x1 + x2) / 2,
                y + bracket_h + 0.6,
                f"p = {p:.3g}",
                ha='center',
                va='bottom')

        max_bracket_y = max(max_bracket_y, y + bracket_h + 2.0)

    # Make sure brackets fit
    ax.set_ylim(0, max(100, max_bracket_y + 2))

    # Labels and grid
    ax.set_ylabel('Cells peaking (%)')
    ax.set_xlabel('Dataset')
    ax.set_title('Percentage of cells peaking per replicate (PATH)')
    ax.tick_params(axis='x', labelrotation=90)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    return fig, ax

