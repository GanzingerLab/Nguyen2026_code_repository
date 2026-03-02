#%%
import nd2
import os
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from PIL import Image, ImageDraw
from skimage.measure import find_contours
from skimage.morphology import (
    binary_closing,
    remove_small_holes,
    remove_small_objects, 
    binary_erosion, 
)
import pandas as pd
from skimage.filters import gaussian, threshold_triangle
from skimage.measure import regionprops
import trackpy as tp
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.measure import regionprops_table
import seaborn as sns
from scipy.stats import mannwhitneyu
#%%
class NFAT: 
    """
    A class for analyzing NFAT translocation in ND2 image files.
    
    This class handles loading ND2 images, segmenting nuclei and cytoplasm,
    tracking cells, and computing N/C ratios for NFAT translocation detection.
    """
    def __init__(self, file_path):
        self.folder = os.path.split(file_path)[0]    
        self.open_image(file_path)
    def open_image(self, file_path):
        """
        Load and extract channels from the ND2 file. Used opon initiinitialization of the the object. 
        
        Parameters:
        file_path (str): Path to the ND2 file.
        
        Sets:
        self.nuc: Nucleus channel (channel 0).
        self.fluo: Fluorescence channel (channel 1).
        """    
        nd2_file = nd2.ND2File(file_path)
        # Access image data
        image_data = nd2_file.asarray()
        #2. Extract channels chanel
        ch1 = image_data[:, 0, :, :].astype(float) 
        fluo = image_data[:, 1, :, :].astype(float)      
        self.nuc = ch1
        self.fluo = fluo
    def process_nuclei_image(self):
        """
        Process the nucleus channel for illumination correction and normalization.
        
        Sets:
        self.nuc_corr: Illumination-corrected nucleus channel.
        self.nuc_corr_norm: Normalized nucleus channel (0-1 range).
        """
        #make average projection
        average = np.mean(self.nuc, axis = 0)
        #calculate background
        bg = gaussian(average, sigma=100, preserve_range=True)
        # Illumination correction 
        self.nuc_corr = self.nuc / (bg + 1e-6)
        # Normalize to 0.1 for stable thresholds
        mn = self.nuc_corr.min(axis=(1, 2), keepdims=True)
        mx = self.nuc_corr.max(axis=(1, 2), keepdims=True)
        self.nuc_corr_norm = (self.nuc_corr - mn) / (mx - mn + 1e-8)
    def _segment_nuclei_frame(self, nuc0, 
        smooth_sigma=1,
        min_size=200,
        min_distance=15):
        """
        Segment nuclei in a single frame. Using a distance-based watershed on a 
        triangle-thresholded nuclei image. 
        
        Parameters:
        nuc0 (np.ndarray): Normalized nucleus image for the frame.
        smooth_sigma (float): Sigma for Gaussian smoothing.
        min_size (int): Minimum size for nuclei objects.
        min_distance (int): Minimum distance between nuclei peaks.
        
        Returns:
        np.ndarray: Labeled nuclei mask.
        """
        #gaussian blur to smooth
        nuc_s = gaussian(nuc0, sigma=smooth_sigma)
        #threshold (triangle)
        tn = threshold_triangle(nuc_s)
        nuc_mask = nuc_s > tn
        #process mask
        nuc_mask = binary_erosion(nuc_mask)
        nuc_mask = remove_small_objects(nuc_mask, min_size)
        nuc_mask = binary_fill_holes(nuc_mask)
        # Split fused nuclei using distance-based watershed on nuclei mask
        dist = distance_transform_edt(nuc_mask)
        peaks = peak_local_max(
            dist,
            labels=nuc_mask,
            min_distance=min_distance,   
            threshold_abs=0
        )
        seed = np.zeros_like(dist, dtype=np.int32)
        seed[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
        
        markers = watershed(-dist, seed, mask=nuc_mask)  # one label per nucleus
        return markers
    def segment_nuclei_stack(self, smooth_sigma=1,
        min_size=200,
        min_distance=15):
        """
        Segment nuclei across all frames.
        
        Parameters:
        smooth_sigma (float): Sigma for Gaussian smoothing.
        min_size (int): Minimum size for nuclei objects.
        min_distance (int): Minimum distance between nuclei peaks.
        
        Sets:
        self.markers: Labeled nuclei stack.
        """
        T = self.nuc.shape[0]
        out = np.zeros_like(self.nuc, dtype=np.int32)
        for i in range(T):
            out[i] = self._segment_nuclei_frame(self.nuc_corr_norm[i], smooth_sigma, min_size, min_distance)
        self.markers = out
    def _segment_cyto_frame(self, cyto_s, markers, R = 10, min_size = 200, hole_size = 100):
        """
        Segment cytoplasm in a single frame based on nuclei markers using a traingle-thresholded 
        cytoplasmatic mask.  
        
        Parameters:
        cyto_s (np.ndarray): Cytoplasm image for the frame.
        markers (np.ndarray): Nuclei markers for the frame.
        R (int): Maximum distance from nuclei for cytoplasm assignment.
        min_size (int): Minimum size for cytoplasm objects.
        hole_size (int): Maximum hole size to fill.
        
        Returns:
        np.ndarray: Labeled cytoplasm mask.
        """
        #base segmentation
        mask = cyto_s > threshold_triangle(cyto_s)
        mask = remove_small_objects(mask, min_size)
        mask = remove_small_holes(mask, hole_size)
        mask = (mask+markers > 0).astype(int)
        mask = binary_closing(mask)
        #Capped grwoth from nuclei
        # Compute distance to nearest nucleus marker + nearest-marker indices
        dist_to_marker, inds = distance_transform_edt(markers == 0, return_indices=True)
        # Nearest nucleus label for each pixel
        nearest_label = markers[inds[0], inds[1]]
        # Only allow assignment inside cytoplasm mask AND within radius R of a nucleus
        allowed = mask & (dist_to_marker <= R)
        labels = np.where(allowed, nearest_label, 0).astype(np.int32)
        return labels
    def segment_cyto_stack(self, R = 10, min_size = 200, hole_size = 100):
        """
        Segment cytoplasm across all frames.
        
        Parameters:
        R (int): Maximum distance from nuclei for cytoplasm assignment.
        min_size (int): Minimum size for cytoplasm objects.
        hole_size (int): Maximum hole size to fill.
        
        Sets:
        self.cyto_labels: Labeled cytoplasm stack.
        """
        T = self.fluo.shape[0]
        out = np.zeros_like(self.fluo, dtype=np.int32)
        for i in range(T):
            out[i] = self._segment_cyto_frame(self.fluo[i], self.markers[i], R, min_size, hole_size)
        self.cyto_labels = out
    def segment_channels(self,  
        smooth_sigma=1,
        min_size=200,
        min_distance=15, 
        R = 10, hole_size = 100):
        """
        Segment both nuclei and cytoplasm channels.
        
        Parameters:
        smooth_sigma (float): Sigma for Gaussian smoothing (nuclei).
        min_size (int): Minimum size for objects.
        min_distance (int): Minimum distance between nuclei peaks.
        R (int): Maximum distance from nuclei for cytoplasm.
        hole_size (int): Maximum hole size to fill in cytoplasm.
        """
        self.segment_nuclei_stack(smooth_sigma, min_size, min_distance)
        self.segment_cyto_stack(R, min_size, hole_size)
    def remove_nuclei_from_cells(self):
        """
        Remove nuclei regions from cytoplasm labels to get cytoplasm-only mask.
        
        Sets:
        self.cyto_only: Cytoplasm labels with nuclei removed.
        """
        cyto_only = self.cyto_labels.copy()
        cyto_only[self.markers > 0] = 0
        self.cyto_only = cyto_only
    def assign_nuc_to_cell(self):
        """
        Assign each nucleus to its corresponding cell.
           
        Returns:
        pd.DataFrame: DataFrame with frame, nuc_label, cell_label.
        """
        rows = []
        T = self.markers.shape[0]

        for t in range(T):
            for nid in np.unique(self.markers[t]):
                if nid == 0:
                    continue

                cell_ids = self.cyto_labels[t][self.markers[t] == nid]
                cell_ids = cell_ids[cell_ids > 0]

                cell_id = int(np.bincount(cell_ids).argmax()) if cell_ids.size else np.nan

                rows.append({
                    "frame": t,
                    "nuc_label": int(nid),
                    "cell_label": int(cell_id) if cell_id == cell_id else np.nan

                })

        return pd.DataFrame(rows)
    def labels_to_trackpy_features(self):
        """
        Convert labeled objects to trackpy-compatible features.
        
        Returns:
        pd.DataFrame: DataFrame with frame, x, y, label, area.
        """
        rows = []
        T = self.markers.shape[0]

        for t in range(T):
            lab = self.markers[t]

            for rp in regionprops(lab):
                y, x = rp.centroid  # row, col

                row = {
                    "frame": t,
                    "x": float(x),
                    "y": float(y),
                    "label": int(rp.label),     # label id within THIS frame
                    "area": float(rp.area),
                }

                rows.append(row)

        return pd.DataFrame(rows)
    def save_cell_outline_video(self,
        cyto,
        cyto_labels,
        markers,
        linked_df,
        out_path,
        dot_radius=3,
        global_contrast=True
    ):
        """
        Save a video of cell outlines and tracks as a TIFF file.
        
        Parameters:
        cyto (np.ndarray): Cytoplasm image stack.
        cyto_labels (np.ndarray): Cytoplasm labels.
        markers (np.ndarray): Nuclei markers.
        linked_df (pd.DataFrame): Linked tracking data.
        out_path (str): Output file path.
        dot_radius (int): Radius for centroid dots.
        global_contrast (bool): Use global contrast scaling.
        
        Returns:
        str: Output file path.
        """

        T, H, W = cyto.shape

        # contrast scaling
        if global_contrast:
            vmin = np.percentile(cyto, 1)
            vmax = np.percentile(cyto, 99)

        # group detections by frame
        by_frame = {t: df for t, df in linked_df.groupby("frame")}

        frames_rgb = []

        for t in range(T):
            img = cyto[t].astype(np.float32)
            if global_contrast:
                img = np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)
            else:
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            base = (img * 255).astype(np.uint8)
            rgb = np.stack([base, base, base], axis=-1)

            im = Image.fromarray(rgb)
            draw = ImageDraw.Draw(im)

            if t in by_frame:
                for _, r in by_frame[t].iterrows():
                    lab = int(r.nuc_label)
                    pid = int(r.particle)
                    x = int(round(r.x))
                    y = int(round(r.y))

                    # --- nucleus contour (green) ---
                    nuc_mask = (markers[t] == lab)
                    for c in find_contours(nuc_mask.astype(float), 0.5):
                        draw.line([tuple(p[::-1]) for p in c], fill=(0, 255, 0), width=1)

                    # --- cytoplasm contour (red) ---
                    cell_mask = (cyto_labels[t] == lab)
                    for c in find_contours(cell_mask.astype(float), 0.5):
                        draw.line([tuple(p[::-1]) for p in c], fill=(255, 0, 0), width=1)

                    # --- centroid ---
                    draw.ellipse(
                        (x-dot_radius, y-dot_radius, x+dot_radius, y+dot_radius),
                        fill=(255, 255, 0)
                    )

                    # --- particle ID ---
                    draw.text((x+4, y-4), str(pid), fill=(255, 255, 0))

            frames_rgb.append(np.array(im))

        frames_rgb = np.stack(frames_rgb, axis=0)
        tiff.imwrite(out_path, frames_rgb, photometric="rgb")

        return out_path
    def precompute_mean_table(self, label_stack, image_stack, value_name,
                            label_col="nuc_label", frame_col="frame"):
        """
        Compute mean intensities for labeled regions.
        
        Parameters:
        label_stack (np.ndarray): Label stack.
        image_stack (np.ndarray): Intensity image stack.
        value_name (str): Name for the mean intensity column.
        label_col (str): Name for the label column.
        frame_col (str): Name for the frame column.
        
        Returns:
        pd.DataFrame: DataFrame with frame, label, mean intensity.
        """
        rows = []
        T = label_stack.shape[0]

        for t in range(T):
            props = regionprops_table(
                label_stack[t],
                intensity_image=image_stack[t],
                properties=("label", "mean_intensity")
            )
            df = pd.DataFrame(props)
            df[frame_col] = t
            df = df.rename(columns={"label": label_col, "mean_intensity": value_name})
            rows.append(df)

        return pd.concat(rows, ignore_index=True)
    def compute_all_means(self, measurements):
        """
        Compute mean intensities for multiple measurements.
        
        Parameters:
        measurements (list): List of measurement dictionaries.
        
        Returns:
        list: List of DataFrames with mean intensities.
        """
        tables = []
        for m in measurements:
            df = self.precompute_mean_table(
                label_stack=m["mask"],
                image_stack=m["image"],
                value_name=m["name"]
            )
            tables.append(df)
        return tables
    def _first_n_consecutive_ones(self, arr, n=4):
        """
        Return the index of the first occurrence of n consecutive ones
        in a 1D binary array.

        Parameters
        ----------
        arr : 1D array-like
            Binary array (values 0 or 1).
        n : int
            Minimum number of consecutive ones required.

        Returns
        -------
        int or None
            Index of the first frame where the streak starts,
            or None if no such streak exists.
        """
        arr = np.asarray(arr)
        kernel = np.ones(n)
        conv = np.convolve(arr, kernel, mode='valid')

        hits = np.where(conv == n)[0]

        if len(hits) > 0:
            return hits[0]  # first starting index
        else:
            return None
    def analyze_NFAT(self, thr = 1.05, n=4, save_outlines=False): 
        """
        Perform full NFAT translocation analysis, including segmentation,
        tracking, intensity extraction, and N/C ratio classification.
        
        Parameters:
        thr (float): Threshold for nuclear-to-cytoplasmic ratio (N/C) used to
                    classify NFAT translocation (default = 1.05).
        save_outlines (bool): Whether to save an outline and tracking overlay video.
        
        Returns:
        dict: Dictionary mapping particle IDs to translocation status 
            (1 = translocated, 0 = not translocated).
        pd.DataFrame: Per-frame measurements for each tracked cell, including
                    nuclear and cytoplasmic mean intensities and N/C ratios.
        """
        # Preprocessing and segmentation
        self.process_nuclei_image()
        self.segment_channels()
        self.remove_nuclei_from_cells()
        nuc_to_cell = self.assign_nuc_to_cell()
        # Convert labeled nuclei into trackpy-compatible features
        df_to_track = self.labels_to_trackpy_features()
        df_to_track = df_to_track.merge(
            nuc_to_cell,
            left_on=["frame", "label"],
            right_on=["frame", "nuc_label"],
            how="left"
        ).drop(columns=["label"])
        # Track nuclei across frames 
        linked = tp.link_df(df_to_track, search_range=25, memory=2)
        # Filter out short tracks (require at least 1/3 of movie length)
        track_len = linked.groupby("particle")["frame"].transform("nunique")
        linked_filt = linked[track_len >= self.nuc_corr.shape[0]/3].copy()
        # Extract mean fluorescence measurements
        measurements = [
            # nuclear region
            {"mask": self.markers,            "image": self.fluo,     "name": "mean_fluo_in_nuc"},
            # cytoplasm (excluding nucleus)
            {"mask": self.cyto_labels,        "image": self.fluo,     "name": "mean_fluo_cell"},
            # cytoplasm (excluding nucleus)
            {"mask": self.cyto_only, "image": self.fluo,     "name": "mean_fluo_cyto"},
        ]
        tables = self.compute_all_means(measurements)
        
        # Merge per-frame intensity measurements into tracking dataframe
        for df in tables:
            linked_filt = linked_filt.merge(df, on=["frame", "nuc_label"], how="left")  
        # Remove incomplete entries
        self.linked_filt2 = linked_filt.dropna().copy()
        # Compute nuclear-to-cell and nuclear-to-cytoplasm ratios
        self.linked_filt2.loc[:,'nuc_cell_ratio'] = self.linked_filt2.loc[:,'mean_fluo_in_nuc']/self.linked_filt2.loc[:,'mean_fluo_cell']
        self.linked_filt2.loc[:,'nuc_cyto_ratio'] = self.linked_filt2.loc[:,'mean_fluo_in_nuc']/self.linked_filt2.loc[:,'mean_fluo_cyto']
        
        # Optional: save diagnostic outline video
        if save_outlines:
            self.save_cell_outline_video(
            cyto=self.fluo,             
            cyto_labels=self.cyto_labels,
            markers=self.markers,
            linked_df=self.linked_filt2,
            out_path=os.path.join(self.folder, "cell_tracking_outlines.tif")
            )
        # Classify NFAT translocation
        particles = self.linked_filt2["particle"].unique()
        trans = 0
        cells = 0
        self.results = {}
        for pid in particles:
            cells+= 1
            is_trans = 0
            d = self.linked_filt2[self.linked_filt2["particle"] == pid].sort_values("frame").copy()
            higher_1 = d.nuc_cyto_ratio > thr
            # Check whether N/C ratio exceeds threshold for ≥4 consecutive frames
            start_frame = self._first_n_consecutive_ones(higher_1, n)
            if start_frame is not None:
                true_frame = d["frame"].values[start_frame]*30
                trans +=1
                is_trans = 1
            else:
                true_frame = None
            self.results[pid] = [is_trans, true_frame]
            
        return  self.results, self.linked_filt2
#%% Find all .nd2 files
path = r'P:\10 CART Chi\6. All data\3. NFAT translocation\with Hoechst nucleus stain'
nd2_files = []

for root, dirs, files in os.walk(path):
    for f in files:
        if f.lower().endswith(".nd2"):
            nd2_files.append(os.path.join(root, f))

#%% Run analuysis for all nd2 files and make df with all results
all_dfs = []
for folder in nd2_files:
    print('Processing:', folder)
    results, linked = NFAT(folder).analyze_NFAT(save_outlines=False, thr = 1.1, n = 5)
    results_df = (
            pd.DataFrame.from_dict(results, orient='index')
            .rename(columns={0: 'translocated', 1: 'start_frame'})
            .rename_axis('cell_id')
            .reset_index()
        )
    results_df['folder'] = folder
    all_dfs.append(results_df)
#%%
all_results = pd.concat(all_dfs, ignore_index=True)
all_results2 = all_results[['folder', 'cell_id','translocated', 'start_frame']].copy()
#%% Calculate proportion of cells that show NFAT translocation
stats = all_results2.groupby('folder')['translocated'].agg(['sum', 'count'])
stats['prop'] = stats['sum'] / stats['count']
stats = stats.reset_index()  # Move 'folder' to a column
stats['expr'] = stats['folder'].str.split(r'\\').str[5] #add expression level
stats['dil'] = stats['folder'].str.split(r'\\').str[6] # add dilution level
stats['CAR'] = stats['folder'].str.split(r'\\').str[8] # Add CAR variant
stats.loc[stats['dil'] == '10His-SNAP', 'CAR'] = "10His-SNAP" #Set controls as controls in CAR too
stats.to_csv(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\nfat_translocation_stats.csv')
all_results2['expr'] = all_results2['folder'].str.split(r'\\').str[5] #add expression level
all_results2['dil'] = all_results2['folder'].str.split(r'\\').str[6] # add dilution level
all_results2['CAR'] = all_results2['folder'].str.split(r'\\').str[8] # Add CAR variant
all_results2.loc[all_results2['dil'] == '10His-SNAP', 'CAR'] = "10His-SNAP" #Set controls as controls in CAR too
all_results2.to_csv(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\nfat_translocation.csv')

#%% Plot
def p_to_text(p):
    """
    Convert a p-value into a compact string for plot annotation.
    """
    if p < 1e-4: return "p<1e-4"
    if p < 1e-3: return "p<1e-3"
    if p < 1e-2: return "p<0.01"
    return f"p={p:.3f}"

def add_sig_bar(ax, x1, x2, y, h, text):
    """
    Draw a statistical significance bracket between two x-positions on a plot.
    Notes
    -----
    - Assumes the y-axis is already scaled appropriately.
    - Does not automatically avoid overlap; vertical stacking must be handled externally.
    """
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')
    ax.text((x1+x2)/2, y+h, text, ha='center', va='bottom')

# Make plot
fig, ax = plt.subplots(figsize=(8,6))
sns.boxplot(data=stats, y='prop', x='CAR', hue='dil', showfliers=False, ax=ax)
sns.stripplot(data=stats, y='prop', x='CAR', color='black', alpha=0.5, jitter=True, ax=ax)
ax.set_ylabel('Proportion of cells with NFAT translocation')
ax.set_ylim(0, 1)


#  Determine postions of the boxes
cars = list(stats["CAR"].unique())
dils = list(stats["dil"].unique())

group_width = 0.8
step = group_width / len(dils)
offsets = np.linspace(-group_width/2 + step/2,
                      group_width/2 - step/2,
                      len(dils))

positions = {}
for i, car in enumerate(cars):
    for j, dil in enumerate(dils):
        positions[(car, dil)] = i + offsets[j]

# Set list comparisons for stats
comparisons = []

comparisons.append((("CART3Hi","100xdilutedCD19"),
                    ("CART4Hi","100xdilutedCD19")))

comparisons.append((("CART3Lo","100xdilutedCD19"),
                    ("CART4Lo","100xdilutedCD19")))

for car in ["CART3Hi","CART4Hi","CART3Lo","CART4Lo"]:
    comparisons.append((("10His-SNAP","10His-SNAP"),
                        (car,"100xdilutedCD19")))

#format plot and add stats
y_base = 0.85
gap = 0.05
h = 0.015
k = 0

for (g1, g2) in comparisons:

    car1, dil1 = g1
    car2, dil2 = g2

    vals1 = stats.loc[(stats["CAR"]==car1) &
                      (stats["dil"]==dil1), "prop"].values
    vals2 = stats.loc[(stats["CAR"]==car2) &
                      (stats["dil"]==dil2), "prop"].values

    if len(vals1)<2 or len(vals2)<2:
        continue

    U,p = mannwhitneyu(vals1, vals2)

    x1 = positions[(car1,dil1)]
    x2 = positions[(car2,dil2)]
    y = y_base + k*gap

    add_sig_bar(ax, x1, x2, y, h, p_to_text(p))
    k += 1

ax.legend(title="dil", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.savefig(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig3_NFAT.pdf', dpi = 600)
plt.savefig(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig3_NFAT.png', dpi = 600)
plt.tight_layout()
#%%#%% Plot for time
def p_to_text(p):
    """
    Convert a p-value into a compact string for plot annotation.
    """
    if p < 1e-4: return "p<1e-4"
    if p < 1e-3: return "p<1e-3"
    if p < 1e-2: return "p<0.01"
    return f"p={p:.3f}"

def add_sig_bar(ax, x1, x2, y, h, text):
    """
    Draw a statistical significance bracket between two x-positions on a plot.
    Notes
    -----
    - Assumes the y-axis is already scaled appropriately.
    - Does not automatically avoid overlap; vertical stacking must be handled externally.
    """
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')
    ax.text((x1+x2)/2, y+h, text, ha='center', va='bottom')

# Make plot
fig, ax = plt.subplots(figsize=(8,6))
sns.boxplot(data=all_results2, y='start_frame', x='CAR', hue='dil', showfliers=False, ax=ax)
sns.stripplot(data=all_results2, y='start_frame', x='CAR', color='black', alpha=0.5, jitter=True, ax=ax)
ax.set_ylabel('Time start of NFAT translocation (sec)')
# ax.set_ylim(0, 1)


#  Determine postions of the boxes
cars = list(stats["CAR"].unique())
dils = list(stats["dil"].unique())

group_width = 0.8
step = group_width / len(dils)
offsets = np.linspace(-group_width/2 + step/2,
                      group_width/2 - step/2,
                      len(dils))

positions = {}
for i, car in enumerate(cars):
    for j, dil in enumerate(dils):
        positions[(car, dil)] = i + offsets[j]

# Set list comparisons for stats
comparisons = []

comparisons.append((("CART3Hi","100xdilutedCD19"),
                    ("CART4Hi","100xdilutedCD19")))

comparisons.append((("CART3Lo","100xdilutedCD19"),
                    ("CART4Lo","100xdilutedCD19")))

for car in ["CART3Hi","CART4Hi","CART3Lo","CART4Lo"]:
    comparisons.append((("10His-SNAP","10His-SNAP"),
                        (car,"100xdilutedCD19")))

#format plot and add stats
ymin, ymax = ax.get_ylim()
yrange = ymax - ymin

y_base = ymax * 0.95          # start near top of plot
gap = yrange * 0.06           # vertical spacing between brackets
h = yrange * 0.02 
k= 0

for (g1, g2) in comparisons:

    car1, dil1 = g1
    car2, dil2 = g2

    vals1 = all_results2.loc[(all_results2["CAR"]==car1) &
                         (all_results2["dil"]==dil1),
                         "start_frame"].dropna().values

    vals2 = all_results2.loc[(all_results2["CAR"]==car2) &
                         (all_results2["dil"]==dil2),
                         "start_frame"].dropna().values

    if len(vals1)<2 or len(vals2)<2:
        continue

    U,p = mannwhitneyu(vals1, vals2)

    x1 = positions[(car1,dil1)]
    x2 = positions[(car2,dil2)]
    y = y_base + k*gap

    add_sig_bar(ax, x1, x2, y, h, p_to_text(p))
    k += 1

ax.legend(title="dil", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.savefig(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig3_NFAT_time.pdf', dpi = 600)
plt.savefig(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig3_NFAT_time.png', dpi = 600)
plt.tight_layout()