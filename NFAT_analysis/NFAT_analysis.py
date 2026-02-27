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
#%%
class NFAT: 
    """
    A class for analyzing NFAT translocation in ND2 image files.
    
    This class handles loading ND2 images, segmenting nuclei and cytoplasm,
    tracking cells, and computing fluorescence ratios for NFAT translocation detection.
    """
    def __init__(self, file_path):
        self.folder = os.path.split(file_path)[0]    
        self.open_image(file_path)
    def open_image(self, file_path):
        """
        Load and extract channels from the ND2 file.
        
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
        Segment nuclei in a single frame.
        
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
        Segment cytoplasm in a single frame based on nuclei markers.
        
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
    def _has_n_consecutive_ones(seld, arr, n=4):
        kernel = np.ones(n)
        conv = np.convolve(arr, kernel, mode='valid')
        return np.any(conv == n)
    def analyze_NFAT(self, thr = 1.05, save_outlines=False): 
        """
        Perform full NFAT translocation analysis.
        
        Parameters:
        save_outlines (bool): Whether to save outline video.
        
        Returns:
        results dict
        linked DataFrame
        """
        self.process_nuclei_image()

        self.segment_channels()
        self.remove_nuclei_from_cells()

        nuc_to_cell = self.assign_nuc_to_cell()
        df_to_track = self.labels_to_trackpy_features()
        df_to_track = df_to_track.merge(
            nuc_to_cell,
            left_on=["frame", "label"],
            right_on=["frame", "nuc_label"],
            how="left"
        ).drop(columns=["label"])

        linked = tp.link_df(df_to_track, search_range=25, memory=2)
        track_len = linked.groupby("particle")["frame"].transform("nunique")
        linked_filt = linked[track_len >= self.nuc_corr.shape[0]/3].copy()

        measurements = [
            # nuclear region
            {"mask": self.markers,            "image": self.fluo,     "name": "mean_fluo_in_nuc"},
            # cytoplasm (excluding nucleus)
            {"mask": self.cyto_labels,        "image": self.fluo,     "name": "mean_fluo_cell"},
            # cytoplasm (excluding nucleus)
            {"mask": self.cyto_only, "image": self.fluo,     "name": "mean_fluo_cyto"},
        ]
        tables = self.compute_all_means(measurements)
        for df in tables:
            linked_filt = linked_filt.merge(df, on=["frame", "nuc_label"], how="left")  

        self.linked_filt2 = linked_filt.dropna().copy()
        self.linked_filt2.loc[:,'nuc_cell_ratio'] = self.linked_filt2.loc[:,'mean_fluo_in_nuc']/self.linked_filt2.loc[:,'mean_fluo_cell']
        self.linked_filt2.loc[:,'nuc_cyto_ratio'] = self.linked_filt2.loc[:,'mean_fluo_in_nuc']/self.linked_filt2.loc[:,'mean_fluo_cyto']
        if save_outlines:
            self.save_cell_outline_video(
            cyto=self.fluo,             
            cyto_labels=self.cyto_labels,
            markers=self.markers,
            linked_df=self.linked_filt2,
            out_path=os.path.join(self.folder, "cell_tracking_outlines.tif")
            )

        # rng = np.random.default_rng(0)
        particles = self.linked_filt2["particle"].unique()
        # k = min(10, len(particles))
        # pick = rng.choice(particles, size=k, replace=False)
        # window = 5
        trans = 0
        cells = 0
        self.results = {}
        for pid in particles:
            cells+= 1
            is_trans = 0
            d = self.linked_filt2[self.linked_filt2["particle"] == pid].sort_values("frame").copy()
            higher_1 = d.nuc_cyto_ratio > thr
            if self._has_n_consecutive_ones(higher_1, 4):
                trans +=1
                is_trans = 1
            self.results[pid] = is_trans
            # d["nuc_cyto_ratio_roll"] = (
            #     d["nuc_cyto_ratio"]
            #     .rolling(window=window, center=True, min_periods=1)
            #     .mean()
            # )
            # d["nuc_cell_ratio_roll"] = (
            #     d["nuc_cell_ratio"]
            #     .rolling(window=window, center=True, min_periods=1)
            #     .mean()
            # )
            # frames = list(d.frame)
            # if len(frames) < 20:
            #     continue
            # cells +=1
            # y = np.log(d["nuc_cell_ratio_roll"].to_numpy() + 1e-8)
            # x = d["frame"].to_numpy().astype(float)
            # X = x.reshape(-1, 1)

            # # robust trend estimation
            # model = TheilSenRegressor(random_state=0)
            # model.fit(X, y)

            # slope = float(model.coef_[0])   # log fold-change per frame
            # yhat = model.predict(X)
            # resid = y - yhat

            # # robust noise estimate (MAD)
            # noise = 1.4826 * np.median(np.abs(resid))

            # # total duration in frames
            # Tspan = float(x.max() - x.min())

            # # biological minimum: 10% increase over the movie
            # min_total_FC = 1.10
            # min_slope = np.log(min_total_FC) / Tspan

            # # noise-based safeguard (3× noise)
            # noise_slope = 3.0 * noise / Tspan

            # # final threshold
            # thr_slope = max(min_slope, noise_slope)
            # is_trans = slope >= thr_slope

            # # store result (convert back to fold-change for reporting)
            # total_FC = float(np.exp(slope * Tspan))
            # self.results[pid] = [int(is_trans), total_FC]

            # if is_trans:
            #     trans += 1

        #     ### --- diagnostic plot (keeps your original style) ---
        #     plt.figure(figsize=(4, 3))
        #     # plt.plot(d["frame"], np.log(d["nuc_cell_ratio_roll"] + 1e-8), label="nuc / cell", marker="o")
        #     # plt.plot(d["frame"], np.log(d["nuc_cyto_ratio_roll"] + 1e-8), label="nuc / cyto", marker="o")
        #     plt.plot(d["frame"], (d["nuc_cell_ratio_roll"] + 1e-8), label="nuc / cell", marker="o")
        #     plt.plot(d["frame"],(d["nuc_cyto_ratio_roll"] + 1e-8), label="nuc / cyto", marker="o")
        #     plt.plot(d["frame"], yhat, label="trend (nuc/cyto)", linewidth=2)

        #     plt.xlabel("frame")
        #     plt.ylabel("ratio")
        #     plt.title(
        #         f"Particle {int(pid)} | FC={total_FC:.3f} | slope={slope:.3g}"
        #     )
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.show()
        #     start = d[d.frame <=frames[10]]
        #     end = d[d.frame >=frames[-10]]

        #     ratio = end['nuc_cyto_ratio_roll'].median()/start['nuc_cyto_ratio_roll'].median()
        #     if ratio >= 1.05:
        #         self.results[pid] = [1,ratio]
        #         trans += 1
        #     else: 
        #         self.results[pid] = [0,ratio]

        #     plt.figure(figsize=(4, 3))
        #     plt.plot(d["frame"], d["nuc_cell_ratio_roll"], label="nuc / cell", marker="o")
        #     plt.plot(d["frame"], d["nuc_cyto_ratio_roll"], label="nuc / cyto", marker="o")

        #     plt.xlabel("frame")
        #     plt.ylabel("ratio")
        #     plt.title(f"Particle {int(pid)}, ratio = {ratio}")
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.show()

        # print(f'number cells: {cells}')
        # print(f'number trans: {trans}')
        # print(f'percentage trans {trans/cells}')
        return  self.results, self.linked_filt2
#%%
path = r'P:\10 CART Chi\6. All data\3. NFAT translocation\with Hoechst nucleus stain'
nd2_files = []

for root, dirs, files in os.walk(path):
    for f in files:
        if f.lower().endswith(".nd2"):
            nd2_files.append(os.path.join(root, f))


# nfat_obj = NFAT(nd2_files[0])
# results, linked_filt2 = nfat_obj.analyze_NFAT(save_outlines=True)
#%%
all_results = pd.DataFrame()
for folder in nd2_files:
    print('Processing:', folder)
    results, linked  = NFAT(folder).analyze_NFAT(save_outlines=False)
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['translocated'])
    results_df['folder'] = folder 
    all_results = pd.concat([all_results, results_df], ignore_index=True)
#%%
all_results2 = all_results[['folder', 'translocated']]
stats = all_results2.groupby('folder')['translocated'].agg(['sum', 'count'])
stats['prop'] = stats['sum'] / stats['count']
stats = stats.reset_index()  # Move 'folder' to a column
stats['expr'] = stats['folder'].str.split(r'\\').str[5]
stats['dil'] = stats['folder'].str.split(r'\\').str[6]
stats['CAR'] = stats['folder'].str.split(r'\\').str[8]
#%%
# all_results.loc[all_results.ratio >= 1, 'translocated'] = 1
# all_results.loc[all_results.ratio < 1, 'translocated'] = 0
stats = all_results.groupby('folder')['translocated'].agg(['sum', 'count'])
stats['prop'] = stats['sum'] / stats['count']
stats = stats.reset_index()  # Move 'folder' to a column
stats['expr'] = stats['folder'].str.split(r'\\').str[5]
stats['dil'] = stats['folder'].str.split(r'\\').str[6]
stats['CAR'] = stats['folder'].str.split(r'\\').str[8]

#%%
import seaborn as sns

from scipy.stats import mannwhitneyu
stats_or = stats.copy()
groups = stats.groupby(["CAR", 'expr', 'dil'])["prop"]
labels = list(groups.groups.keys())

# g1 = groups.get_group(labels[0])
# g2 = groups.get_group(labels[1])

# u, p = mannwhitneyu(g1, g2, alternative="two-sided")

# y = stats["prop"].max() + 0.05
stats.loc[stats.dil == '10His-SNAP', 'CAR'] = '10His-SNAP'


fig, ax = plt.subplots(figsize=(8,6))
sns.boxplot(data=stats, y='prop', x='CAR', hue= 'dil', showfliers=False)
sns.stripplot(data=stats, y='prop', x='CAR', color='black', alpha=0.5, jitter=True)
# ax.plot([0, 1], [y, y], color="black")
# ax.text(
#     0.5,
#     y + 0.02,
#     f"p = {p:.4f}",   # <-- decimal notation
#     ha="center"
# )
plt.ylabel('Proportion of cells with NFAT translocation')
plt.ylim(0,1)
#%%
from scipy.stats import mannwhitneyu
import itertools

pairwise_results = []

# unique groups exactly as shown in plot
groups = stats["CAR"].unique()

for g1, g2 in itertools.combinations(groups, 2):

    vals1 = stats.loc[stats["CAR"] == g1, "prop"].values
    vals2 = stats.loc[stats["CAR"] == g2, "prop"].values

    if len(vals1) < 2 or len(vals2) < 2:
        continue

    U, p = mannwhitneyu(vals1, vals2, alternative="two-sided")

    pairwise_results.append({
        "group1": g1,
        "group2": g2,
        "n1": len(vals1),
        "n2": len(vals2),
        "median1": np.median(vals1),
        "median2": np.median(vals2),
        "U_stat": U,
        "p_value": p
    })

pairwise_table = pd.DataFrame(pairwise_results)

pairwise_table = pairwise_table.sort_values("p_value").reset_index(drop=True)

pairwise_table[['group1', 'group2', 'p_value']]