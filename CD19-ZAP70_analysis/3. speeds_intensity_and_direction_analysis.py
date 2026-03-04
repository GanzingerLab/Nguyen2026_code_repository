#%%
import os
from postSPIT import plotting_classes as plc
import pandas as pd 
import numpy as np
from natsort import natsorted
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm 
import re
from itertools import combinations
def find_run_folders(path):
    """
    Recursively find and return paths to directories whose names match a "^Run*" pattern.
    Returns a list with the paths
    """
    run_pattern = re.compile(r'^Run*') 
    run_paths = []
    for root, dirs, files in os.walk(path):
        matching_dirs = [d for d in dirs if run_pattern.match(d)]
        for d in matching_dirs:
            full_path = os.path.join(root, d)
            run_paths.append(full_path)
        dirs[:] = [d for d in dirs if not run_pattern.match(d)]
    return run_paths

def smooth_with_nans(series, sigma=2):
    """
    Smooth a pandas Series with optional NaN values using a Gaussian filter,
    while preserving the original NaN positions.

    Returns an np.ndarray
    """
    nan_mask = series.isna()  # mask original NaNs
    filled = series.interpolate(method='linear').bfill().ffill()  # fill gaps for smoothing
    smoothed = gaussian_filter1d(filled, sigma=sigma)  # Gaussian smoothing
    smoothed[nan_mask] = np.nan  # restore NaNs
    return smoothed

def compute_radial_metrics(
    tracks ,
    stats,
    nm2px=108,
    t_col="time_s",
    coloc_col = "colocID",
):
    """
    Compute and append instantaneous radial motion metrics per row for both (CD19 and ZAP70) tracks.

    This function adds multiple columns describing:
      - distance to the cell-center centroid,
      - displacement magnitude,
      - radial displacement component toward/away from the centroid,
      - Directionality or radial_fraction (cosine of angle between displacement and radial direction),
      - radial speeds computed from projection and from distance change,

    Parameters
    ----------
    tracks : pd.DataFrame
        Colocalizing tracks from SPIT.
    stats : pd.DataFrame
        Stats of the colocalizing tracks from SPIT.
    nm2px : float, optional
        Pixel size. Default is 108 for K2 microscope.
    t_col : str, optional
        Column name for physical time (seconds) used to compute velocities. Default "time_s".
    coloc_col : str, optional
        Column name identifying a colocalized track group. Default "colocID".

    Returns
    -------
    pd.DataFrame
        Copy of `tracks` with additional columns for each metric and each particle:
        e.g. dist_to_center_0, radial_speed_from_dist_1, angle_cos_0, etc.
    """
    tracks = tracks.copy()

    # Build centroid dict {colocID -> (cx_px, cy_px)} in pixel units.
    centroid_map = {}
    for cid, g in stats.groupby(coloc_col):
        c_nm = g["centroid"].values[0][0]  # (x_nm, y_nm)
        cx_px = c_nm[0] * nm2px
        cy_px = c_nm[1] * nm2px
        centroid_map[cid] = (cx_px, cy_px)

    # List of per-row metrics we will compute and store for each particle.
    radial_cols = [
    "dist_to_center",
    "disp",
    "radial_component",
    "radial_fraction",
    "radial_speed",
    "delta_dist",
    "radial_speed_from_dist",
    "angle_cos",
]
    # Allocate output columns for particle 0 and particle 1; initialize with NaNs.
    for col in radial_cols:
        tracks[f"{col}_0"] = np.nan
        tracks[f"{col}_1"] = np.nan

    # Helper: compute metric arrays for a single particle trajectory (x, y, t) relative to (cx, cy).
    # Returns arrays aligned to input length by padding with a leading NaN for step-based quantities.
    def _compute_metrics_arrays(x, y, t, cx, cy):
        # Vector from particle position to center at each timepoint.
        rx = cx - x
        ry = cy - y
        # Distance from particle to center at each timepoint.
        dist = np.sqrt(rx**2 + ry**2)
        # Step-to-step displacement in x/y and corresponding time step.
        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(t)
        # Guard against non-positive time steps (kept as NaN to avoid division errors).
        dt_safe = np.where(dt <= 0, np.nan, dt)
        # Step-to-step displacement magnitude.
        disp = np.sqrt(dx**2 + dy**2)
        # Norm of radial vector at the START of each step (length n-1).
        rnorm = np.sqrt(rx[:-1]**2 + ry[:-1]**2)
        # Dot product between displacement and radial vector (start-of-step) for projection.
        dot = dx * rx[:-1] + dy * ry[:-1]
        # Radial displacement component (projection of displacement onto radial direction).
        radial_component = np.full_like(disp, np.nan)
        valid = rnorm > 0
        radial_component[valid] = dot[valid] / rnorm[valid]
        # Radial fraction or directionality -> cos(theta) between displacement and radial direction.
        radial_fraction = np.full_like(disp, np.nan)
        mask = (disp > 0) & (rnorm > 0)
        radial_fraction[mask] = dot[mask] / (disp[mask] * rnorm[mask])
        # Radial speed estimated from projection / dt.
        radial_speed = np.full_like(disp, np.nan)
        radial_speed[mask] = radial_component[mask] / dt_safe[mask]
        # Change in distance-to-center between frames; sign convention depends on dist definition.
        delta_dist = np.diff(dist)
        # Radial speed computed from distance-to-center derivative (negative means toward center if dist decreases).
        radial_speed_from_dist = -delta_dist / dt_safe

        # pad with leading NaN to align to length n
        def pad(a): return np.concatenate([[np.nan], a])

        return {
            "dist_to_center": dist,
            "disp": pad(disp),
            "radial_component": pad(radial_component),
            "radial_fraction": pad(radial_fraction),
            "radial_speed": pad(radial_speed),
            "delta_dist": pad(delta_dist),
            "radial_speed_from_dist": pad(radial_speed_from_dist),
            "angle_cos": pad(radial_fraction),
        }

    # Compute metrics per colocID group and write them back to the correct rows in `tracks`.
    for cid, g in tracks.groupby(coloc_col, sort=False):
        # Skip if we don't have a centroid for this colocID.
        if cid not in centroid_map:
            continue
        cx, cy = centroid_map[cid]
        # Ensure chronological order before differencing.
        g = g.sort_values(t_col)
        idxs = g.index.to_numpy()
        # Time array (seconds).
        t = g[t_col].to_numpy(float)

        # particle 0 (CD19)
        x0 = g['x_0'].to_numpy(float)
        y0 = g['y_0'].to_numpy(float)
        # Valid rows require finite x, y, and time.
        m0 = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(t)
        # Need at least two points to compute step-based metrics.
        if m0.sum() >= 2:
            arrs0 = _compute_metrics_arrays(x0[m0], y0[m0], t[m0], cx, cy)
            # write only where valid; keep NaN elsewhere
            for k in radial_cols:
                tracks.loc[idxs[m0], f"{k}_0"] = arrs0[k]


        # particle 1 (Zap70)
        x1 = g['x_1'].to_numpy(float)
        y1 = g['y_1'].to_numpy(float)
        m1 = np.isfinite(x1) & np.isfinite(y1) & np.isfinite(t)

        if m1.sum() >= 2:
            arrs1 = _compute_metrics_arrays(x1[m1], y1[m1], t[m1], cx, cy)
            for k in radial_cols:
                tracks.loc[idxs[m1], f"{k}_1"] = arrs1[k]
    return tracks

def compute_velocities(df):
    """
    Compute instantaneous velocities for each particle track.

    Velocities are computed per-channel ('track.id_0' and 'track.id_1')
    and for colocalized tracks ('colocID'), using the physical time
    column 'time_s'.

    The velocity is computed as frame-to-frame displacement divided by frame-to-frame
    time difference using 'time_s'.

    Parameters
    ----------
    tracks : pd.DataFrame
        Colocalizing tracks from SPIT

    Returns
    -------
    pd.DataFrame
        Copy of `df` with additional columns for each particle:
    """
    # Particle 0 velocity (per track.id_0), computed as sqrt(dx^2 + dy^2) / dt.

    df['velocity_0'] = df.groupby('track.id_0').apply(
        lambda g: np.sqrt(g['x_0'].diff()**2 + g['y_0'].diff()**2) / g['time_s'].diff()
    ).reset_index(level=0, drop=True)
    # Particle 1 velocity (per track.id_1).
    df['velocity_1'] = df.groupby('track.id_1').apply(
        lambda g: np.sqrt(g['x_1'].diff()**2 + g['y_1'].diff()**2) / g['time_s'].diff()
    ).reset_index(level=0, drop=True)
    # Colocalized track velocity (per colocID) from combined (x, y) coordinates
    df['velocity_coloc'] = df.groupby('colocID').apply(
        lambda g: np.sqrt(g['x'].diff()**2 + g['y'].diff()**2) / g['time_s'].diff()
    ).reset_index(level=0, drop=True)

    return df

def compute_intensities(df, ch0, ch1, nm2px,):
    """
    Add per-row normalized 3x3 patch intensities from image stacks.

     For each row in `df` (when coordinates exist):
      - Use frame index = int(df['t'])
      - Channel 0 (CD19): compute mean intensity in a 3x3 patch centered at (y_0, x_0)
        and normalize by the frame median.
      - Channel 1 (ZAP70): compute mean intensity in a 3x3 patch centered at (y_1, x_1)
        and normalize by the frame median.
    Parameters
    ----------
    df : pd.DataFrame
        Colocalizing tracks from SPIT
    ch0 : Image stack for channel 0 with shape (T, H, W).
    ch1 : Image stack for channel 1 with shape (T, H, W).
    nm2px : Conversion factor from nm to pixels.
    Returns
    -------
    pd.DataFrame
        Copy of `df` with added columns:
          - 'im_int_0'
          - 'im_int_1'
    """
    df = df.copy()
    # Initialize intensity columns with NaNs (will be filled where coordinates exist).
    df['im_int_0'] = np.nan
    df['im_int_1'] = np.nan

    # precompute per-frame medians once for background normalization of the image
    ch0_meds = np.array([np.median(fr) for fr in ch0], dtype=float)
    ch1_meds = np.array([np.median(fr) for fr in ch1], dtype=float)

    def patch3x3_mean(img, ys, xs):
        # vectorized 3x3 mean at multiple (y,x) locations (ys/xs are int arrays)
        return (
            img[ys-1, xs-1] + img[ys-1, xs] + img[ys-1, xs+1] +
            img[ys,   xs-1] + img[ys,   xs] + img[ys,   xs+1] +
            img[ys+1, xs-1] + img[ys+1, xs] + img[ys+1, xs+1]
        ) / 9.0

    #channel 0
    m0 = df['y_0'].notna() & df['x_0'].notna()
    if m0.any():
        # Frame indices for valid rows.
        t0 = df.loc[m0, 't'].astype(int).to_numpy()
        # Convert nm coordinates to pixel indices (integer) and clip to safe patch centers.
        y0 = (df.loc[m0, 'y_0'].to_numpy() / nm2px).astype(int)
        x0 = (df.loc[m0, 'x_0'].to_numpy() / nm2px).astype(int)
        H0, W0 = ch0[0].shape
        y0 = np.clip(y0, 1, H0 - 2)
        x0 = np.clip(x0, 1, W0 - 2)
        # Compute normalized patch mean per row, iterating per unique frame for efficiency.
        im0 = np.empty_like(t0, dtype=float)
        for fr in np.unique(t0):
            sel = (t0 == fr) # boolean mask selecting rows that belong to this frame
            frame = ch0[fr] # 2D image for this frame
            im0[sel] = patch3x3_mean(frame, y0[sel], x0[sel]) / ch0_meds[fr] #vectorized calculation of means for each localization 

        df.loc[m0, 'im_int_0'] = im0

    # channel 1
    m1 = df['y_1'].notna() & df['x_1'].notna()
    if m1.any():
        t1 = df.loc[m1, 't'].astype(int).to_numpy()
        y1 = (df.loc[m1, 'y_1'].to_numpy() / nm2px).astype(int)
        x1 = (df.loc[m1, 'x_1'].to_numpy() / nm2px).astype(int)

        H1, W1 = ch1[0].shape
        y1 = np.clip(y1, 1, H1 - 2)
        x1 = np.clip(x1, 1, W1 - 2)

        im1 = np.empty_like(t1, dtype=float)
        for fr in np.unique(t1):
            sel = (t1 == fr)
            frame = ch1[fr]
            im1[sel] = patch3x3_mean(frame, y1[sel], x1[sel]) / ch1_meds[fr]

        df.loc[m1, 'im_int_1'] = im1

    return df

def filter_and_compute_stats(tracks, stats, subdf, folder, dt, coloc_col = "colocID", t_col = "time_s"):
    """
    Filter tracks to colocIDs within subdf, then compute per-timepoint and summary statistics for both 
    particles (CD19 and ZAP70) across pre/during/post colocalization phases.

    Parameters
    ----------
    tracks : pd.DataFrame
        Colocalizing tracks from SPIT with (optional) instantaneous speeds, intensity, and directionality parameters:
    stats : pd.DataFrame
        Stats of the colocalizing tracks from SPIT.
    subdf : pd.DataFrame
        Subset dataframe defining which colocIDs to analyze for this folder.
    folder : str
        Folder path for the current run; used for output annotations.
    dt : float
        Time per frame (seconds/frame). Used to convert overlap frame indices to seconds.
    coloc_col : str, optional
        Name of the coloc ID column. Default "colocID".
    t_col : str, optional
        Name of the time column in seconds. Default "time_s".

    Returns
    -------
    all_results : list[dict]
        List of per-track, per-phase summary rows (one per particle per phase).
    all_timepoints_df : pd.DataFrame
        Long-form per-timepoint dataframe combining both particles and phases.
    """
    all_results = []
    all_timepoints = []
    # Parse condition from folder path (assumes fixed depth).
    condition = folder.split("\\")[5]
    # only analyze colocIDs within subdf
    coloc_ids = subdf[coloc_col].unique()
    tracks_f = tracks[tracks[coloc_col].isin(coloc_ids)].copy()

    for track_id, g in tracks_f.groupby(coloc_col, sort=False):
        # Sort by time so phases and derivatives are consistent.
        g = g.sort_values(t_col).reset_index(drop=True)

        # convert overlapping frames to seconds to match time indices
        overlap_frames = stats.loc[stats[coloc_col] == track_id, "overlap_t"].values
        if len(overlap_frames) == 0:
            continue
        coloc = overlap_frames[0] * dt
        start_sec, end_sec = coloc[0], coloc[-1]
        # Build boolean masks for each phase relative to the overlap window.
        time = g[t_col].to_numpy(float)
        pre = time < start_sec
        during = (time >= start_sec) & (time <= end_sec)
        post = time > end_sec

        # build long per-timepoint df (CD19 + Zap70) from precomputed columns 
        def particle_long(particle, suffix):
            """
            Convert one particle's per-row metrics (suffix 0/1) into a long-form dataframe.
            """
            out = pd.DataFrame({
                "colocID": track_id,
                "run": folder,
                "condition": condition,
                "particle": particle,
                "phase": np.where(pre, "pre", np.where(during, "during", "post")),
                "t": g["t"].to_numpy() if "t" in g else np.arange(len(g)),
                "time_s": time,
                "velocity": g[f"velocity_{suffix}"].to_numpy() if f"velocity_{suffix}" in g else np.full(len(g), np.nan),
                "intensity": g[f"im_int_{suffix}"].to_numpy() if f"im_int_{suffix}" in g else np.full(len(g), np.nan),
                "dist_to_center": g[f"dist_to_center_{suffix}"].to_numpy(),
                "radial_speed_from_dist": g[f"radial_speed_from_dist_{suffix}"].to_numpy(),
                "angle_cos": g[f"angle_cos_{suffix}"].to_numpy(),
            })
            return out
        # Build and store per-timepoint metrics for both particles.
        df_cd19 = particle_long("CD19", "0")
        df_zap70 = particle_long("Zap70", "1")
        full_metrics_df = pd.concat([df_cd19, df_zap70], ignore_index=True)
        all_timepoints.append(full_metrics_df)

        # --- summaries ---
        def summarize(particle, suffix, mask, timing_label):
            """
            Compute summary statistics for one particle within a selected phase.
            """
            if mask.sum() == 0:
                return
            # Pull per-row arrays for this phase
            v = g.get(f"velocity_{suffix}", pd.Series(np.nan, index=g.index)).to_numpy()[mask]
            rs = g[f"radial_speed_from_dist_{suffix}"].to_numpy()[mask]
            ang = g[f"angle_cos_{suffix}"].to_numpy()[mask]
            inten = g.get(f"im_int_{suffix}", pd.Series(np.nan, index=g.index)).to_numpy()[mask]
            # Compute averages ignoring NaNs.
            avg_speed = np.nanmean(v) if np.isfinite(v).any() else np.nan
            avg_radial = np.nanmean(rs) if np.isfinite(rs).any() else np.nan
            directionality = np.nanmean(ang) if np.isfinite(ang).any() else np.nan
            avg_intensity = np.nanmean(inten) if np.isfinite(inten).any() else np.nan
            median_intensity = np.nanmedian(inten) if np.isfinite(inten).any() else np.nan

            # Linear slope of distance-to-center vs time within this phase (seconds).
            tseg = time[mask]
            dseg = g[f"dist_to_center_{suffix}"].to_numpy()[mask]
            ok = np.isfinite(tseg) & np.isfinite(dseg)
            slope = np.polyfit(tseg[ok], dseg[ok], 1)[0] if ok.sum() > 1 and len(np.unique(tseg[ok])) > 1 else np.nan

            all_results.append({
                "colocID": track_id,
                "run": folder,
                "condition": condition,
                "particle": particle,
                "timing": timing_label,
                "avg_speed": avg_speed,
                "directionality": directionality,
                "slope_distance": slope,
                "avg_radial_speed": avg_radial,
                "avg_intensity": avg_intensity,
                "median_intensity": median_intensity,
            })
        # Compute summary stats for each phase and each particle.
        for timing, mask in [("pre", pre), ("during", during), ("post", post)]:
            summarize("CD19", "0", mask, timing)
            summarize("Zap70", "1", mask, timing)
    # Concatenate all per-timepoint particle dataframes into one long dataframe.
    all_timepoints_df = pd.concat(all_timepoints, ignore_index=True)
    return all_results, all_timepoints_df

def safe_cos(dx, dy, rx, ry):
    """
    Compute cosine of the angle between a displacement vector and a radial-to-center vector.
    """
    #calculate dot product of the movement vector and the distance to the center vector
    dot = dx * rx + dy * ry
    #calculate magnitude of movement vector
    D = np.hypot(dx, dy)
    #calculate magnitude of distance to the center vector
    R = np.hypot(rx, ry)
    #if the magnitudes are not 0, calculate the dot the angle between the two vectors and return them. 
    if D == 0 or R == 0:
        return np.nan, D, R
    return dot / (D * R), D, R
#%% Instantaneous speeds, intensities, and directionality paremeters
tracks_to_analyze = pd.read_csv(r'D:\Data\Chi_data\20250801_filtered\output\cotracks_longer_10frames.csv')

all_results = []
all_timepoints = []
# Process each run 
for folder, subdf in tqdm(tracks_to_analyze.groupby("folder")):
    print(f'Analysizng {folder}')
    #get necessary data
    run = plc.Combined_analysis(folder, ch0_hint='638nm', ch1_hint='488nm')
    tracks = run.cotracks_outside_clusters
    stats = run.cotracks_outside_clusters_stats
    ch0 = run.tracked.ch0
    ch1 = run.tracked.ch1

    #Transform frame indices to seconds
    dt = run.clusters.get_time_interval()
    tracks['time_s'] = tracks['t'] * dt


    #Compute instantaneous velocities, intensities, and radial metrics
    tracks = compute_velocities(tracks)
    tracks = compute_intensities(tracks, ch0, ch1, nm2px=run.nm2px)
    tracks = compute_radial_metrics(tracks, stats, nm2px=run.nm2px)
    results, full_metrics_df = filter_and_compute_stats(tracks, stats, subdf, folder, dt)
    all_results.extend(results)
    all_timepoints.append(full_metrics_df)
all_timepoints_df = pd.concat(all_timepoints, ignore_index=True)
all_results_df = pd.DataFrame(all_results)
#%%
all_results_df.to_csv(r'D:\Data\Chi_data\20250801_filtered\output\analysis2026\all_cotracks_radial_velocity_intensity_stats.csv', index=False)
all_timepoints_df.to_csv(r'D:\Data\Chi_data\20250801_filtered\output\analysis2026\all_cotracks_radial_velocity_intensity_timepoints.csv', index=False)
#%% per-cotrack phase directionalities directionalities
tracks_to_analyze = pd.read_csv(r'D:\Data\Chi_data\20250801_filtered\output\cotracks_longer_10frames.csv')

all_results = []

for folder, subdf in tqdm(tracks_to_analyze.groupby("folder")):
    base = os.path.join(folder, 'cluster_analysis_spots_filtered')
    tracks = pd.read_csv(os.path.join(base, '638nm_roi_locs_nm_trackpy_ColocsTracks.csv'))
    stats  = pd.read_hdf(os.path.join(base, '638nm_roi_locs_nm_trackpy_ColocsTracks_stats.hdf'))

    cell_id = stats['cell_id'].iat[0]

    # Only analyze the IDs requested for this folder
    for track_id in subdf["colocID"].unique():
        #extract stats from the track
        st = stats.loc[stats["colocID"] == track_id].iloc[0]
        #save centroid position in nm
        cx, cy = (np.array(st["centroid"][0]) * 108)  
        #save colocalization start and ending frame
        coloc = st["overlap_t"]
        start, end = coloc[0], coloc[-1]
        #get track (frame and position), remvoing possible NaNs
        tr = tracks.loc[tracks["colocID"] == track_id, ["t", "x_0", "y_0"]].dropna(subset=["x_0", "y_0"]).copy()
        if tr.empty:
            continue

        # Make frame the index, for faster indexing
        tr_by_t = tr.set_index("t")

        # Build the 4 landmark locations: 
        # loc0:start of the track
        # loc1:start of the colocalization
        # loc2:end of the colocalization
        # loc3:end of the track.
        try:
            locs = {
                "loc0": (tr["x_0"].iat[0],      tr["y_0"].iat[0],      0),
                "loc1": (tr_by_t.at[start, "x_0"], tr_by_t.at[start, "y_0"], start * 2),
                "loc2": (tr_by_t.at[end,   "x_0"], tr_by_t.at[end,   "y_0"], end   * 2),
                "loc3": (tr["x_0"].iat[-1],     tr["y_0"].iat[-1],     tr["t"].iat[-1] * 2),
            }
        except KeyError:
            # start/end not present in the track's t index
            continue
        #for each combination of landmark locations
        for p1, p2 in combinations(locs, 2):
            #extract positions and frame
            x1, y1, t1 = locs[p1]
            x2, y2, t2 = locs[p2]
            #make net movement vector
            dx, dy = x2 - x1, y2 - y1
            #make distance to the center vector
            rx, ry = cx - x1, cy - y1
            #calculate the directionality and the magnitude of the vectors. 
            cos_theta, D, R = safe_cos(dx, dy, rx, ry)
            # calculate the radial component (i.e. how much of the displacement is toward or away from the center)
            rad = D * cos_theta if np.isfinite(cos_theta) else np.nan
            #caluclate time passed 
            dt = (t2 - t1)
            #calculate radial speed. 
            rad_velocity = rad / dt if dt != 0 else np.nan
            # add result to results
            all_results.append({
                "run": folder,
                "cell": cell_id,
                "colocID": track_id,
                "from": p1,
                "to": p2,
                "dist_to_center": R,
                "directionality": cos_theta,
                "moved_distance": D,
                "radial_component": rad,
                "radial_velocity": rad_velocity,
            })
#transform the results into a dataframe 
directionalities = pd.DataFrame(all_results)

#add metadate 
directionalities["cond"] = directionalities["run"].astype(str).apply(lambda p: os.path.normpath(p).split(os.sep)[5])
#add transitions
transition_pairs = {("loc0", "loc1"), ("loc1", "loc2"), ("loc2", "loc3")}
pair = list(zip(directionalities["from"], directionalities["to"]))
#and keep only relevant transitions
directionalities["transition"] = np.where(
    pd.Series(pair).isin(transition_pairs),
    directionalities["from"] + "-" + directionalities["to"],
    pd.NA
)
#add maturation information. 
df_raw = pd.read_csv(r'D:\Data\Chi_data\20250801_filtered\output\maturation_count_withDil.csv')
directionalities_maturation = directionalities.merge(
    df_raw[["run", "cell", "category"]],
    on=["run", "cell"],
    how="left"
)