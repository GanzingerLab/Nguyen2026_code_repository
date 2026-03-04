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
    run_pattern = re.compile(r'^Run*') #now Run folders only need to have "Run" in their name
    #and look for all subfolders within path that contain Run in it. 
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

    Parameters
    ----------
    series : pd.Series
        Data to be smoothed. Can contain NaNs.
    sigma : float, optional
        Standard deviation for Gaussian. Controls the amount of smoothing.
        Larger values = stronger smoothing. Default is 2.

    Returns
    -------
    smoothed : np.ndarray
        Smoothed data as a NumPy array. NaNs are preserved in their
        original positions.
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
    Add per-row radial metrics for each particle dataframe.
    """
    tracks = tracks.copy()

    # build centroid dict {colocID -> (cx_px, cy_px)}
    centroid_map = {}
    for cid, g in stats.groupby(coloc_col):
        c_nm = g["centroid"].values[0][0]  # (x_nm, y_nm)
        cx_px = c_nm[0] * nm2px
        cy_px = c_nm[1] * nm2px
        centroid_map[cid] = (cx_px, cy_px)

    # allocate output columns (0 and 1)
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

    for col in radial_cols:
        tracks[f"{col}_0"] = np.nan
        tracks[f"{col}_1"] = np.nan


    # helper: compute metrics arrays for one particle inside one colocID group
    def _compute_metrics_arrays(x, y, t, cx, cy):
        rx = cx - x
        ry = cy - y
        dist = np.sqrt(rx**2 + ry**2)

        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(t)
        dt_safe = np.where(dt <= 0, np.nan, dt)
        disp = np.sqrt(dx**2 + dy**2)

        rnorm = np.sqrt(rx[:-1]**2 + ry[:-1]**2)
        dot = dx * rx[:-1] + dy * ry[:-1]

        radial_component = np.full_like(disp, np.nan)
        valid = rnorm > 0
        radial_component[valid] = dot[valid] / rnorm[valid]

        radial_fraction = np.full_like(disp, np.nan)
        mask = (disp > 0) & (rnorm > 0)
        radial_fraction[mask] = dot[mask] / (disp[mask] * rnorm[mask])

        radial_speed = np.full_like(disp, np.nan)
        radial_speed[mask] = radial_component[mask] / dt_safe[mask]

        delta_dist = np.diff(dist)
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

    # compute per colocID, fill rows back by index
    for cid, g in tracks.groupby(coloc_col, sort=False):
        if cid not in centroid_map:
            continue
        cx, cy = centroid_map[cid]

        g = g.sort_values(t_col)
        idxs = g.index.to_numpy()

        t = g[t_col].to_numpy(float)

        # particle 0 (CD19)
        x0 = g['x_0'].to_numpy(float)
        y0 = g['y_0'].to_numpy(float)
        m0 = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(t)

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

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with at least:
        - 'track.id_0', 'x_0', 'y_0'
        - 'track.id_1', 'x_1', 'y_1'
        - 'colocID', 'x', 'y'
        - 'time_s' (time in seconds)

    Returns
    -------
    df : pd.DataFrame
        Input dataframe with three new columns:
        - velocity_0
        - velocity_1
        - velocity_coloc
    """
    df['velocity_0'] = df.groupby('track.id_0').apply(
        lambda g: np.sqrt(g['x_0'].diff()**2 + g['y_0'].diff()**2) / g['time_s'].diff()
    ).reset_index(level=0, drop=True)

    df['velocity_1'] = df.groupby('track.id_1').apply(
        lambda g: np.sqrt(g['x_1'].diff()**2 + g['y_1'].diff()**2) / g['time_s'].diff()
    ).reset_index(level=0, drop=True)

    df['velocity_coloc'] = df.groupby('colocID').apply(
        lambda g: np.sqrt(g['x'].diff()**2 + g['y'].diff()**2) / g['time_s'].diff()
    ).reset_index(level=0, drop=True)

    return df

def compute_intensities(df, ch0, ch1, nm2px,):
    """
    Add per-row normalized 3x3 patch intensities from image stacks.

    For each row:
      - take frame index = int(df[t_col])
      - if (y_0,x_0) exists: intensity_0 = mean(ch0[t][y-1:y+2, x-1:x+2]) / median(ch0[t])
      - if (y_1,x_1) exists: intensity_1 = mean(ch1[t][y-1:y+2, x-1:x+2]) / median(ch1[t])

    Notes:
      - Clips coordinates to avoid border issues (needs 1..H-2, 1..W-2).
      - Loops over unique frames present in df (fast), not over rows.
    """
    df = df.copy()
    df['im_int_0'] = np.nan
    df['im_int_1'] = np.nan

    # precompute per-frame medians once
    ch0_meds = np.array([np.median(fr) for fr in ch0], dtype=float)
    ch1_meds = np.array([np.median(fr) for fr in ch1], dtype=float)

    def patch3x3_mean(img, ys, xs):
        # vectorized 3x3 mean at multiple (y,x) centers (ys/xs are int arrays)
        return (
            img[ys-1, xs-1] + img[ys-1, xs] + img[ys-1, xs+1] +
            img[ys,   xs-1] + img[ys,   xs] + img[ys,   xs+1] +
            img[ys+1, xs-1] + img[ys+1, xs] + img[ys+1, xs+1]
        ) / 9.0

    #channel 0
    m0 = df['y_0'].notna() & df['x_0'].notna()
    if m0.any():
        t0 = df.loc[m0, 't'].astype(int).to_numpy()
        y0 = (df.loc[m0, 'y_0'].to_numpy() / nm2px).astype(int)
        x0 = (df.loc[m0, 'x_0'].to_numpy() / nm2px).astype(int)
        H0, W0 = ch0[0].shape
        y0 = np.clip(y0, 1, H0 - 2)
        x0 = np.clip(x0, 1, W0 - 2)

        im0 = np.empty_like(t0, dtype=float)
        for fr in np.unique(t0):
            sel = (t0 == fr)
            frame = ch0[fr]
            im0[sel] = patch3x3_mean(frame, y0[sel], x0[sel]) / ch0_meds[fr]

        df.loc[m0, 'im_int_0'] = im0

    # --- channel 1 ---
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
    all_results = []
    all_timepoints = []

    condition = folder.split("\\")[5]

    # only analyze these colocIDs
    coloc_ids = subdf[coloc_col].unique()
    # print(coloc_ids)
    tracks_f = tracks[tracks[coloc_col].isin(coloc_ids)].copy()

    for track_id, g in tracks_f.groupby(coloc_col, sort=False):
        g = g.sort_values(t_col).reset_index(drop=True)

        # window in seconds
        overlap_frames = stats.loc[stats[coloc_col] == track_id, "overlap_t"].values
        if len(overlap_frames) == 0:
            continue
        coloc = overlap_frames[0] * dt
        start_sec, end_sec = coloc[0], coloc[-1]

        time = g[t_col].to_numpy(float)
        pre = time < start_sec
        during = (time >= start_sec) & (time <= end_sec)
        post = time > end_sec

        # --- build long per-timepoint df (CD19 + Zap70) from precomputed columns ---
        def particle_long(particle, suffix):
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

                # radial metrics (already computed)
                "dist_to_center": g[f"dist_to_center_{suffix}"].to_numpy(),
                "radial_speed_from_dist": g[f"radial_speed_from_dist_{suffix}"].to_numpy(),
                "angle_cos": g[f"angle_cos_{suffix}"].to_numpy(),
            })
            return out

        df_cd19 = particle_long("CD19", "0")
        df_zap70 = particle_long("Zap70", "1")
        full_metrics_df = pd.concat([df_cd19, df_zap70], ignore_index=True)
        all_timepoints.append(full_metrics_df)

        # --- summaries ---
        def summarize(particle, suffix, mask, timing_label):
            if mask.sum() == 0:
                return

            v = g.get(f"velocity_{suffix}", pd.Series(np.nan, index=g.index)).to_numpy()[mask]
            rs = g[f"radial_speed_from_dist_{suffix}"].to_numpy()[mask]
            ang = g[f"angle_cos_{suffix}"].to_numpy()[mask]
            inten = g.get(f"im_int_{suffix}", pd.Series(np.nan, index=g.index)).to_numpy()[mask]

            avg_speed = np.nanmean(v) if np.isfinite(v).any() else np.nan
            avg_radial = np.nanmean(rs) if np.isfinite(rs).any() else np.nan
            directionality = np.nanmean(ang) if np.isfinite(ang).any() else np.nan
            avg_intensity = np.nanmean(inten) if np.isfinite(inten).any() else np.nan
            median_intensity = np.nanmedian(inten) if np.isfinite(inten).any() else np.nan

            # slope of distance vs time (seconds)
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

        for timing, mask in [("pre", pre), ("during", during), ("post", post)]:
            summarize("CD19", "0", mask, timing)
            summarize("Zap70", "1", mask, timing)
    all_timepoints_df = pd.concat(all_timepoints, ignore_index=True)
    return all_results, all_timepoints_df

def safe_cos(dx, dy, rx, ry):
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


    #Compute velocities
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