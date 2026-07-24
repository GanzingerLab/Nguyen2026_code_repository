#%%
import os

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.lines import Line2D
from postSPIT import tirf_analysis as ta
from tqdm import tqdm
from scipy.signal import find_peaks
from matplotlib.lines import Line2D


#%%

DIL_GROUP_MAP = {
    "50xdilutedCD19": "dense",
    "100xdilutedCD19": "dense",
    "500xdilutedCD19": "intermediate",
    "1000xdilutedCD19": "intermediate",
    "1500xdilutedCD19": "intermediate", 
    "3000xdilutedCD19": "sparse",
    "6000xdilutedCD19": "sparse",
}
#%%
vel_stats = pd.read_csv(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\all_cotracks_velocities&directionality_stats_correctedtime.csv'
)
check_folders = vel_stats['run'].unique()
velocities = pd.read_csv(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\all_cotracks_velocities&directionality_correctedtime.csv'
)
#%%
len_co = 5
before  = -2
len_ex = len_co
results = []

for p in tqdm(check_folders):
    path_a = os.path.join(
        p, "cluster_analysis_spots_filtered",
        "638nm_roi_locs_nm_trackpy_ColocsTracks_stats.hdf"
    )
    path_b = os.path.join(
        p, "cluster_analysis_spots_filtered",
        "638nm_roi_locs_nm_trackpy_ColocsTracks.csv"
    )

    a = pd.read_hdf(path_a)
    b = pd.read_csv(path_b)

    # keep only relevant coloc events
    a_fil = a[a["num_frames_coloc"] >= len_co].copy()
    if a_fil.empty:
        continue

    # extract start/end from overlap_t once
    meta = a_fil[["colocID", "overlap_t"]].copy()
    meta["start_coloc"] = meta["overlap_t"].str[0]
    meta["end_coloc"] = meta["overlap_t"].str[-1]
    meta["end_coloc_corr"] = meta["end_coloc"] - meta["start_coloc"]
    meta = meta[["colocID", "start_coloc", "end_coloc_corr"]]

    # add timing info to b
    b2 = b.merge(meta, on="colocID", how="inner")
    b2["t_corr"] = b2["t"] - b2["start_coloc"]

    # per-coloc min/max t_corr
    b_ranges = (
        b2.groupby("colocID", as_index=False)["t_corr"]
          .agg(tmin="min", tmax="max")
    )

    # keep only colocIDs that satisfy window condition
    valid = b_ranges[
        (b_ranges["tmin"] <= before) &
        (b_ranges["tmax"] >= len_ex - 1)
    ][["colocID", "tmin", "tmax"]]

    if valid.empty:
        continue

    # subset velocities once for this run
    v_run = velocities[
        (velocities["run"] == p) &
        (velocities["particle"] == "CD19")
    ].copy()

    if v_run.empty:
        continue

    # attach coloc metadata and valid time windows
    v2 = (
        v_run.merge(meta, on="colocID", how="inner")
             .merge(valid, on="colocID", how="inner")
    )

    v2["t_corr"] = v2["t"] - v2["start_coloc"]

    # final time filtering
    velocities_filt = v2[
        v2["t_corr"].between(v2["tmin"], v2["end_coloc_corr"])
    ].copy()

    if not velocities_filt.empty:
        results.append(velocities_filt)


#%%
results_df = pd.concat(results, ignore_index=True)
category_map = (
    vel_stats[["colocID", "run", "category"]]
    .drop_duplicates(subset=["colocID", "run"])
)

results_df2 = results_df.merge(
    category_map,
    on=["colocID", "run"],
    how="left")
#%%
results_df2.to_csv(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\decrease_analysis_Zap70_velocities_Full_tracks_cotrackslonger5frames.csv', index=False)
#%%
results_df2= pd.read_csv(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\decrease_analysis_Zap70_velocities_Full_tracks_cotrackslonger5frames.csv')
results_df1 = pd.read_csv(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\decrease_analysis_CD19_velocities_Full_tracks_cotrackslonger5frames.csv')

#%%

results_df2['dilution'] = results_df2['run'].str.extract(r'(\d+xdilutedCD19)')
results_df2['dilution_group'] = results_df2['dilution'].map(DIL_GROUP_MAP)
results_df1['dilution'] = results_df1['run'].str.extract(r'(\d+xdilutedCD19)')
results_df1['dilution_group'] = results_df1['dilution'].map(DIL_GROUP_MAP)
#%%

tmp = results_df2[(results_df2["category"] == 1)].copy()
tmp = tmp[(tmp["dilution_group"] == "dense")].copy()

avg = (
    tmp.groupby(["condition", "t_corr"])["velocity"]
    .agg(
        log_mean=lambda x: np.log(x).mean(),
        log_sem=lambda x: np.log(x).std() / np.sqrt(len(x))
    )
    .assign(
        mean=lambda x: np.exp(x["log_mean"]),
        sem=lambda x: np.exp(x["log_mean"]) * x["log_sem"]
    )
    .reset_index()
    .sort_values(["condition", "t_corr"])
)
avg["t"] = avg["t_corr"]*2
n_tracks = (
    tmp[["colocID", "run", "condition"]]
    .drop_duplicates()
    .groupby("condition")
    .size()
)


for cond, df_cond in avg.groupby("condition"):
    n = n_tracks.loc[cond]
    # norm = df_cond["mean"][df_cond['t']==-2].values[0]
    plt.plot(df_cond["t"], df_cond["mean"], label=f"{cond} (n={n})")
    plt.fill_between(
        df_cond["t"],
        df_cond["mean"] - df_cond["sem"],
        df_cond["mean"] + df_cond["sem"],
        alpha=0.3
    )

plt.legend()
plt.axvline(0, linestyle="--", color="k", alpha=0.5)
plt.xlabel("t")
plt.xlim(-120, 120)
plt.ylim(0, 160)
plt.ylabel("intensity")
# plt.savefig(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260428_average_traces\Zap70_intensity_never-mature.pdf', dpi=600)
# plt.savefig(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260428_average_traces\Zap70_intensity_never-mature.png', dpi=600)
plt.show()

#%% find peaks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

tmp_boot = tmp.copy()
tmp_boot = tmp_boot[
    tmp_boot["velocity"].notna() &
    np.isfinite(tmp_boot["velocity"]) &
    (tmp_boot["velocity"] > 0)
].copy()

tmp_boot["t_corr"] = tmp_boot["t_corr"].round().astype(int)

track_cols = ["colocID", "run"]

#Build observed average curve
avg = (
    tmp_boot.groupby(["condition", "t_corr"])["velocity"]
    .agg(
        log_mean=lambda x: np.log(x).mean(),
        log_sem=lambda x: np.log(x).std() / np.sqrt(len(x))
    )
    .assign(
        mean=lambda x: np.exp(x["log_mean"]),
        sem=lambda x: np.exp(x["log_mean"]) * x["log_sem"]
    )
    .reset_index()
    .sort_values(["condition", "t_corr"])
)

avg["t"] = avg["t_corr"]

n_tracks = (
    tmp_boot[track_cols + ["condition"]]
    .drop_duplicates()
    .groupby("condition")
    .size()
)

# =========================================================
# 2. Peak metrics from one averaged curve
# =========================================================
def compute_peak_metrics(
    df_cond,
    peak_window=(-10, 5),
    baseline_before_window=(-45, -10),
    baseline_after_window=(10, 40),
    post_drop_window=(10, 50),
    prominence=None
):
    """
    df_cond must contain columns: t, mean
    Returns peak-centered metrics on the non-normalized averaged curve.
    """
    df_cond = df_cond.sort_values("t").copy()

    t = df_cond["t"].to_numpy()
    y = df_cond["mean"].to_numpy()

    out = {
        "peak_t": np.nan,
        "peak_v": np.nan,
        "baseline_before": np.nan,
        "baseline_after": np.nan,
        "rise": np.nan,
        "drop": np.nan,
        "rel_rise": np.nan,
        "rel_drop_from_peak": np.nan,
        "pre_slope": np.nan,
        "post_slope": np.nan,
        "post_drop_mean": np.nan,
    }

    if len(df_cond) < 3:
        return out

    # -------------------------
    # peak near t = 0
    # -------------------------
    peak_mask = (t >= peak_window[0]) & (t <= peak_window[1])
    if not np.any(peak_mask):
        return out

    t_peak_region = t[peak_mask]
    y_peak_region = y[peak_mask]

    # optional find_peaks inside the window
    peaks, props = find_peaks(
        y_peak_region,
        prominence=prominence if prominence is not None else 0
    )

    if len(peaks) == 0:
        # fallback: just take max in the window
        local_idx = np.argmax(y_peak_region)
    else:
        # choose peak closest to t = 0
        local_idx = peaks[np.argmin(np.abs(t_peak_region[peaks]))]

    peak_t = t_peak_region[local_idx]
    peak_v = y_peak_region[local_idx]

    out["peak_t"] = peak_t
    out["peak_v"] = peak_v

    # -------------------------
    # baselines
    # -------------------------
    before_mask = (t >= baseline_before_window[0]) & (t <= baseline_before_window[1])
    after_mask = (t >= baseline_after_window[0]) & (t <= baseline_after_window[1])

    if np.any(before_mask):
        baseline_before = np.mean(y[before_mask])
        out["baseline_before"] = baseline_before

    if np.any(after_mask):
        baseline_after = np.mean(y[after_mask])
        out["baseline_after"] = baseline_after

    if np.isfinite(out["baseline_before"]):
        out["rise"] = peak_v - out["baseline_before"]
        out["rel_rise"] = out["rise"] / out["baseline_before"] if out["baseline_before"] != 0 else np.nan

    if np.isfinite(out["baseline_after"]):
        out["drop"] = peak_v - out["baseline_after"]
        out["rel_drop_from_peak"] = out["drop"] / peak_v if peak_v != 0 else np.nan

    # -------------------------
    # slopes
    # -------------------------
    pre_mask = (t >= -30) & (t <= -5)
    post_mask = (t >= 0) & (t <= 20)

    if np.sum(pre_mask) >= 2:
        out["pre_slope"] = np.polyfit(t[pre_mask], y[pre_mask], 1)[0]

    if np.sum(post_mask) >= 2:
        out["post_slope"] = np.polyfit(t[post_mask], y[post_mask], 1)[0]

    # -------------------------
    # mean speed after drop
    # -------------------------
    pd_mask = (t >= post_drop_window[0]) & (t <= post_drop_window[1])
    if np.any(pd_mask):
        out["post_drop_mean"] = np.mean(y[pd_mask])

    return out


# =========================================================
# 3. Observed metrics from the real averaged curves
# =========================================================
observed_metrics = []

for cond, df_cond in avg.groupby("condition"):
    metrics = compute_peak_metrics(df_cond)
    metrics["condition"] = cond
    observed_metrics.append(metrics)

observed_metrics = pd.DataFrame(observed_metrics).set_index("condition")
print("Observed metrics:")
print(observed_metrics)

# =========================================================
# 4. Bootstrap full tracks and recompute peak metrics
# =========================================================
def bootstrap_peak_metrics(
    df,
    n_boot=500,
    random_state=42,
    peak_window=(-10, 5),
    baseline_before_window=(-45, -10),
    baseline_after_window=(10, 40),
    post_drop_window=(10, 50),
    prominence=None
):
    rng = np.random.default_rng(random_state)
    results = []

    for cond in df["condition"].dropna().unique():
        df_cond = df[df["condition"] == cond].copy()

        tracks = df_cond[track_cols].drop_duplicates().reset_index(drop=True)
        n_tracks = len(tracks)

        if n_tracks == 0:
            continue

        for b in range(n_boot):
            sample_idx = rng.integers(0, n_tracks, size=n_tracks)
            sampled = tracks.iloc[sample_idx].copy().reset_index(drop=True)
            sampled["boot_track_id"] = np.arange(len(sampled))

            df_sample = sampled.merge(df_cond, on=track_cols, how="left")

            avg_sample = (
                df_sample.groupby(["condition", "boot_track_id", "t_corr"])["velocity"]
                .first()
                .reset_index()
                .groupby(["condition", "t_corr"])["velocity"]
                .agg(
                    log_mean=lambda x: np.log(x).mean(),
                    log_sem=lambda x: np.log(x).std() / np.sqrt(len(x))
                )
                .assign(
                    mean=lambda x: np.exp(x["log_mean"]),
                    sem=lambda x: np.exp(x["log_mean"]) * x["log_sem"]
                )
                .reset_index()
                .sort_values(["condition", "t_corr"])
            )

            avg_sample["t"] = avg_sample["t_corr"]

            metrics = compute_peak_metrics(
                avg_sample,
                peak_window=peak_window,
                baseline_before_window=baseline_before_window,
                baseline_after_window=baseline_after_window,
                post_drop_window=post_drop_window,
                prominence=prominence
            )
            metrics["condition"] = cond
            metrics["bootstrap"] = b
            results.append(metrics)

    return pd.DataFrame(results)


boot_metrics = bootstrap_peak_metrics(
    tmp_boot,
    n_boot=500,
    random_state=42,
    peak_window=(-10, 5),
    baseline_before_window=(-45, -10),
    baseline_after_window=(10, 40),
    post_drop_window=(10, 50),
    prominence=5,   # tune if needed
)

# =========================================================
# 5. Summarize bootstrap metrics
# =========================================================
metric_names = [
    "peak_t",
    "peak_v",
    "baseline_before",
    "baseline_after",
    "rise",
    "drop",
    "rel_rise",
    "rel_drop_from_peak",
    "pre_slope",
    "post_slope",
    "post_drop_mean",
]

summary_tables = {}

for metric in metric_names:
    summary = (
        boot_metrics.groupby("condition")[metric]
        .agg(
            mean=lambda x: np.nanmean(x),
            median=lambda x: np.nanmedian(x),
            ci_low=lambda x: np.nanpercentile(x, 2.5),
            ci_high=lambda x: np.nanpercentile(x, 97.5),
        )
    )
    summary["observed"] = summary.index.map(observed_metrics[metric])
    summary_tables[metric] = summary

print("\nBootstrap summary for peak_v:")
print(summary_tables["peak_v"])

print("\nBootstrap summary for drop:")
print(summary_tables["drop"])

print("\nBootstrap summary for post_drop_mean:")
print(summary_tables["post_drop_mean"])

# =========================================================
# 6. Compare two conditions for one metric
# =========================================================
def compare_conditions(boot_df, cond1, cond2, metric):
    a = boot_df.loc[boot_df["condition"] == cond1, metric].dropna().to_numpy()
    b = boot_df.loc[boot_df["condition"] == cond2, metric].dropna().to_numpy()

    diff = np.subtract.outer(a, b).ravel()

    p_two_sided = 2 * min(np.mean(diff >= 0), np.mean(diff <= 0))
    p_cond1_greater = np.mean(diff <= 0)
    p_cond2_greater = np.mean(diff >= 0)

    ci_diff = np.percentile(diff, [2.5, 50, 97.5])

    return {
        "metric": metric,
        "cond1": cond1,
        "cond2": cond2,
        "p_two_sided": p_two_sided,
        "p_cond1_greater": p_cond1_greater,
        "p_cond2_greater": p_cond2_greater,
        "diff_ci_low": ci_diff[0],
        "diff_median": ci_diff[1],
        "diff_ci_high": ci_diff[2],
    }

cond1 = "CART3 FMC63 High aff Low exp"
cond2 = "CART4 CAT Low aff Low exp"

print("\nComparison on peak_v:")
print(compare_conditions(boot_metrics, cond1, cond2, "peak_v"))

print("\nComparison on drop:")
print(compare_conditions(boot_metrics, cond1, cond2, "drop"))

print("\nComparison on post_drop_mean:")
print(compare_conditions(boot_metrics, cond1, cond2, "post_drop_mean"))

# =========================================================
# 7. Plot observed curves with peak and baselines
# =========================================================
order = [
    "CART3 FMC63 High aff High exp",
    "CART3 FMC63 High aff Low exp",
    "CART4 CAT Low aff High exp",
    "CART4 CAT Low aff Low exp",
]

plt.figure(figsize=(8, 5))

for cond, df_cond in avg.groupby("condition"):
    if cond not in order:
        continue

    n = n_tracks.loc[cond]
    df_cond = df_cond.sort_values("t")

    line, = plt.plot(df_cond["t"], df_cond["mean"], label=f"{cond} (n={n})")
    color = line.get_color()

    plt.fill_between(
        df_cond["t"],
        df_cond["mean"] - df_cond["sem"],
        df_cond["mean"] + df_cond["sem"],
        alpha=0.25,
        color=color
    )

    m = observed_metrics.loc[cond]

    if np.isfinite(m["peak_t"]) and np.isfinite(m["peak_v"]):
        plt.scatter([m["peak_t"]], [m["peak_v"]], color=color, s=40, zorder=5)

    if np.isfinite(m["baseline_before"]):
        plt.hlines(
            m["baseline_before"], xmin=-45, xmax=-10,
            color=color, linestyle=":", linewidth=2
        )

    if np.isfinite(m["baseline_after"]):
        plt.hlines(
            m["baseline_after"], xmin=10, xmax=40,
            color=color, linestyle="--", linewidth=2
        )

plt.axvline(0, color="k", linestyle="--", alpha=0.5)
plt.xlabel("t")
plt.ylabel("velocity")
plt.xlim(-50, 50)
plt.ylim(0, 160)
plt.legend()
plt.tight_layout()
plt.show()
#%%
# =========================================================
# 8. Dot + CI plot for one chosen metric
# =========================================================
metric_to_plot = "post_slope"   # change to "peak_v", "rise", "post_drop_mean", etc.

plot_summary = summary_tables[metric_to_plot].reset_index()

plt.figure(figsize=(7, 4))
x = np.arange(len(plot_summary))
y = plot_summary["mean"].to_numpy()
yerr_lower = y - plot_summary["ci_low"].to_numpy()
yerr_upper = plot_summary["ci_high"].to_numpy() - y

plt.errorbar(
    x, y,
    yerr=[yerr_lower, yerr_upper],
    fmt="o",
    capsize=5
)

plt.xticks(x, plot_summary["condition"], rotation=45, ha="right")
plt.ylabel(metric_to_plot)
plt.tight_layout()
plt.show()

#%% bootstrap analysis to compare conditions



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Assumptions:
# - tmp already contains the extracted/aligned data
# - tmp has at least these columns:
#   ["condition", "colocID", "run", "t_corr", "velocity"]
# - avg is your observed curve built from the same tmp
# --------------------------------------------------

# ----------------------------
# 0. Keep only valid rows for geometric mean
# ----------------------------
tmp_boot = tmp.copy()
tmp_boot = tmp_boot[
    tmp_boot["velocity"].notna() &
    (tmp_boot["velocity"] > 0)
].copy()

tmp_boot["t_corr"] = tmp_boot["t_corr"].round().astype(int)

# ----------------------------
# 1. Build observed average curve
# ----------------------------
avg = (
    tmp_boot.groupby(["condition", "t_corr"])["velocity"]
    .agg(
        log_mean=lambda x: np.log(x).mean(),
        log_sem=lambda x: np.log(x).std() / np.sqrt(len(x))
    )
    .assign(
        mean=lambda x: np.exp(x["log_mean"]),
        sem=lambda x: np.exp(x["log_mean"]) * x["log_sem"]
    )
    .reset_index()
    .sort_values(["condition", "t_corr"])
)

avg["t"] = avg["t_corr"]

# optional: number of tracks per condition
n_tracks = (
    tmp_boot[["colocID", "run", "condition"]]
    .drop_duplicates()
    .groupby("condition")
    .size()
)

# ----------------------------
# 2. Metric to compare conditions
# ----------------------------
def post_drop_mean(df_cond, t_min=5, t_max=30):
    df_window = df_cond[(df_cond["t"] >= t_min) & (df_cond["t"] <= t_max)]
    if len(df_window) == 0:
        return np.nan
    return df_window["mean"].mean()

# ----------------------------
# 3. Bootstrap by FULL TRACKS
# ----------------------------
def bootstrap_post_drop(df, n_boot=1000, t_min=5, t_max=30, random_state=42):
    rng = np.random.default_rng(random_state)
    results = []

    for cond in tqdm(df["condition"].dropna().unique()):
        df_cond = df[df["condition"] == cond].copy()

        # one row per full track
        tracks = df_cond[["colocID", "run"]].drop_duplicates().reset_index(drop=True)
        n_tracks = len(tracks)

        if n_tracks == 0:
            continue

        for b in tqdm(range(n_boot)):
            # resample tracks WITH replacement
            sample_idx = rng.integers(0, n_tracks, size=n_tracks)
            sampled = tracks.iloc[sample_idx].copy().reset_index(drop=True)

            # unique bootstrap id so repeated tracks stay repeated
            sampled["boot_track_id"] = np.arange(len(sampled))

            # attach full track data back
            df_sample = sampled.merge(df_cond, on=["colocID", "run"], how="left")

            # average per bootstrap sample at each aligned time
            avg_sample = (
                df_sample.groupby(["condition", "boot_track_id", "t_corr"])["velocity"]
                .first()
                .reset_index()
                .groupby(["condition", "t_corr"])["velocity"]
                .agg(
                    log_mean=lambda x: np.log(x).mean(),
                    log_sem=lambda x: np.log(x).std() / np.sqrt(len(x))
                )
                .assign(
                    mean=lambda x: np.exp(x["log_mean"]),
                    sem=lambda x: np.exp(x["log_mean"]) * x["log_sem"]
                )
                .reset_index()
                .sort_values(["condition", "t_corr"])
            )

            avg_sample["t"] = avg_sample["t_corr"]

            val = post_drop_mean(avg_sample, t_min=t_min, t_max=t_max)

            results.append({
                "condition": cond,
                "bootstrap": b,
                "metric": val
            })

    return pd.DataFrame(results)

# observed metric from the real average curves
observed_metric = {}
for cond, df_cond in avg.groupby("condition"):
    observed_metric[cond] = post_drop_mean(df_cond, t_min=5, t_max=30)

observed_metric = pd.Series(observed_metric, name="observed_metric")
print("Observed metric:")
print(observed_metric)

boot_metric = bootstrap_post_drop(tmp_boot, n_boot=500, t_min=5, t_max=30)

# ----------------------------
# 4. Summarize bootstrap results
# ----------------------------
summary_metric = (
    boot_metric.groupby("condition")["metric"]
    .agg(
        mean=lambda x: np.nanmean(x),
        median=lambda x: np.nanmedian(x),
        ci_low=lambda x: np.nanpercentile(x, 2.5),
        ci_high=lambda x: np.nanpercentile(x, 97.5),
    )
)

summary_metric["observed"] = summary_metric.index.map(observed_metric)

print("\nBootstrap summary:")
print(summary_metric)

# ----------------------------
# 5. Compare two conditions
# ----------------------------
cond1 = "CART3 FMC63 High aff Low exp"
cond2 = "CART4 CAT Low aff Low exp"

a = boot_metric.loc[boot_metric["condition"] == cond1, "metric"].dropna().to_numpy()
b = boot_metric.loc[boot_metric["condition"] == cond2, "metric"].dropna().to_numpy()

# all pairwise differences (more robust than a[:m] - b[:m])
diff = np.subtract.outer(a, b).ravel()

# two-sided bootstrap p-value
p_two_sided = 2 * min(np.mean(diff >= 0), np.mean(diff <= 0))

# one-sided p-value for: cond1 > cond2
p_cond1_greater = np.mean(diff <= 0)

# one-sided p-value for: cond2 > cond1
p_cond2_greater = np.mean(diff >= 0)

print("\nCondition comparison:")
print(f"{cond1} vs {cond2}")
print("two-sided p =", p_two_sided)
print(f"one-sided p ({cond1} > {cond2}) =", p_cond1_greater)
print(f"one-sided p ({cond2} > {cond1}) =", p_cond2_greater)

# ----------------------------
# 6. Optional: plot observed curves
# ----------------------------
plt.figure(figsize=(7, 5))

for cond, df_cond in avg.groupby("condition"):
    n = n_tracks.loc[cond]
    plt.plot(df_cond["t"] * 2, df_cond["mean"], label=f"{cond} (n={n})")
    plt.fill_between(
        df_cond["t"] * 2,
        df_cond["mean"] - df_cond["sem"],
        df_cond["mean"] + df_cond["sem"],
        alpha=0.25
    )

plt.axvline(0, linestyle="--", color="k", alpha=0.5)
plt.xlabel("t")
plt.ylabel("intensity ratio (Zap70/CD19)")
plt.xlim(-100, 100)
plt.ylim(0, 150)
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# 7. Optional: dot + CI plot for the metric
# ----------------------------
plot_summary = summary_metric.reset_index().copy()

plt.figure(figsize=(6, 4))
x = np.arange(len(plot_summary))
y = plot_summary["mean"].to_numpy()
yerr_lower = y - plot_summary["ci_low"].to_numpy()
yerr_upper = plot_summary["ci_high"].to_numpy() - y

plt.errorbar(
    x, y,
    yerr=[yerr_lower, yerr_upper],
    fmt="o",
    capsize=5
)

plt.xticks(x, plot_summary["condition"], rotation=45, ha="right")
plt.ylabel("Post-drop mean velocity")
plt.tight_layout()
plt.show()

#%%
from matplotlib.lines import Line2D

################################################################
#FOR RATIO ZAP70/CD19
results_df2['ratio'] = results_df2['intensity']/results_df1['intensity'] 
################################################################

tmp = results_df2[
    (results_df2["category"] == 1)
    & results_df2["dilution_group"].notna()
].copy()

dil_order = ["sparse", "intermediate", "dense"]

dil_order = [
    d for d in dil_order
    if d in tmp["dilution_group"].unique()
]

conditions = sorted(tmp["condition"].unique())

palette = plt.cm.tab10.colors

condition_colors = {
    cond: palette[i % len(palette)]
    for i, cond in enumerate(conditions)
}

fig, axes = plt.subplots(
    1,
    len(dil_order),
    figsize=(5.2 * len(dil_order), 3.8),
    sharey=True
)

if len(dil_order) == 1:
    axes = [axes]

for ax, dil in zip(axes, dil_order):

    tmp_dil = tmp[tmp["dilution_group"] == dil].copy()

    avg = (
        tmp_dil.groupby(["condition", "t_corr"])["ratio"]
        .agg(
            log_mean=lambda x: np.log(x).mean(),
            log_sem=lambda x: np.log(x).std() / np.sqrt(len(x))
        )
        .assign(
            mean=lambda x: np.exp(x["log_mean"]),
            sem=lambda x: np.exp(x["log_mean"]) * x["log_sem"]
        )
        .reset_index()
        .sort_values(["condition", "t_corr"])
    )

    avg["t"] = avg["t_corr"]

    n_tracks = (
        tmp_dil[["colocID", "run", "condition"]]
        .drop_duplicates()
        .groupby("condition")
        .size()
    )

    legend_handles = []

    for cond, df_cond in avg.groupby("condition"):

        n = n_tracks.loc[cond]

        color = condition_colors[cond]

        ax.plot(
            df_cond["t"] * 2,
            df_cond["mean"],
            color=color,
            linewidth=2
        )

        ax.fill_between(
            df_cond["t"] * 2,
            df_cond["mean"] - df_cond["sem"],
            df_cond["mean"] + df_cond["sem"],
            color=color,
            alpha=0.25
        )

        legend_handles.append(
            Line2D(
                [0], [0],
                color=color,
                linewidth=2,
                label=f"{cond} (n={n})"
            )
        )

    ax.axvline(0, linestyle="--", color="k", alpha=0.5)

    ax.set_title(f"dilution_group: {dil}")

    ax.set_xlabel("t")

    ax.set_xlim(-120, 120)

    ax.set_ylim(0, 4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=8,
        loc="upper right"
    )

axes[0].set_ylabel("intensity ratio (Zap70/CD19)")

plt.tight_layout()

# plt.savefig(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260507_separate_densities\Zap70overCD19_trace_average_intensity_by_density_matures.pdf', dpi=600)
# plt.savefig(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260507_separate_densities\Zap70overCD19_trace_average_intensity_by_density_matures.png', dpi=600)

plt.show()

#%%
from matplotlib.lines import Line2D

results_df2["ratio"] = results_df2["intensity"] / results_df1["intensity"]

tmp = results_df2[
    (results_df2["category"] == 0)
    & results_df2["ratio"].notna()
    & np.isfinite(results_df2["ratio"])
    & (results_df2["ratio"] > 0)
].copy()

conditions = sorted(tmp["condition"].unique())

palette = plt.cm.tab10.colors

condition_colors = {
    cond: palette[i % len(palette)]
    for i, cond in enumerate(conditions)
}

avg = (
    tmp.groupby(["condition", "t_corr"])["ratio"]
    .agg(
        log_mean=lambda x: np.log(x).mean(),
        log_sem=lambda x: np.log(x).std() / np.sqrt(len(x))
    )
    .assign(
        mean=lambda x: np.exp(x["log_mean"]),
        sem=lambda x: np.exp(x["log_mean"]) * x["log_sem"]
    )
    .reset_index()
    .sort_values(["condition", "t_corr"])
)

avg["t"] = avg["t_corr"]

n_tracks = (
    tmp[["colocID", "run", "condition"]]
    .drop_duplicates()
    .groupby("condition")
    .size()
)

fig, ax = plt.subplots(figsize=(5.2, 3.8))

legend_handles = []

for cond, df_cond in avg.groupby("condition"):

    n = n_tracks.loc[cond]
    color = condition_colors[cond]

    ax.plot(
        df_cond["t"] * 2,
        df_cond["mean"],
        color=color,
        linewidth=2
    )

    ax.fill_between(
        df_cond["t"] * 2,
        df_cond["mean"] - df_cond["sem"],
        df_cond["mean"] + df_cond["sem"],
        color=color,
        alpha=0.25
    )

    legend_handles.append(
        Line2D(
            [0], [0],
            color=color,
            linewidth=2,
            label=f"{cond} (n={n})"
        )
    )

ax.axvline(0, linestyle="--", color="k", alpha=0.5)

ax.set_xlabel("t")
ax.set_ylabel("intensity ratio (Zap70/CD19)")
ax.set_xlim(-120, 120)
ax.set_ylim(0, 4)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    handles=legend_handles,
    frameon=False,
    fontsize=8,
    loc="upper right"
)

plt.tight_layout()

plt.savefig(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260428_average_traces\Zap70overCD19_trace_average_intensity_never-matures.pdf', dpi=600)
plt.savefig(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260428_average_traces\Zap70overCD19_trace_average_intensity_never-matures.png', dpi=600)

plt.show()