#%%
import os
from matplotlib.lines import Line2D

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

MATURATION_MAP = {
    0: "never-mature",
    1: "mature",
    2: "started_mature"}
#%%
vel_stats = pd.read_csv(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\all_cotracks_velocities&directionality_stats_correctedtime.csv'
)
check_folders = vel_stats['run'].unique()
velocities = pd.read_csv(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\all_cotracks_velocities&directionality_correctedtime.csv'
)
#%% Run twice, changing the particle variable to "CD19" and "ZAP70" to get values from both.  
particle = "CD19"
# particle = "ZAP70"
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
        (velocities["particle"] == particle)
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

results_df = pd.concat(results, ignore_index=True)
category_map = (
    vel_stats[["colocID", "run", "category"]]
    .drop_duplicates(subset=["colocID", "run"])
)

results_df2 = results_df.merge(
    category_map,
    on=["colocID", "run"],
    how="left")

if particle == "CD19":
    results_df2.to_csv(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\decrease_analysis_CD19_velocities_Full_tracks_cotrackslonger5frames.csv', index=False)
if particle == "ZAP70":
    results_df2.to_csv(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\decrease_analysis_ZAP70_velocities_Full_tracks_cotrackslonger5frames.csv', index=False)
#%% Load data once prevuious code has run for each particles 
results_df2= pd.read_csv(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\decrease_analysis_Zap70_velocities_Full_tracks_cotrackslonger5frames.csv')
results_df1 = pd.read_csv(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\decrease_analysis_CD19_velocities_Full_tracks_cotrackslonger5frames.csv')

#%%Adding dilution group to the dataframes
results_df2['dilution'] = results_df2['run'].str.extract(r'(\d+xdilutedCD19)')
results_df2['dilution_group'] = results_df2['dilution'].map(DIL_GROUP_MAP)
results_df1['dilution'] = results_df1['run'].str.extract(r'(\d+xdilutedCD19)')
results_df1['dilution_group'] = results_df1['dilution'].map(DIL_GROUP_MAP)
#%% CD19 velocity
maturation_category = 1


tmp = results_df1[
    (results_df1["category"] == 1)
    & results_df1["dilution_group"].notna()
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
        tmp_dil.groupby(["condition", "t_corr"])["velocity"]
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

    ax.set_ylim(0, 180)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=8,
        loc="upper right"
    )

axes[0].set_ylabel("intensity Zap70")

plt.tight_layout()

plt.savefig(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260507_separate_densities\CD19_trace_average_velocity_by_density_{MATURATION_MAP[maturation_category]}.pdf', dpi=600)
plt.savefig(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260507_separate_densities\CD19_trace_average_velocity_by_density_{MATURATION_MAP[maturation_category]}.png', dpi=600)

plt.show()
#%% CD19 intensity
maturation_category = 1


tmp = results_df1[
    (results_df1["category"] == 1)
    & results_df1["dilution_group"].notna()
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
        tmp_dil.groupby(["condition", "t_corr"])["intensity"]
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

axes[0].set_ylabel("intensity Zap70")

plt.tight_layout()

plt.savefig(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260507_separate_densities\CD19_trace_average_intensity_by_density_{MATURATION_MAP[maturation_category]}.pdf', dpi=600)
plt.savefig(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260507_separate_densities\CD19_trace_average_intensity_by_density_{MATURATION_MAP[maturation_category]}.png', dpi=600)

plt.show()
#%% ZAP70 intensity
maturation_category = 1


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
        tmp_dil.groupby(["condition", "t_corr"])["intensity"]
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

    ax.set_ylim(0, 8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=8,
        loc="upper right"
    )

axes[0].set_ylabel("intensity Zap70")

plt.tight_layout()

plt.savefig(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260507_separate_densities\Zap70_trace_average_intensity_by_density_{MATURATION_MAP[maturation_category]}.pdf', dpi=600)
plt.savefig(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260507_separate_densities\Zap70_trace_average_intensity_by_density_{MATURATION_MAP[maturation_category]}.png', dpi=600)

plt.show()
#%% ZAP70/CD19 intensity ratio
maturation_category = 1

results_df2['ratio'] = results_df2['intensity']/results_df1['intensity'] 

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

plt.savefig(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260507_separate_densities\Zap70overCD19_trace_average_intensity_by_density_{MATURATION_MAP[maturation_category]}.pdf', dpi=600)
plt.savefig(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260507_separate_densities\Zap70overCD19_trace_average_intensity_by_density_{MATURATION_MAP[maturation_category]}.png', dpi=600)

plt.show()