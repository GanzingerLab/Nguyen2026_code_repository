#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from tqdm import tqdm
from matplotlib.lines import Line2D


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
vel_stats = pd.read_csv(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\all_cotracks_velocities&directionality_stats_correctedtime.csv')
check_folders = vel_stats['run'].unique()
velocities = pd.read_csv(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\all_cotracks_velocities&directionality_correctedtime.csv')
#%%
len_co = 5
before  = -2
len_ex = len_co
results = []

for p in tqdm(check_folders):
    path_a = os.path.join(p, "cluster_analysis_spots_filtered", "638nm_roi_locs_nm_trackpy_ColocsTracks_stats.hdf")
    path_b = os.path.join(p, "cluster_analysis_spots_filtered", "638nm_roi_locs_nm_trackpy_ColocsTracks.csv")

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
    b_ranges = (b2.groupby("colocID", as_index=False)["t_corr"].agg(tmin="min", tmax="max"))

    # keep only colocIDs that satisfy window condition
    valid = b_ranges[(b_ranges["tmin"] <= before) & (b_ranges["tmax"] >= len_ex - 1)][["colocID", "tmin", "tmax"]]

    if valid.empty:
        continue

    # subset velocities once for this run
    v_run = velocities[(velocities["run"] == p) & (velocities["particle"] == "CD19")].copy()

    if v_run.empty:
        continue

    # attach coloc metadata and valid time windows
    v2 = (v_run.merge(meta, on="colocID", how="inner").merge(valid, on="colocID", how="inner"))

    v2["t_corr"] = v2["t"] - v2["start_coloc"]

    # final time filtering
    velocities_filt = v2[v2["t_corr"].between(v2["tmin"], v2["end_coloc_corr"])].copy()

    if not velocities_filt.empty:
        results.append(velocities_filt)

results_df = pd.concat(results, ignore_index=True)
category_map = (vel_stats[["colocID", "run", "category"]].drop_duplicates(subset=["colocID", "run"]))
results_df2 = results_df.merge(category_map, on=["colocID", "run"], how="left")
results_df2.to_csv(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\decrease_analysis_Zap70_velocities_Full_tracks_cotrackslonger5frames.csv', index=False)
#%%
results_df2= pd.read_csv(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\decrease_analysis_Zap70_velocities_Full_tracks_cotrackslonger5frames.csv')
results_df1 = pd.read_csv(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\decrease_analysis_CD19_velocities_Full_tracks_cotrackslonger5frames.csv')
results_df2['dilution'] = results_df2['run'].str.extract(r'(\d+xdilutedCD19)')
results_df2['dilution_group'] = results_df2['dilution'].map(DIL_GROUP_MAP)
results_df1['dilution'] = results_df1['run'].str.extract(r'(\d+xdilutedCD19)')
results_df1['dilution_group'] = results_df1['dilution'].map(DIL_GROUP_MAP)
#%%For separate CD19 and ZAP70 metrics
PROTEIN = 'CD19' #ZAP70 or CD19
METRIC = 'velocity' # velocity or intensity
CD19_DENSITY = 'sparse' # sparse, intermediate, or dense
MATURATION = 1 # 1 for mature, 0 for disorganized
mat_map = {1: 'mature', 0: 'disorganized'}
if PROTEIN == 'ZAP70':
    results = results_df2.copy()
elif PROTEIN == 'CD19':
    results = results_df1.copy()
else: 
    raise ValueError("Invalid PROTEIN value. Must be 'ZAP70' or 'CD19'.")

if METRIC == 'velocity':
    ylim = (0, 160)
if METRIC == 'intensity':
    ylim = (0, 10)

tmp = results_df2[(results_df2["category"] == MATURATION)].copy()
tmp = tmp[(tmp["dilution_group"] == CD19_DENSITY)].copy()

avg = (
    tmp.groupby(["condition", "t_corr"])[METRIC]
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
n_tracks = (tmp[["colocID", "run", "condition"]].drop_duplicates().groupby("condition").size())

for cond, df_cond in avg.groupby("condition"):
    n = n_tracks.loc[cond]
    plt.plot(df_cond["t"], df_cond["mean"], label=f"{cond} (n={n})")
    plt.fill_between(df_cond["t"], df_cond["mean"] - df_cond["sem"], df_cond["mean"] + df_cond["sem"], alpha=0.3)

plt.legend()
plt.axvline(0, linestyle="--", color="k", alpha=0.5)
plt.xlabel("t")
plt.xlim(-120, 120)
plt.ylim(ylim)
plt.ylabel(PROTEIN+' '+METRIC)
plt.savefig(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260428_average_traces\{PROTEIN}_{METRIC}_{CD19_DENSITY}_{mat_map[MATURATION]}.pdf', dpi=600)
plt.savefig(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260428_average_traces\{PROTEIN}_{METRIC}_{CD19_DENSITY}_{mat_map[MATURATION]}.png', dpi=600)
plt.show()

#%%#FOR RATIO ZAP70/CD19
METRIC = 'intensity' # velocity or intensity
CD19_DENSITY = 'intermediate' # sparse, intermediate, or dense
MATURATION = 1 # 1 for mature, 0 for disorganized
mat_map = {1: 'mature', 0: 'disorganized'}

if METRIC == 'velocity':
    ylim = (0, 160)
if METRIC == 'intensity':
    ylim = (0, 5)
################################################################
#FOR RATIO ZAP70/CD19
results_df2['ratio'] = results_df2[METRIC]/results_df1[METRIC] 
################################################################

tmp = results_df2[(results_df2["category"] == MATURATION) & results_df2["ratio"].notna() & np.isfinite(results_df2["ratio"]) & (results_df2["ratio"] > 0)].copy()
tmp = tmp[(tmp["dilution_group"] == CD19_DENSITY)].copy()

conditions = sorted(tmp["condition"].unique())

palette = plt.cm.tab10.colors

condition_colors = {cond: palette[i % len(palette)] for i, cond in enumerate(conditions)}

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
ax.set_ylabel(f"{METRIC} ratio (Zap70/CD19)")
ax.set_xlim(-120, 120)
ax.set_ylim(ylim)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    handles=legend_handles,
    frameon=False,
    fontsize=8,
    loc="upper right"
)

plt.tight_layout()

# plt.savefig(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260428_average_traces\ZAP70overCD19_{METRIC}_{CD19_DENSITY}_{mat_map[MATURATION]}.pdf', dpi=600)
# plt.savefig(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260428_average_traces\ZAP70overCD19_{METRIC}_{CD19_DENSITY}_{mat_map[MATURATION]}.png', dpi=600)
plt.show()