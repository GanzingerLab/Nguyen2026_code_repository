# %% Imports
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
# %% 
SPOT_BOX_SIZE = 3
SPOT_AREA = SPOT_BOX_SIZE ** 2
MATURE_CATEGORY = 1

ZAP_LABEL = "Zap70"
CD19_LABEL = "CD19"
ZAP_CLUSTER_FILE = "clusters_488nm.hdf"
CD19_CLUSTER_FILE = "clusters_638nm.hdf"

PATH = r'D:\Data\Chi_data\20250801_filtered\output'
# %% Functions
def sum_intensity_per_cell(spot_df, cluster_df):
    """
    Combine spot intensity (summed per frame per cell) and cluster norm_sum_int
    (summed per frame per cell), then compute total = spots + clusters.
    """

    # Spots: sum intensity columns per (t, cell_id); keep other cols as 'first'
    int_cols = [c for c in spot_df.columns if c.startswith("intensity")]
    other_cols = [c for c in spot_df.columns if c not in int_cols + ['t']]

    int_spots = (
        spot_df
        .groupby(['t', 'cell_id'], as_index=False)
        .agg({**{col: 'sum' for col in int_cols},
              **{col: 'first' for col in other_cols}})
    )

    # Clusters: sum integrated intensity columns per (frame, cell_id); keep other cols as 'first'
    int_cols = [c for c in cluster_df.columns if "sum_int" in c]
    other_cols = [c for c in cluster_df.columns if c not in int_cols + ['t']]

    int_clusters = (
        cluster_df
        .groupby(['frame', 'cell_id'], as_index=False)
        .agg({**{col: 'sum' for col in int_cols},
              **{col: 'first' for col in other_cols}})
    )

    # Spots per frame per cell
    spots_fc = (
        int_spots
        .groupby(['t', 'cell_id'], as_index=False)['intensity']
        .sum()
        .rename(columns={'t': 'frame', 'intensity': 'spots_intensity_sum'})
    )

    # Clusters per frame per cell
    clusters_fc = (
        int_clusters
        .groupby(['frame', 'cell_id'], as_index=False)['norm_sum_int']
        .sum()
        .rename(columns={'norm_sum_int': 'clusters_norm_sum_int'})
    )

    # Merge and fill missing with 0
    merged = spots_fc.merge(clusters_fc, on=['frame', 'cell_id'], how='outer')
    merged[['spots_intensity_sum', 'clusters_norm_sum_int']] = (
        merged[['spots_intensity_sum', 'clusters_norm_sum_int']].fillna(0)
    )

    # Total per frame+cell
    merged['spots_plus_clusters'] = (
        merged['spots_intensity_sum'] + merged['clusters_norm_sum_int']
    )

    return merged

def summarize(df):
    """Per-cell summary statistics, separately for mature vs not-mature frames."""
    return (
        df.groupby(['cell_id', 'mature'], as_index=False)
          .agg(
              total_mean=('spots_plus_clusters', 'mean'),
              total_max=('spots_plus_clusters', 'max'),
              spots_mean=('spots_intensity_sum', 'mean'),
              spots_max=('spots_intensity_sum', 'max'),
              clusters_mean=('clusters_norm_sum_int', 'mean'),
              clusters_max=('clusters_norm_sum_int', 'max'),
          )
    )

def build_particle_table(int_to_use, clusters_path, particle_label, cluster_file, mature_cells, mature_cells_frame):
    """
    For a given particle (Zap70 or CD19), compute per-frame per-cell total intensity,
    label frames mature/not-mature, then summarize per cell and maturity state.
    """
    # Spot intensities for this particle
    spot_df = int_to_use[int_to_use['particle'] == particle_label]

    # Cluster table for this particle
    cluster_df = pd.read_hdf(os.path.join(clusters_path, cluster_file))
    cluster_df = cluster_df[['cell_id', 'frame', 'sum_int', 'norm_sum_int']][cluster_df.cell_id.isin(mature_cells)]

    # Per-frame totals
    all_frames = sum_intensity_per_cell(spot_df, cluster_df)

    # Label frames by each cell's crossing frame
    all_frames["mature_from_frame"] = all_frames["cell_id"].map(mature_cells_frame)
    all_frames["mature"] = np.where(
        all_frames["frame"] >= all_frames["mature_from_frame"],
        "mature",
        "not-mature"
    )

    # Per-cell summary
    return summarize(all_frames).copy()


# %% Load dataset + input tables

#obtain experiment paths
a = plc.Dataset_combined_analysis(PATH)
run_paths = a.run_paths

intensities_spots = pd.read_csv(
    os.path.join(
        r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output',
        r'all_cotracks_velocities&directionality_correctedtime.csv'
    )
)
cell_id = pd.read_csv(
    os.path.join(
        r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output',
        r'all_cotracks_velocities&directionality_stats_correctedtime.csv'
    )
)


# %% Attach cell_id to spot intensities and scale to integrated intensity

right_unique = cell_id[['run', 'colocID', 'cell_id']].drop_duplicates(['run', 'colocID'])
intensities_spots_2 = intensities_spots.merge(right_unique, on=['run', 'colocID'], how='left')
#drop unnecessary columns
cols = intensities_spots_2.columns
int_df = intensities_spots_2[cols[9:]]

# Convert median(3x3) to approximate integrated intensity (multiply by area)
int_df.loc[:, 'intensity'] = int_df['intensity'] * SPOT_AREA


# %% Main analysis loop across runs

result = []

for i in tqdm(run_paths):
    #open maturation and cluster analysis file. 
    mature_path = os.path.join(i, 'maturation_analysis')
    cluster_and_spots_path = os.path.join(i, 'cluster_analysis_spots_filtered')  
    clusters_path = os.path.join(i, 'cluster_analysis')

    maturation_json = os.path.join(mature_path, "maturation__488nm.json")
    if not os.path.exists(maturation_json):
        continue

    with open(maturation_json, "r") as f:
        maturation = pd.DataFrame(json.load(f))

    # Get cells that mature over time and the frame where they are considered mature. 
    mature_cells = list(maturation[maturation.category == MATURE_CATEGORY]['cell'])
    mature_cells_frame = maturation[maturation.category == MATURE_CATEGORY]['crossing_frame'].to_dict()

    try:
        if not mature_cells:
            continue

        # Spot table restricted to this run and only maturing cells
        int_to_use = int_df[int_df['run'] == i].copy()
        int_to_use = int_to_use[int_to_use['cell_id'].isin(mature_cells)]

        # Build integrated intensity tables for Zap70 and CD19
        final_zap_df = build_particle_table(
            int_to_use=int_to_use,
            clusters_path=clusters_path,
            particle_label=ZAP_LABEL,
            cluster_file=ZAP_CLUSTER_FILE,
            mature_cells=mature_cells,
            mature_cells_frame=mature_cells_frame
        )

        final_cd_df = build_particle_table(
            int_to_use=int_to_use,
            clusters_path=clusters_path,
            particle_label=CD19_LABEL,
            cluster_file=CD19_CLUSTER_FILE,
            mature_cells=mature_cells,
            mature_cells_frame=mature_cells_frame
        )

        # Merge Zap and CD summaries
        merged = final_zap_df.merge(final_cd_df, on=['cell_id', 'mature'], suffixes=('_zap', '_cd'))

        # Compute Zap/CD ratios for all metrics
        ratio_df = merged[['cell_id', 'mature']].copy()
        for col in merged.columns:
            if col.endswith('_zap'):
                base = col.replace('_zap', '')
                col_cd = f"{base}_cd"
                ratio_df[f"{base}_ratio"] = merged[col] / merged[col_cd]

        ratio_df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

        merged_final = merged.merge(ratio_df, on=['cell_id', 'mature'])
        merged_final['run'] = i
        result.append(merged_final)

    except:
        print('not worked')


# %% Combine all runs and add metadata columns, then save

a_final_result = pd.concat(result)

a_final_result['CART'] = a_final_result['run'].apply(lambda x: x.split('\\')[5][0:5])

a_final_result["expr"] = np.select(
    [a_final_result["run"].str.contains("High exp", na=False),
     a_final_result["run"].str.contains("Low exp",  na=False)],
    ["High exp", "Low exp"],
    default=np.nan
)

a_final_result['dil'] = a_final_result['run'].apply(lambda x: x.split('\\')[6])

a_final_result.to_csv(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\intensity_maturation_summary.csv',
    index=False
)