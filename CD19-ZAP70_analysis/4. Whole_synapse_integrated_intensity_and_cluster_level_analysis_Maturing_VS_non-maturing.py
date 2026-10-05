# %% Imports
import os
import json
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

from postSPIT import tirf_analysis as plc
from spit import tools


# %% Configuration
SPOT_BOX_SIZE = 3
SPOT_AREA = SPOT_BOX_SIZE ** 2

IMMATURE_CATEGORY = 0
MATURING_CATEGORY = 1

ZAP_LABEL = "Zap70"
CD19_LABEL = "CD19"

ZAP_CLUSTER_FILE = "clusters_488nm.hdf"
CD19_CLUSTER_FILE = "clusters_638nm.hdf"

PATH = r"D:\Data\Chi_data\20250801_filtered\output"

ANALYSIS_OUTPUT_PATH = (
    r"P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered"
    r"\output\Nguyen2026_analysis\analysis_output"
)

SPOT_INTENSITY_FILE = os.path.join(
    ANALYSIS_OUTPUT_PATH,
    "all_cotracks_velocities&directionality_correctedtime.csv",
)

CELL_ID_FILE = os.path.join(
    ANALYSIS_OUTPUT_PATH,
    "all_cotracks_velocities&directionality_stats_correctedtime.csv",
)

OUTPUT_FILE = os.path.join(
    ANALYSIS_OUTPUT_PATH,
    "Integratedintensity_non-matureVSMature_summary.csv",
)


# %% Functions
def sum_intensity_per_cell(spot_df, cluster_df):
    """
    Calculate spot, cluster, and combined intensity for every frame and cell.

    Spots are summed per (frame, cell_id). Clusters are summed per
    (frame, cell_id). The two tables are outer-merged, and missing intensity
    values are treated as zero.
    """
    required_spot_cols = {"t", "cell_id", "intensity"}
    missing_spot_cols = required_spot_cols.difference(spot_df.columns)
    if missing_spot_cols:
        raise KeyError(
            f"Spot table is missing required columns: {sorted(missing_spot_cols)}"
        )

    required_cluster_cols = {"frame", "cell_id", "norm_sum_int"}
    missing_cluster_cols = required_cluster_cols.difference(cluster_df.columns)
    if missing_cluster_cols:
        raise KeyError(
            f"Cluster table is missing required columns: {sorted(missing_cluster_cols)}"
        )

    spots_per_frame = (
        spot_df.groupby(["t", "cell_id"], as_index=False)["intensity"]
        .sum()
        .rename(
            columns={
                "t": "frame",
                "intensity": "spots_intensity_sum",
            }
        )
    )

    clusters_per_frame = (
        cluster_df.groupby(["frame", "cell_id"], as_index=False)[
            "norm_sum_int"
        ]
        .sum()
        .rename(
            columns={
                "norm_sum_int": "clusters_norm_sum_int",
            }
        )
    )

    merged = spots_per_frame.merge(
        clusters_per_frame,
        on=["frame", "cell_id"],
        how="outer",
    )

    intensity_columns = [
        "spots_intensity_sum",
        "clusters_norm_sum_int",
    ]
    merged[intensity_columns] = merged[intensity_columns].fillna(0)

    merged["spots_plus_clusters"] = (
        merged["spots_intensity_sum"]
        + merged["clusters_norm_sum_int"]
    )

    return merged


def summarize_all_frames(df):
    """
    Calculate one summary row per cell using all available frames.

    This keeps the same statistics as the previous analysis: mean and maximum
    values for total, spot, and cluster intensity.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "cell_id",
                "n_frames",
                "total_mean",
                "total_max",
                "spots_mean",
                "spots_max",
                "clusters_mean",
                "clusters_max",
            ]
        )

    return (
        df.groupby("cell_id", as_index=False)
        .agg(
            n_frames=("frame", "nunique"),
            total_mean=("spots_plus_clusters", "mean"),
            total_max=("spots_plus_clusters", "max"),
            spots_mean=("spots_intensity_sum", "mean"),
            spots_max=("spots_intensity_sum", "max"),
            clusters_mean=("clusters_norm_sum_int", "mean"),
            clusters_max=("clusters_norm_sum_int", "max"),
        )
    )


def build_particle_table(
    int_to_use,
    clusters_path,
    particle_label,
    cluster_file,
    selected_cells,
):
    """
    Build a per-cell intensity summary for one particle using all frames.

    Parameters
    ----------
    int_to_use : pandas.DataFrame
        Spot intensity table already restricted to one run and the selected
        category-0/category-1 cells.
    clusters_path : str
        Folder containing the cluster HDF files.
    particle_label : str
        Particle name in the spot table, for example 'Zap70' or 'CD19'.
    cluster_file : str
        HDF filename for the corresponding channel.
    selected_cells : list
        Cell IDs belonging to category 0 or category 1.
    """
    spot_df = int_to_use[
        int_to_use["particle"] == particle_label
    ].copy()

    cluster_path = os.path.join(clusters_path, cluster_file)
    if not os.path.exists(cluster_path):
        raise FileNotFoundError(f"Cluster file not found: {cluster_path}")

    cluster_df = pd.read_hdf(cluster_path)

    required_columns = ["cell_id", "frame", "sum_int", "norm_sum_int"]
    missing_columns = [
        column for column in required_columns if column not in cluster_df.columns
    ]
    if missing_columns:
        raise KeyError(
            f"{cluster_path} is missing columns: {missing_columns}"
        )

    cluster_df = cluster_df.loc[
        cluster_df["cell_id"].isin(selected_cells),
        required_columns,
    ].copy()

    all_frames = sum_intensity_per_cell(spot_df, cluster_df)
    return summarize_all_frames(all_frames)


def get_time_interval(folder):
    """Extract the frame interval, in seconds, from a result.txt file."""
    result_files = glob(os.path.join(folder, "*result.txt"))
    if not result_files:
        raise FileNotFoundError("No result.txt file found in the folder.")

    with open(result_files[0], "r") as file:
        result_lines = file.readlines()

    interval_line = tools.find_string(result_lines, "Interval")

    if interval_line:
        interval = interval_line.split(":")[-1].strip()
        value, unit = interval.split(" ")[:2]

        if unit == "sec":
            dt = float(value)
        elif unit == "ms":
            dt = 0.001 * float(value)
        else:
            raise ValueError(f"Unsupported interval unit: {unit}")
    else:
        exposure_line = tools.find_string(result_lines, "Camera Exposure")
        if not exposure_line:
            raise ValueError(
                "Neither Interval nor Camera Exposure was found in result.txt"
            )

        dt_string = exposure_line[17:-1]
        numeric_text = "".join(
            character
            for character in dt_string
            if character.isdigit() or character == "."
        )
        dt = 0.001 * float(numeric_text)

    return dt


def add_run_metadata(df):
    """Add CART, expression level, and dilution metadata from each run path."""
    if df.empty:
        return df

    path_parts = df["run"].str.split("\\")

    df["CART"] = path_parts.apply(
        lambda parts: parts[5][:5] if len(parts) > 5 else np.nan
    )

    df["expr"] = np.select(
        [
            df["run"].str.contains("High exp", na=False),
            df["run"].str.contains("Low exp", na=False),
        ],
        ["High exp", "Low exp"],
        default=None,
    )

    df["dil"] = path_parts.apply(
        lambda parts: parts[6] if len(parts) > 6 else np.nan
    )

    return df


# %% Load dataset and input tables
dataset = plc.Dataset_combined_analysis(PATH)
run_paths = dataset.run_paths

intensities_spots = pd.read_csv(SPOT_INTENSITY_FILE)
cell_id_table = pd.read_csv(CELL_ID_FILE)


# %% Attach cell_id to spot intensities and scale to integrated intensity
right_unique = cell_id_table[["run", "colocID", "cell_id"]].drop_duplicates(
    ["run", "colocID"]
)

intensities_spots_2 = intensities_spots.merge(
    right_unique,
    on=["run", "colocID"],
    how="left",
    validate="many_to_one",
)

#drop unnecessary columns
cols = intensities_spots_2.columns
int_df = intensities_spots_2[cols[9:]]

# Convert median intensity in the 3 x 3 spot box to approximate integrated
# intensity by multiplying it by the box area.
int_df.loc[:, "intensity"] = int_df["intensity"] * SPOT_AREA


# %% Main analysis: category 0 versus category 1
results = []
failed_runs = []

for run_path in tqdm(run_paths):
    mature_path = os.path.join(run_path, "maturation_analysis")
    clusters_path = os.path.join(run_path, "cluster_analysis")

    maturation_json = os.path.join(
        mature_path,
        "maturation__488nm.json",
    )

    if not os.path.exists(maturation_json):
        continue

    try:
        with open(maturation_json, "r") as file:
            maturation = pd.DataFrame(json.load(file))

        required_maturation_columns = {"cell", "category"}
        missing_maturation_columns = required_maturation_columns.difference(
            maturation.columns
        )
        if missing_maturation_columns:
            raise KeyError(
                "Maturation table is missing required columns: "
                f"{sorted(missing_maturation_columns)}"
            )

        # Keep only cells that are explicitly category 0 or category 1.
        selected_maturation = maturation[
            maturation["category"].isin(
                [IMMATURE_CATEGORY, MATURING_CATEGORY]
            )
        ].copy()

        if selected_maturation.empty:
            continue

        selected_cells = selected_maturation["cell"].tolist()

        category_by_cell = (
            selected_maturation[["cell", "category"]]
            .drop_duplicates("cell")
            .set_index("cell")["category"]
            .to_dict()
        )

        # Restrict the spot table to this run and to category-0/category-1 cells.
        int_to_use = int_df[
            (int_df["run"] == run_path)
            & (int_df["cell_id"].isin(selected_cells))
        ].copy()

        final_zap_df = build_particle_table(
            int_to_use=int_to_use,
            clusters_path=clusters_path,
            particle_label=ZAP_LABEL,
            cluster_file=ZAP_CLUSTER_FILE,
            selected_cells=selected_cells,
        )

        final_cd_df = build_particle_table(
            int_to_use=int_to_use,
            clusters_path=clusters_path,
            particle_label=CD19_LABEL,
            cluster_file=CD19_CLUSTER_FILE,
            selected_cells=selected_cells,
        )

        # Outer merge retains a cell even if one channel has no detected signal.
        merged = final_zap_df.merge(
            final_cd_df,
            on="cell_id",
            how="outer",
            suffixes=("_zap", "_cd"),
        )

        merged["category"] = merged["cell_id"].map(category_by_cell)
        merged["maturation_group"] = merged["category"].map(
            {
                IMMATURE_CATEGORY: "non-maturing",
                MATURING_CATEGORY: "maturing",
            }
        )

        # Missing channel summaries are treated as zero, matching the original
        # handling of missing spot/cluster intensities.
        metric_columns = [
            column
            for column in merged.columns
            if column.endswith("_zap") or column.endswith("_cd")
        ]
        merged[metric_columns] = merged[metric_columns].fillna(0)

        # Calculate Zap70/CD19 ratios for all matching summary metrics.
        for zap_column in [
            column for column in merged.columns if column.endswith("_zap")
        ]:
            base_name = zap_column.removesuffix("_zap")
            cd_column = f"{base_name}_cd"

            if cd_column in merged.columns:
                merged[f"{base_name}_ratio"] = (
                    merged[zap_column] / merged[cd_column]
                )

        ratio_columns = [
            column for column in merged.columns if column.endswith("_ratio")
        ]
        merged[ratio_columns] = merged[ratio_columns].replace(
            [np.inf, -np.inf],
            np.nan,
        )
        merged[ratio_columns] = merged[ratio_columns].fillna(0)

        merged["run"] = run_path
        results.append(merged)

    except Exception as error:
        failed_runs.append(
            {
                "run": run_path,
                "error": str(error),
            }
        )
        print(f"{run_path} failed: {error}")


# %% Combine runs, add metadata, and save
if not results:
    raise RuntimeError(
        "No run produced a result. Check the printed errors and input paths."
    )

final_result = pd.concat(results, ignore_index=True)
final_result = add_run_metadata(final_result)

# Put identifying columns first.
identifier_columns = [
    "run",
    "CART",
    "expr",
    "dil",
    "cell_id",
    "category",
    "maturation_group",
]
remaining_columns = [
    column
    for column in final_result.columns
    if column not in identifier_columns
]
final_result = final_result[identifier_columns + remaining_columns]

os.makedirs(ANALYSIS_OUTPUT_PATH, exist_ok=True)
final_result.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(final_result)} cell summaries to:")
print(OUTPUT_FILE)

# if failed_runs:
#     failed_runs_file = os.path.join(
#         ANALYSIS_OUTPUT_PATH,
#         "intensity_category0_vs_category1_all_frames_failed_runs.csv",
#     )
#     pd.DataFrame(failed_runs).to_csv(failed_runs_file, index=False)
#     print(f"Saved failed-run details to: {failed_runs_file}")
