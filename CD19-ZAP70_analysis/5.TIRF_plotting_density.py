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

#%% Constants and helper functions
SAVE_FOLDER = r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\20260507_separate_densities\dotplots'
DIL_GROUP_MAP = {
    "50xdilutedCD19": "dense",
    "100xdilutedCD19": "dense",
    "500xdilutedCD19": "intermediate",
    "1000xdilutedCD19": "intermediate",
    "1500xdilutedCD19": "intermediate", 
    "3000xdilutedCD19": "sparse",
    "6000xdilutedCD19": "sparse",
}

CATEGORY_MAP = {
    0: "never matures",
    1: "matures",
    2: "starts mature",
}

CATEGORY_ORDER_ALL = ["never matures", "matures", "starts mature"]

CATEGORY_COLORS = {
    "matures": "#2ecc71",
    "never matures": "#b23a8a",
    "starts mature": "#1A7A42",
}

def add_common_columns(
    df,
    *,
    condition_col="condition",
    run_col="run",
    category_col="category",
    cart_out_col="cart",
    expr_out_col="expr",
    dil_out_col="dil",
    dil_group_out_col="dil_group",
    category_label_out_col="category_label",
):
    """
    Add the columns that are reused throughout the plotting code.
    """
    
    out = df.copy()

    # CART
    if condition_col in out.columns:
        out[cart_out_col] = np.select(
            [out[condition_col].str.contains("CART3", na=False),
             out[condition_col].str.contains("CART4", na=False)],
            ["CART3", "CART4"],
            default=np.nan
        )

        # Expression
        out[expr_out_col] = np.select(
            [out[condition_col].str.contains("High exp", na=False),
             out[condition_col].str.contains("Low exp",  na=False)],
            ["High exp", "Low exp"],
            default=np.nan
        )

    # Dilution + group (from run)
    if run_col in out.columns:
        out[dil_out_col] = out[run_col].astype(str).str.extract(r"(\d+xdilutedCD19)")[0]
        out[dil_group_out_col] = out[dil_out_col].map(DIL_GROUP_MAP)

    # Category label
    if category_col in out.columns:
        out[category_label_out_col] = out[category_col].map(CATEGORY_MAP)
        out[category_label_out_col] = pd.Categorical(out[category_label_out_col], categories=CATEGORY_ORDER_ALL, ordered=True)
    return out


def bootstrap_ci_median(x, n_boot=5000, ci=95, seed=42):
    """
    Compute the median and a nonparametric bootstrap confidence interval.

    The interval is estimated by resampling the data with replacement and using
    percentile bounds from the bootstrap median distribution. Missing values are
    ignored.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True)
    meds = np.median(boots, axis=1)
    lo = np.percentile(meds, (100 - ci) / 2)
    hi = np.percentile(meds, 100 - (100 - ci) / 2)
    return np.median(x), lo, hi


def geometric_mean_confidence_interval(data, confidence=0.95):
    """
    Compute the gemoetric mean and the confidence interval.
    """
    a_log = np.log(data)
    a_log_mean = np.mean(a_log)
    standard_deviation = np.std(a_log, ddof=1)
    standard_error = standard_deviation / np.sqrt(len(a_log))
    alpha = 1 - confidence
    tcrit = stats.t.ppf(1 - alpha/2, df=len(a_log) - 1)
    a_low = a_log_mean - tcrit * standard_error
    a_high = a_log_mean + tcrit * standard_error
    return np.exp(a_log_mean), np.exp(a_low), np.exp(a_high)


def signflip_permutation_pvalue(delta, n_perm=10000, seed=42, stat="median", sides="two-sided"):
    """
    Paired sign-flip permutation test for the null hypothesis that delta is
    symmetric around 0, meaning no systematic change.

    By default, the test uses the median of delta as a robust summary statistic.
    """
    rng = np.random.default_rng(seed)
    delta = np.asarray(delta, dtype=float)
    delta = delta[~np.isnan(delta)]
    if len(delta) == 0:
        return np.nan

    if stat == "median":
        obs = np.median(delta)
        stat_fn = np.median
    elif stat == "mean":
        obs = np.mean(delta)
        stat_fn = np.mean
    else:
        raise ValueError("stat must be 'median' or 'mean'")

    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(delta)), replace=True)
    perm_stats = stat_fn(signs * delta, axis=1)

    if sides == "two-sided":
        p = (np.sum(np.abs(perm_stats) >= np.abs(obs)) + 1) / (n_perm + 1)
    elif sides == "one-sided":
        p = (np.sum(perm_stats >= obs) + 1) / (n_perm + 1)
    else:
        raise ValueError("sides must be 'two-sided' or 'one-sided'")
    return p


def permutation_test_between_groups(x1, x2, n_perm=10000, seed=42, stat="median"):
    """
    Two-sided label-permutation test for a difference in location between two
    independent groups.
    """
    rng = np.random.default_rng(seed)
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]
    if len(x1) == 0 or len(x2) == 0:
        return np.nan

    if stat == "median":
        stat_fn = np.median
    elif stat == "mean":
        stat_fn = np.mean
    else:
        raise ValueError("stat must be 'median' or 'mean'")

    obs = stat_fn(x1) - stat_fn(x2)

    pooled = np.concatenate([x1, x2])
    n1 = len(x1)
    perm_stats = np.empty(n_perm, dtype=float)

    for i in range(n_perm):
        rng.shuffle(pooled)
        perm_x1 = pooled[:n1]
        perm_x2 = pooled[n1:]
        perm_stats[i] = stat_fn(perm_x1) - stat_fn(perm_x2)

    p = (np.sum(np.abs(perm_stats) >= np.abs(obs)) + 1) / (n_perm + 1)
    return p


def format_p(p):
    if pd.isna(p):
        return "p = NA"
    return f"p = {p:.3g}"


def add_bracket_with_p(ax, x1, x2, yb, h, p_text, *, lw=1, fontsize=9):
    """
    Draw a bracket from x1 to x2 at baseline yb, with height h, and place the
    p-value label above it.
    """
    ax.plot([x1, x1, x2, x2],
            [yb, yb + h, yb + h, yb],
            color="black", linewidth=lw)
    ax.text((x1 + x2) / 2,
            yb + h * 1.2,
            p_text,
            ha="center", va="bottom",
            fontsize=fontsize, color="black")
def expand_tuple_column(df, col="tmp", names=("median", "ci_low", "ci_high")):
    df[list(names)] = pd.DataFrame(df[col].tolist(), index=df.index)
    return df.drop(columns=[col])

#%%Directionality plot - change expression on the first line

# Directionality is bounded between -1 and 1 and may not follow a normal distribution.
# For that reason, the plots use the median as the summary statistic.
# Uncertainty is shown with nonparametric bootstrap confidence intervals
# based on 5000 resamples, which works well for bounded and potentially skewed data.
# See: http://staff.ustc.edu.cn/~zwp/teach/Stat-Comp/Efron_Bootstrap_CIs.pdf
expression = "Low exp"
ycol = "directionality"

directionalities_maturation = pd.read_csv(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\long_term_directionality_with_maturation.csv'
)
# Remove rows without an assigned transition.
plot_df = directionalities_maturation[directionalities_maturation["transition"].notna()].copy()

# Remove cells that could not be classified.
plot_df = plot_df[plot_df["category"] <= 2].copy()

#Add grouping columns
plot_df = add_common_columns(plot_df, condition_col="cond", run_col="run", category_col="category")

# Keep only the selected expression level.
plot_df = plot_df[plot_df["expr"] == expression]

# group and make plotting table 
summary_wide = (
    plot_df.groupby(["dil_group", "cart", "transition", "category"])[ycol]
    .apply(lambda s: bootstrap_ci_median(s.values))
    .reset_index(name="tmp")
)
summary_wide = expand_tuple_column(summary_wide, col="tmp")

summary_wide = summary_wide[summary_wide['transition'] != 'loc2-loc3']

summary_wide["category_label"] = summary_wide["category"].map(CATEGORY_MAP)

order = list(plot_df["transition"].dropna().unique())
# hue_order = ["never matures", "matures", "starts mature"]

def plot_points_with_ci(data, **kws):
    ax = plt.gca()

    sns.pointplot(
        data=data,
        x="transition", y="median",
        hue="category_label",
        order=order, hue_order=CATEGORY_ORDER_ALL,
        linestyle="None", dodge=0.5,
        errorbar=None,
        palette=CATEGORY_COLORS ,
        ax=ax
    )

    # Draw confidence interval bars at the shifted x positions.
    xticks = ax.get_xticks()
    n_hue = len(CATEGORY_ORDER_ALL)
    dodge = 0.5
    hue_to_j = {h: j for j, h in enumerate(CATEGORY_ORDER_ALL)}

    for _, r in data.iterrows():
        i = order.index(r["transition"])
        j = hue_to_j[r["category_label"]]
        offset = 0.0 if n_hue == 1 else (j - (n_hue - 1) / 2) * (dodge / (n_hue - 1))
        x = xticks[i] + offset

        y = r["median"]
        yerr = [[y - r["ci_low"]], [r["ci_high"] - y]]
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", elinewidth=1, capsize=3)

    # Add the reference line and set the plot limits.
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Transition")
    ax.set_ylabel(f"{ycol} (median ± 95% bootstrap CI)")

    # Remove per-axis legends because a single figure legend is added later.
    leg = ax.get_legend()
    if leg:
        leg.remove()

g = sns.FacetGrid(
    summary_wide,
    row="dil_group",
    col="cart",
    row_order=sorted(summary_wide["dil_group"].dropna().unique()),
    col_order=["CART3", "CART4"],
    sharey=True,
    height=4,
    aspect=1.2
)
g.map_dataframe(plot_points_with_ci)

# Add one legend for the full figure.
g.add_legend(title="category", label_order=CATEGORY_ORDER_ALL)
summary_wide.to_csv(os.path.join(SAVE_FOLDER, f'Fig2_directionality_{expression}.csv'), index=False)
# Use simple facet titles.
g.set_titles(col_template="{col_name}")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_FOLDER, f'Fig2_directionality_{expression}.png'), dpi=600)
plt.savefig(os.path.join(SAVE_FOLDER, f'Fig2_directionality_{expression}.pdf'), dpi=600)
plt.show()


#%% Directionality difference: permutation test around 0 of the per-track difference of directionality
expression = "Low exp"
ycol = "directionality"

directionalities_maturation = pd.read_csv(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\long_term_directionality_with_maturation.csv'
)

# Remove rows without an assigned transition.
plot_df = directionalities_maturation[directionalities_maturation["transition"].notna()].copy()
# Remove cells that could not be classified.
plot_df = plot_df[plot_df["category"] <= 2].copy()

# Add grouping columns
plot_df = add_common_columns(plot_df, condition_col="cond", run_col="run", category_col="category")

# Keep only the selected expression level.
plot_df = plot_df[plot_df["expr"] == expression]

# Calculate difference in directionality between during (loc1-loc2) and before localization (loc0-loc1)
needed_transitions = ["loc0-loc1", "loc1-loc2"]
plot_df = plot_df[plot_df["transition"].isin(needed_transitions)].copy()

plot_df["category_label"] = plot_df["category"].map(CATEGORY_MAP)

wide = (
    plot_df.pivot_table(
        index=["run","cart", "category_label", "colocID", "dil_group"],
        columns="transition",
        values=ycol,
        aggfunc="first",
    )
    .reset_index()
)

if not all(t in wide.columns for t in needed_transitions):
    raise KeyError(
        f"Missing one of {needed_transitions} in pivoted columns. "
        f"Found: {sorted([c for c in wide.columns if c not in ['cart','category_label','colocID','dil_group']])}"
    )

wide["delta"] = wide["loc1-loc2"] - wide["loc0-loc1"]
delta_df = wide.dropna(subset=["delta"]).copy()

# Summarize with median and bootstrap confidence interval and make plotting table
summary_delta = (
    delta_df.groupby(["cart", "category_label", "dil_group"])["delta"]
    .apply(lambda s: bootstrap_ci_median(s.values))
    .reset_index(name="tmp")
)
summary_delta[["median", "ci_low", "ci_high"]] = pd.DataFrame(
    summary_delta["tmp"].tolist(), index=summary_delta.index
)
summary_delta = summary_delta.drop(columns="tmp")

# Test whether the per-track difference differs from zero for each CAR and category.
pvals = (
    delta_df.groupby(["cart", "category_label", "dil_group"])["delta"]
    .apply(lambda s: signflip_permutation_pvalue(
        s.values, n_perm=10000, seed=42, stat="median", sides="one-sided"
    ))
    .reset_index(name="p_value")
)
summary_delta = summary_delta.merge(pvals, on=["cart", "category_label", "dil_group"], how="left")
summary_delta["transition"] = "Δ (loc1-loc2 − loc0-loc1)"

# Plot the delta panel using the same visual style as above.
order = summary_delta["transition"].dropna().unique().tolist()
hue_order = CATEGORY_ORDER_ALL
def plot_points_with_ci_delta(data, **kws):
    ax = plt.gca()

    sns.pointplot(
        data=data,
        x="transition", y="median",
        hue="category_label",
        order=order, hue_order=CATEGORY_ORDER_ALL,
        linestyle="None", dodge=0.5,
        errorbar=None,
        palette=CATEGORY_COLORS ,
        ax=ax
    )

    # Draw confidence interval bars at the shifted x positions.
    xticks = ax.get_xticks()
    n_hue = len(hue_order)
    dodge = 0.5
    hue_to_j = {h: j for j, h in enumerate(hue_order)}

    # Add confidence intervals and p-value labels.
    for _, r in data.iterrows():
        i = order.index(r["transition"])
        j = hue_to_j[r["category_label"]]
        offset = 0.0 if n_hue == 1 else (j - (n_hue - 1) / 2) * (dodge / (n_hue - 1))
        x = xticks[i] + offset

        y = r["median"]
        yerr = [[y - r["ci_low"]], [r["ci_high"] - y]]
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", elinewidth=1, capsize=3)

        p = r.get("p_value", np.nan)
        if not pd.isna(p):
            y_text = r["ci_high"] + 0.08
            ax.text(
                x, y_text,
                f"p = {p:.3g}",
                ha="center", va="bottom",
                fontsize=9,
                color="black"
            )

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Transition")
    ax.set_ylabel("Δ directionality (median ± 95% bootstrap CI)")

    leg = ax.get_legend()
    if leg:
        leg.remove()

dil_order = ["sparse", "intermediate", "dense"]
dil_order = [
    d for d in dil_order
    if d in summary_delta["dil_group"].dropna().unique()
]

g = sns.FacetGrid(
    summary_delta,
    row="dil_group",
    col="cart",
    row_order=dil_order,
    col_order=["CART3", "CART4"],
    sharey=True,
    height=4,
    aspect=1.2
)

g.map_dataframe(plot_points_with_ci_delta)
g.add_legend(title="category", label_order=hue_order)
g.set_titles(row_template="{row_name}", col_template="{col_name}")
g.map_dataframe(plot_points_with_ci_delta)
g.add_legend(title="category", label_order=hue_order)
g.set_titles(col_template="{col_name}")

# summary_delta.to_csv(os.path.join(SAVE_FOLDER, f'Fig2_directionalityDIFF_{expression}.csv'), index=False)
plt.tight_layout()
# plt.savefig(os.path.join(SAVE_FOLDER, f'Fig2_directionalityDIFF_{expression}.png'), dpi=600)
# plt.savefig(os.path.join(SAVE_FOLDER, f'Fig2_directionalityDIFF_{expression}.pdf'), dpi=600)
plt.show()


#%%velocities per maturation - Welch t-test
#%%velocities per maturation - Welch t-test separated by density
expression = "Low exp"
particle = "CD19"

velocities = pd.read_csv(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\all_cotracks_velocities&directionality_correctedtime.csv'
)

velocities_stats = pd.read_csv(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\all_cotracks_velocities&directionality_stats_correctedtime.csv'
)

velocities_groups = velocities.groupby(["run", "colocID", "particle"])
velocities_means = velocities_groups["velocity"].mean().reset_index()

right_unique = (
    velocities_stats[
        ["run", "colocID", "category", "condition"]
    ]
    .drop_duplicates(["run", "colocID"])
)

velocities_means_maturation = velocities_means.merge(
    right_unique,
    on=["run", "colocID"],
    how="left"
)

velocities_means_maturation = add_common_columns(
    velocities_means_maturation,
    condition_col="condition",
    run_col="run",
    category_col="category"
)

plot_df = velocities_means_maturation[
    (velocities_means_maturation["expr"] == expression)
    & (velocities_means_maturation["category"] < 2)
    & (velocities_means_maturation["particle"] == particle)
    & velocities_means_maturation["dil_group"].notna()
].copy()

plot_df["dil_group"] = (
    plot_df["dil_group"]
    .astype(str)
    .str.strip()
    .str.lower()
)

ycol = "velocity"

plot_df = plot_df[
    plot_df[ycol].notna()
    & (plot_df[ycol] > 0)
].copy()

dil_order = ["sparse", "intermediate", "dense"]
dil_order = [
    d for d in dil_order
    if d in plot_df["dil_group"].unique()
]

gm_ci = (
    plot_df.groupby(["dil_group", "cart", "category_label"])[ycol]
    .apply(lambda s: geometric_mean_confidence_interval(s.values))
)

summary = gm_ci.apply(pd.Series)
summary.columns = ["geo_mean", "ci_low", "ci_high"]
summary = summary.reset_index()

pvals = []

for d in dil_order:
    for cat in ["never matures", "matures"]:
        sub = plot_df[
            (plot_df["dil_group"] == d)
            & (plot_df["category_label"] == cat)
        ].copy()

        g1 = np.log(sub.loc[sub["cart"] == "CART3", ycol].values)
        g2 = np.log(sub.loc[sub["cart"] == "CART4", ycol].values)

        if len(g1) < 2 or len(g2) < 2:
            p = np.nan
        else:
            _, p = stats.ttest_ind(g1, g2, equal_var=False)

        pvals.append({
            "dil_group": d,
            "category_label": cat,
            "p_value": p
        })

pvals = pd.DataFrame(pvals)

summary = summary.merge(
    pvals,
    on=["dil_group", "category_label"],
    how="left"
)

hue_order = ["never matures", "matures"]

cart_order = ["CART3", "CART4"]
cart_markers = {"CART3": "s", "CART4": "o"}

fig, axes = plt.subplots(
    1,
    len(dil_order),
    figsize=(6 * len(dil_order), 4),
    sharey=True
)

if len(dil_order) == 1:
    axes = [axes]

cart_to_i = {c: i for i, c in enumerate(cart_order)}
hue_to_j = {h: j for j, h in enumerate(hue_order)}

cart_step = 0.24
cat_step = 0.10

all_ci_high = summary["ci_high"].max()
y_base = all_ci_high * 0.95
h = all_ci_high * 0.05
gap = all_ci_high * 0.12

for ax, d in zip(axes, dil_order):

    sub_summary = summary[summary["dil_group"] == d].copy()
    sub_pvals = pvals[pvals["dil_group"] == d].copy()

    pos = {}

    for _, r in sub_summary.iterrows():
        if pd.isna(r["geo_mean"]) or pd.isna(r["ci_low"]) or pd.isna(r["ci_high"]):
            continue

        if r["cart"] not in cart_to_i or r["category_label"] not in hue_to_j:
            continue

        cart_i = cart_to_i[r["cart"]]
        cart_offset = (cart_i - (len(cart_order) - 1) / 2) * cart_step

        hue_j = hue_to_j[r["category_label"]]
        cat_offset = (hue_j - (len(hue_order) - 1) / 2) * cat_step

        x = cart_offset + cat_offset
        y = r["geo_mean"]

        ax.plot(
            x,
            y,
            marker=cart_markers.get(r["cart"], "o"),
            linestyle="None",
            markerfacecolor=CATEGORY_COLORS[r["category_label"]],
            markeredgecolor=CATEGORY_COLORS[r["category_label"]],
            markersize=7,
        )

        ax.errorbar(
            x,
            y,
            yerr=[[y - r["ci_low"]], [r["ci_high"] - y]],
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=3
        )

        pos.setdefault(r["category_label"], {})[r["cart"]] = (x, r["ci_high"])

    cat_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=CATEGORY_COLORS[c],
            markeredgecolor=CATEGORY_COLORS[c],
            label=c,
            markersize=7
        )
        for c in hue_order
    ]

    ax.set_title(f"dil_group: {d}")

    ax.set_xticks([
        (cart_to_i[c] - (len(cart_order) - 1) / 2) * cart_step
        for c in cart_order
    ])
    ax.set_xticklabels(cart_order)

    ax.set_xlabel("CAR")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    leg1 = ax.legend(
        handles=cat_handles,
        title="Category",
        loc="lower right",
        frameon=False
    )
    ax.add_artist(leg1)

    for j, cat in enumerate(hue_order):
        if cat not in pos or "CART3" not in pos[cat] or "CART4" not in pos[cat]:
            continue

        x1, _ = pos[cat]["CART3"]
        x2, _ = pos[cat]["CART4"]
        y_bracket = y_base + j * gap

        p_row = sub_pvals[sub_pvals["category_label"] == cat]

        if p_row.empty:
            continue

        p = float(p_row["p_value"].values[0])
        add_bracket_with_p(ax, x1, x2, y_bracket, h, format_p(p))

    ax.set_ylim(
        bottom=0,
        top=y_base + (len(hue_order) - 1) * gap + h * 2.5
    )

axes[0].set_ylabel(f"Velocity {particle} (GM ± 95% t-CI on log)")

plt.tight_layout()

summary.to_csv(
    os.path.join(
        SAVE_FOLDER,
        rf"Fig2_velocity_maturation_{particle}_{expression}_by_density.csv"
    ),
    index=False
)

plt.savefig(os.path.join(SAVE_FOLDER,rf"Fig2_velocity_maturation_{particle}_{expression}_by_density.png"),dpi=600)

plt.savefig(os.path.join(SAVE_FOLDER,rf"Fig2_velocity_maturation_{particle}_{expression}_by_density.pdf"),dpi=600)

plt.show()

#%% velocities per stage - mature only
expression = "High exp"
particle = 'CD19'
# Load the input table.
velocities = pd.read_csv(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\all_cotracks_velocities&directionality_stats_correctedtime.csv'
)

# Apply the filtering choices used for this panel.
velocities = velocities[velocities.particle == particle]

velocities_clean = velocities.loc[
    velocities['avg_speed'].notna()
    & (velocities['avg_speed'] > 0)
    & (velocities['timing'].isin(['pre', 'during']))   
    & (velocities['category'] == 1)                  
].copy()

# Add grouping columns
velocities_clean = add_common_columns(
    velocities_clean,
    condition_col="condition",
    run_col="run",
    category_col="category"
)

plot_df = velocities_clean[
    (velocities_clean["expr"] == expression)
    # & (velocities_clean["dil_group"] != "intermediate")
    & velocities_clean["cart"].notna()
].copy()
ycol = "avg_speed"

dil_order = sorted(plot_df["dil_group"].dropna().unique())

gm_ci = (
    plot_df.groupby(["dil_group", "timing", "cart"])[ycol]
    .apply(lambda s: geometric_mean_confidence_interval(s.values))
)

summary = gm_ci.apply(pd.Series)
summary.columns = ["geo_mean", "ci_low", "ci_high"]
summary = summary.reset_index()

# Compare CART3 and CART4 within each dil_group and timing
order = ["pre", "during"]

pvals = []
for d in dil_order:
    for t in order:
        sub = plot_df[
            (plot_df["dil_group"] == d)
            & (plot_df["timing"] == t)
        ].copy()

        g1 = np.log(sub.loc[sub["cart"] == "CART3", ycol].values)
        g2 = np.log(sub.loc[sub["cart"] == "CART4", ycol].values)

        if len(g1) < 2 or len(g2) < 2:
            p = np.nan
        else:
            _, p = stats.ttest_ind(g1, g2, equal_var=False)

        pvals.append({
            "dil_group": d,
            "timing": t,
            "p_value": p
        })

pvals = pd.DataFrame(pvals)

# Plot one subplot per dil_group
fig, axes = plt.subplots(
    1, len(dil_order),
    figsize=(5.2 * len(dil_order), 3.8),
    sharey=True
)

if len(dil_order) == 1:
    axes = [axes]

timing_spacing = 0.55
x_base = {t: i * timing_spacing for i, t in enumerate(order)}

cart_order = ["CART3", "CART4"]
cart_to_i = {c: i for i, c in enumerate(cart_order)}
cart_markers = {"CART3": "s", "CART4": "o"}

car_step = 0.18
color = "#2ecc71"

all_ci_high = summary["ci_high"].max()
h = all_ci_high * 0.05
gap = all_ci_high * 0.08
y_base_bracket = all_ci_high * 1.10

for ax, d in zip(axes, dil_order):

    sub_summary = summary[summary["dil_group"] == d].copy()
    pos = {}

    for _, r in sub_summary.iterrows():
        if r["timing"] not in x_base or r["cart"] not in cart_to_i:
            continue

        base = x_base[r["timing"]]
        cart_i = cart_to_i[r["cart"]]
        cart_offset = (cart_i - (len(cart_order) - 1) / 2) * car_step

        x = base + cart_offset
        y = r["geo_mean"]

        ax.plot(
            x, y,
            marker=cart_markers[r["cart"]],
            linestyle="None",
            color=color,
            markersize=7
        )

        ax.errorbar(
            x, y,
            yerr=[[y - r["ci_low"]], [r["ci_high"] - y]],
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=3
        )

        pos.setdefault(r["timing"], {})[r["cart"]] = (x, r["ci_high"])

    # Brackets and p-values
    for j, t in enumerate(order):
        if t not in pos or "CART3" not in pos[t] or "CART4" not in pos[t]:
            continue

        x1, _ = pos[t]["CART3"]
        x2, _ = pos[t]["CART4"]
        yb = y_base_bracket + j * gap

        p_row = pvals[
            (pvals["dil_group"] == d)
            & (pvals["timing"] == t)
        ]

        p = float(p_row["p_value"].values[0])
        add_bracket_with_p(ax, x1, x2, yb, h, format_p(p))

    ax.set_title(f"dil_group: {d}")
    ax.set_xticks([x_base[t] for t in order])
    ax.set_xticklabels(order)
    ax.set_xlabel("Timing")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Velocity (GM ± 95% t-CI on log)")

for ax in axes:
    ax.set_ylim(bottom=0, top=y_base_bracket + (len(order) - 1) * gap + h * 3)
    ax.set_xlim(-0.5, x_base[order[-1]] + 0.5)

handles = [
    Line2D(
        [0], [0],
        marker=cart_markers[c],
        linestyle="None",
        color=color,
        label=c,
        markersize=7
    )
    for c in cart_order
]

axes[-1].legend(handles=handles, title="CAR", frameon=False, loc="lower right")

plt.tight_layout()
summary.to_csv(
    os.path.join(SAVE_FOLDER, fr'Fig2_velocity_stage_{particle}_{expression}.csv'),
    index=False
)
plt.savefig(os.path.join(SAVE_FOLDER, fr'Fig2_velocity_stage_{particle}_{expression}.png'), dpi=600)
plt.savefig(os.path.join(SAVE_FOLDER, fr'Fig2_velocity_stage_{particle}_{expression}.pdf'), dpi=600)
plt.show()


#%% paired difference of velocities (during − pre) per track for statsitics of decrase. T-test on log scale for between difference of 1, and welch t-test for between CARs.
expression = "High exp"
particle = "CD19"

velocities = pd.read_csv(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\all_cotracks_velocities&directionality_stats_correctedtime.csv'
)

velocities = velocities[velocities.particle == particle]

velocities_clean = velocities.loc[
    velocities["avg_speed"].notna()
    & (velocities["avg_speed"] > 0)
    & (velocities["timing"].isin(["pre", "during"]))
    & (velocities["category"] == 1)
].copy()

velocities_clean = add_common_columns(
    velocities_clean,
    condition_col="condition",
    run_col="run",
    category_col="category"
)

plot_df = velocities_clean[
    (velocities_clean["expr"] == expression)
    & velocities_clean["cart"].notna()
    & velocities_clean["dil_group"].notna()
].copy()

plot_df["dil_group"] = (
    plot_df["dil_group"]
    .astype(str)
    .str.strip()
    .str.lower()
)

ycol = "avg_speed"

index_cols = ["dil_group", "colocID", "cart"]
if "run" in plot_df.columns:
    index_cols = ["dil_group", "run", "colocID", "cart"]

wide = (
    plot_df.pivot_table(
        index=index_cols,
        columns="timing",
        values=ycol,
        aggfunc="first",
    )
    .reset_index()
)

wide = wide.dropna(subset=["pre", "during"]).copy()
wide["fold_change"] = wide["during"] / wide["pre"]
wide = wide[wide["fold_change"].notna() & (wide["fold_change"] > 0)].copy()
wide["log_fc"] = np.log(wide["fold_change"])

dil_order = ["sparse", "intermediate", "dense"]
dil_order = [
    d for d in dil_order
    if d in wide["dil_group"].dropna().unique()
]

gm_ci = (
    wide.groupby(["dil_group", "cart"])["fold_change"]
    .apply(lambda s: geometric_mean_confidence_interval(s.values))
)

summary = gm_ci.apply(pd.Series)
summary.columns = ["geo_mean", "ci_low", "ci_high"]
summary = summary.reset_index()

p_within = []

for d in dil_order:
    for c in ["CART3", "CART4"]:
        vals = wide.loc[
            (wide["dil_group"] == d)
            & (wide["cart"] == c),
            "log_fc"
        ].values

        if len(vals) < 2:
            p = np.nan
        else:
            _, p = stats.ttest_1samp(vals, 0)

        p_within.append({
            "dil_group": d,
            "cart": c,
            "p_within": p
        })

p_within = pd.DataFrame(p_within)

summary = summary.merge(
    p_within,
    on=["dil_group", "cart"],
    how="left"
)

p_between = []

for d in dil_order:
    g1 = wide.loc[
        (wide["dil_group"] == d)
        & (wide["cart"] == "CART3"),
        "log_fc"
    ].values

    g2 = wide.loc[
        (wide["dil_group"] == d)
        & (wide["cart"] == "CART4"),
        "log_fc"
    ].values

    if len(g1) < 2 or len(g2) < 2:
        p = np.nan
    else:
        _, p = stats.ttest_ind(g1, g2, equal_var=False)

    p_between.append({
        "dil_group": d,
        "p_between": p
    })

p_between = pd.DataFrame(p_between)

cart_order = ["CART3", "CART4"]
cart_markers = {"CART3": "s", "CART4": "o"}

fig, axes = plt.subplots(
    1,
    len(dil_order),
    figsize=(5.2 * len(dil_order), 3.8),
    sharey=True
)

if len(dil_order) == 1:
    axes = [axes]

color = "#2ecc71"
x_base = {c: i for i, c in enumerate(cart_order)}

all_ci_high = summary["ci_high"].max()
y_pad = all_ci_high * 0.06
yb = all_ci_high * 1.18
h = all_ci_high * 0.06
top = all_ci_high * 1.35 if len(summary) else 1

for ax, d in zip(axes, dil_order):

    sub_summary = summary[summary["dil_group"] == d].copy()
    pos = {}

    for _, r in sub_summary.iterrows():
        if r["cart"] not in x_base:
            continue

        x = x_base[r["cart"]]
        y = r["geo_mean"]

        ax.plot(
            x,
            y,
            marker=cart_markers[r["cart"]],
            linestyle="None",
            color=color,
            markersize=7
        )

        ax.errorbar(
            x,
            y,
            yerr=[[y - r["ci_low"]], [r["ci_high"] - y]],
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=3
        )

        pos[r["cart"]] = (x, y, r["ci_high"], r.get("p_within", np.nan))

    ax.axhline(1, color="gray", linestyle="--", linewidth=1)

    for c in cart_order:
        if c not in pos:
            continue

        x, y, ci_hi, p = pos[c]

        ax.text(
            x,
            ci_hi + y_pad,
            format_p(p),
            ha="center",
            va="bottom",
            fontsize=9,
            color="black"
        )

    if "CART3" in pos and "CART4" in pos:
        x1, _, _, _ = pos["CART3"]
        x2, _, _, _ = pos["CART4"]

        p = float(
            p_between.loc[
                p_between["dil_group"] == d,
                "p_between"
            ].values[0]
        )

        add_bracket_with_p(ax, x1, x2, yb, h, format_p(p))

    ax.set_title(f"dil_group: {d}")
    ax.set_xticks([x_base[c] for c in cart_order])
    ax.set_xticklabels(cart_order)
    ax.set_xlabel("CAR")
    ax.set_ylim(bottom=0, top=top)
    ax.set_xlim(-0.5, len(cart_order) - 0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Velocity fold-change (during / pre)\n(GM ± 95% t-CI on log)")

handles = [
    Line2D(
        [0], [0],
        marker=cart_markers[c],
        linestyle="None",
        color=color,
        label=c,
        markersize=7
    )
    for c in cart_order
]

axes[-1].legend(handles=handles, title="CAR", frameon=False, loc="lower right")

summary.to_csv(os.path.join(SAVE_FOLDER, f'Fig2_velocity_stage_Diff{particle}_{expression}_by_density.csv'), index=False)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_FOLDER, f'Fig2_velocity_stage_Diff{particle}_{expression}_by_density.png'), dpi=600)
plt.savefig(os.path.join(SAVE_FOLDER, f'Fig2_velocity_stage_Diff{particle}_{expression}_by_density.pdf'), dpi=600)
plt.show()
#%% Intensities small clusters - mature over time only (with permutation test CART3 vs CART4 per timing)
expression = "Low exp"
particle = "Zap70"

intensities = pd.read_csv(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\all_cotracks_velocities&directionality_stats_correctedtime.csv'
)

intensities_particle = intensities[intensities.particle == particle].copy()

plot_df = intensities_particle[intensities_particle["timing"].notna()].copy()
plot_df = plot_df[plot_df["category"] == 1].copy()

plot_df = add_common_columns(
    plot_df,
    condition_col="condition",
    run_col="run",
    category_col="category"
)

plot_df = plot_df[
    (plot_df["expr"] == expression)
    & plot_df["dil_group"].notna()
].copy()

plot_df["dil_group"] = (
    plot_df["dil_group"]
    .astype(str)
    .str.strip()
    .str.lower()
)

plot_df = plot_df[plot_df["timing"] != "post"].copy()

ycol = "median_intensity"

summary_wide = (
    plot_df.groupby(["dil_group", "cart", "timing", "category"])[ycol]
    .apply(lambda s: bootstrap_ci_median(s.values))
    .reset_index(name="tmp")
)

summary_wide = expand_tuple_column(summary_wide, col="tmp")

order = ["pre", "during"]
summary_plot = summary_wide[summary_wide["timing"].isin(order)].copy()

dil_order = ["sparse", "intermediate", "dense"]
dil_order = [
    d for d in dil_order
    if d in summary_plot["dil_group"].dropna().unique()
]

pvals = []

for d in dil_order:
    for t in order:
        sub = plot_df[
            (plot_df["dil_group"] == d)
            & (plot_df["timing"] == t)
        ]

        g1 = sub.loc[sub["cart"] == "CART3", ycol].values
        g2 = sub.loc[sub["cart"] == "CART4", ycol].values

        if len(g1) < 2 or len(g2) < 2:
            p = np.nan
        else:
            p = permutation_test_between_groups(
                g1,
                g2,
                n_perm=10000,
                seed=42,
                stat="median"
            )

        pvals.append({
            "dil_group": d,
            "timing": t,
            "p_value": p
        })

pvals = pd.DataFrame(pvals)

timing_spacing = 0.55
x_base = {t: i * timing_spacing for i, t in enumerate(order)}

cart_order = ["CART3", "CART4"]
cart_to_i = {c: i for i, c in enumerate(cart_order)}
cart_markers = {"CART3": "s", "CART4": "o"}

car_step = 0.18
color = "#2ecc71"

fig, axes = plt.subplots(
    1,
    len(dil_order),
    figsize=(5.2 * len(dil_order), 3.8),
    sharey=True
)

if len(dil_order) == 1:
    axes = [axes]

all_ci_high = summary_plot["ci_high"].max()
h = all_ci_high * 0.05
gap = all_ci_high * 0.08
y_base_bracket = all_ci_high * 1.10

for ax, d in zip(axes, dil_order):

    sub_summary = summary_plot[summary_plot["dil_group"] == d].copy()
    pos = {}

    for _, r in sub_summary.iterrows():
        if r["timing"] not in x_base or r["cart"] not in cart_to_i:
            continue

        base = x_base[r["timing"]]
        cart_i = cart_to_i[r["cart"]]
        cart_offset = (cart_i - (len(cart_order) - 1) / 2) * car_step

        x = base + cart_offset
        y = r["median"]

        ax.plot(
            x,
            y,
            marker=cart_markers[r["cart"]],
            linestyle="None",
            color=color,
            markersize=7
        )

        ax.errorbar(
            x,
            y,
            yerr=[[y - r["ci_low"]], [r["ci_high"] - y]],
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=3
        )

        pos.setdefault(r["timing"], {})[r["cart"]] = (x, r["ci_high"])

    for j, t in enumerate(order):
        if t not in pos or "CART3" not in pos[t] or "CART4" not in pos[t]:
            continue

        x1, _ = pos[t]["CART3"]
        x2, _ = pos[t]["CART4"]

        yb = y_base_bracket + j * gap

        p = float(
            pvals.loc[
                (pvals["dil_group"] == d)
                & (pvals["timing"] == t),
                "p_value"
            ].values[0]
        )

        add_bracket_with_p(ax, x1, x2, yb, h, format_p(p))

    ax.set_title(f"dil_group: {d}")
    ax.set_xticks([x_base[t] for t in order])
    ax.set_xticklabels(order)
    ax.set_xlabel("Timing")
    ax.set_ylim(bottom=1, top=y_base_bracket + (len(order) - 1) * gap + h * 3)
    ax.set_xlim(-0.5, x_base[order[-1]] + 0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel(f"{ycol} (median ± 95% bootstrap CI)")

handles = [
    Line2D(
        [0], [0],
        marker=cart_markers[c],
        linestyle="None",
        color=color,
        label=c,
        markersize=7
    )
    for c in cart_order
]

axes[-1].legend(handles=handles, title="CAR", frameon=False, loc="upper center")

# summary_plot.to_csv(os.path.join(SAVE_FOLDER, f'Fig2_intensity_stage_matureOnly_{expression}_{particle}_by_density.csv'), index=False)

plt.tight_layout()
# plt.savefig(os.path.join(SAVE_FOLDER, f'Fig2_intensity_stage_matureOnly_{expression}_{particle}_by_density.png'), dpi=600)
# plt.savefig(os.path.join(SAVE_FOLDER, f'Fig2_intensity_stage_matureOnly_{expression}_{particle}_by_density.pdf'), dpi=600)

plt.show()


#%% Intensities fold-change panel (matures only) + bootstrap CI + permutation tests
expression = "High exp"
particle = "CD19"
ycol = "median_intensity"

intensities = pd.read_csv(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\all_cotracks_velocities&directionality_stats_correctedtime.csv'
)

df = intensities[intensities.particle == particle].copy()

df = df[df["timing"].notna()].copy()
df = df[df["category"] == 1].copy()

df = add_common_columns(
    df,
    condition_col="condition",
    run_col="run",
    category_col="category"
)

df = df[
    (df["expr"] == expression)
    & df["dil_group"].notna()
].copy()

df["dil_group"] = (
    df["dil_group"]
    .astype(str)
    .str.strip()
    .str.lower()
)

order = ["pre", "during"]
df = df[df["timing"].isin(order)].copy()

df = df[df[ycol].notna() & (df[ycol] > 0)].copy()

index_cols = ["dil_group", "colocID", "cart"]
if "run" in df.columns:
    index_cols = ["dil_group", "run", "colocID", "cart"]

wide = (
    df.pivot_table(
        index=index_cols,
        columns="timing",
        values=ycol,
        aggfunc="first",
    )
    .reset_index()
)

wide = wide.dropna(subset=["pre", "during"]).copy()
wide["fold_change"] = wide["during"] / wide["pre"]
wide = wide[wide["fold_change"].notna() & (wide["fold_change"] > 0)].copy()
wide["log_fc"] = np.log(wide["fold_change"])

dil_order = ["sparse", "intermediate", "dense"]
dil_order = [
    d for d in dil_order
    if d in wide["dil_group"].dropna().unique()
]

summary = (
    wide.groupby(["dil_group", "cart"])["log_fc"]
    .apply(lambda s: bootstrap_ci_median(s.values))
)

summary = summary.apply(pd.Series)
summary.columns = ["median_log", "ci_low_log", "ci_high_log"]
summary = summary.reset_index()

summary["median"] = np.exp(summary["median_log"])
summary["ci_low"] = np.exp(summary["ci_low_log"])
summary["ci_high"] = np.exp(summary["ci_high_log"])

summary = summary.drop(columns=["median_log", "ci_low_log", "ci_high_log"])

p_within = []

for d in dil_order:
    for c in ["CART3", "CART4"]:
        vals = wide.loc[
            (wide["dil_group"] == d)
            & (wide["cart"] == c),
            "log_fc"
        ].values

        if len(vals) < 2:
            p = np.nan
        else:
            p = signflip_permutation_pvalue(
                vals,
                n_perm=10000,
                seed=42,
                stat="median",
                sides="two-sided"
            )

        p_within.append({
            "dil_group": d,
            "cart": c,
            "p_within": p
        })

p_within = pd.DataFrame(p_within)

summary = summary.merge(
    p_within,
    on=["dil_group", "cart"],
    how="left"
)

p_between = []

for d in dil_order:
    g1 = wide.loc[
        (wide["dil_group"] == d)
        & (wide["cart"] == "CART3"),
        "log_fc"
    ].values

    g2 = wide.loc[
        (wide["dil_group"] == d)
        & (wide["cart"] == "CART4"),
        "log_fc"
    ].values

    if len(g1) < 2 or len(g2) < 2:
        p = np.nan
    else:
        p = permutation_test_between_groups(
            g1,
            g2,
            n_perm=10000,
            seed=42,
            stat="median"
        )

    p_between.append({
        "dil_group": d,
        "p_between": p
    })

p_between = pd.DataFrame(p_between)

cart_order = ["CART3", "CART4"]
cart_markers = {"CART3": "s", "CART4": "o"}

fig, axes = plt.subplots(
    1,
    len(dil_order),
    figsize=(5.2 * len(dil_order), 3.8),
    sharey=True
)

if len(dil_order) == 1:
    axes = [axes]

color = "#2ecc71"
x_base = {c: i for i, c in enumerate(cart_order)}

all_ci_high = summary["ci_high"].max()
y_pad = all_ci_high * 0.06 if np.isfinite(all_ci_high) else 0.1
yb = all_ci_high * 1.18 if np.isfinite(all_ci_high) else 1.2
h = all_ci_high * 0.06 if np.isfinite(all_ci_high) else 0.1
top = all_ci_high * 1.35 if np.isfinite(all_ci_high) else 2

for ax, d in zip(axes, dil_order):

    sub_summary = summary[summary["dil_group"] == d].copy()
    pos = {}

    for _, r in sub_summary.iterrows():
        if r["cart"] not in x_base:
            continue

        x = x_base[r["cart"]]
        y = r["median"]

        ax.plot(
            x,
            y,
            marker=cart_markers[r["cart"]],
            linestyle="None",
            color=color,
            markersize=7
        )

        ax.errorbar(
            x,
            y,
            yerr=[[y - r["ci_low"]], [r["ci_high"] - y]],
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=3
        )

        pos[r["cart"]] = (x, r["ci_high"], r.get("p_within", np.nan))

    ax.axhline(1, color="gray", linestyle="--", linewidth=1)

    for c in cart_order:
        if c not in pos:
            continue

        x, ci_hi, p = pos[c]

        ax.text(
            x,
            ci_hi + y_pad,
            format_p(p),
            ha="center",
            va="bottom",
            fontsize=9,
            color="black"
        )

    if "CART3" in pos and "CART4" in pos:
        x1, _, _ = pos["CART3"]
        x2, _, _ = pos["CART4"]

        p = float(
            p_between.loc[
                p_between["dil_group"] == d,
                "p_between"
            ].values[0]
        )

        add_bracket_with_p(ax, x1, x2, yb, h, format_p(p))

    ax.set_title(f"dil_group: {d}")
    ax.set_xticks([x_base[c] for c in cart_order])
    ax.set_xticklabels(cart_order)
    ax.set_xlabel("CAR")
    ax.set_xlim(-0.5, len(cart_order) - 0.5)
    ax.set_ylim(bottom=1, top=top)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Intensity fold-change (during / pre)\n(median ± 95% bootstrap CI)")

handles = [
    Line2D(
        [0], [0],
        marker=cart_markers[c],
        linestyle="None",
        color=color,
        label=c,
        markersize=7
    )
    for c in cart_order
]

axes[-1].legend(handles=handles, title="CAR", frameon=False, loc="upper right")

summary.to_csv(os.path.join(SAVE_FOLDER, f'Fig2_intensityRatio_stage_matureOnly_{expression}_{particle}_by_density.csv'), index=False)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_FOLDER, f'Fig2_intensityRatio_stage_matureOnly_{expression}_{particle}_by_density.png'), dpi=600)
plt.savefig(os.path.join(SAVE_FOLDER, f'Fig2_intensityRatio_stage_matureOnly_{expression}_{particle}_by_density.pdf'), dpi=600)
plt.show()
#%% Intensity zap70/CD19 ratio — Panel 1 with Welch t-test
expression = "Low exp"
ycol = "median_intensity"

intensities = pd.read_csv(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\all_cotracks_velocities&directionality_stats_correctedtime.csv'
)

df = intensities[intensities["timing"].notna()].copy()
df = df[df["category"] == 1].copy()

df = add_common_columns(
    df,
    condition_col="condition",
    run_col="run",
    category_col="category"
)

df = df[
    (df["expr"] == expression)
    & df["dil_group"].notna()
].copy()

df["dil_group"] = (
    df["dil_group"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df = df[df["particle"].isin(["Zap70", "CD19"])].copy()

order = ["pre", "during"]
df = df[df["timing"].isin(order)].copy()

key_cols = ["run", "cell_id", "colocID", "timing"]

wide = (
    df.pivot_table(
        index=key_cols,
        columns="particle",
        values=ycol,
        aggfunc="first"
    )
    .reset_index()
)

wide = wide.dropna(subset=["Zap70", "CD19"]).copy()
wide = wide[wide["CD19"] > 0].copy()

wide["zap70_over_cd19"] = wide["Zap70"] / wide["CD19"]
wide["log_ratio"] = np.log(wide["zap70_over_cd19"])

meta = (
    df.loc[df["particle"] == "Zap70", key_cols + ["cart", "dil_group"]]
    .drop_duplicates(subset=key_cols)
)

wide = wide.merge(meta, on=key_cols, how="left")
wide = wide[wide["cart"].isin(["CART3", "CART4"])].copy()

dil_order = ["sparse", "intermediate", "dense"]
dil_order = [
    d for d in dil_order
    if d in wide["dil_group"].dropna().unique()
]

tmp = (
    wide.groupby(["dil_group", "timing", "cart"])["zap70_over_cd19"]
    .apply(lambda s: geometric_mean_confidence_interval(s.values))
)

summary = tmp.apply(pd.Series)
summary.columns = ["gmean", "ci_low", "ci_high"]
summary = summary.reset_index()

pvals = []

for d in dil_order:
    for t in order:
        sub = wide[
            (wide["dil_group"] == d)
            & (wide["timing"] == t)
        ]

        g1 = sub.loc[sub["cart"] == "CART3", "log_ratio"].values
        g2 = sub.loc[sub["cart"] == "CART4", "log_ratio"].values

        if len(g1) < 2 or len(g2) < 2:
            p = np.nan
        else:
            _, p = stats.ttest_ind(g1, g2, equal_var=False)

        pvals.append({
            "dil_group": d,
            "timing": t,
            "p_value": p
        })

pvals = pd.DataFrame(pvals)

timing_spacing = 0.55
x_base = {t: i * timing_spacing for i, t in enumerate(order)}

cart_order = ["CART3", "CART4"]
cart_to_i = {c: i for i, c in enumerate(cart_order)}
cart_markers = {"CART3": "s", "CART4": "o"}
car_step = 0.18

fig, axes = plt.subplots(
    1,
    len(dil_order),
    figsize=(5.2 * len(dil_order), 3.8),
    sharey=True
)

if len(dil_order) == 1:
    axes = [axes]

color = "#2ecc71"

all_ci_high = summary["ci_high"].max()
h = all_ci_high * 0.05
gap = all_ci_high * 0.08
y_base_bracket = all_ci_high * 1.10

for ax, d in zip(axes, dil_order):

    sub_summary = summary[summary["dil_group"] == d].copy()
    pos = {}

    for _, r in sub_summary.iterrows():
        if r["timing"] not in x_base or r["cart"] not in cart_to_i:
            continue

        base = x_base[r["timing"]]
        cart_i = cart_to_i[r["cart"]]
        cart_offset = (cart_i - (len(cart_order) - 1) / 2) * car_step

        x = base + cart_offset
        y = r["gmean"]

        ax.plot(
            x,
            y,
            marker=cart_markers[r["cart"]],
            linestyle="None",
            color=color,
            markersize=7
        )

        ax.errorbar(
            x,
            y,
            yerr=[[y - r["ci_low"]], [r["ci_high"] - y]],
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=3
        )

        pos.setdefault(r["timing"], {})[r["cart"]] = (x, r["ci_high"])

    for j, t in enumerate(order):
        if t not in pos or "CART3" not in pos[t] or "CART4" not in pos[t]:
            continue

        x1, _ = pos[t]["CART3"]
        x2, _ = pos[t]["CART4"]

        yb = y_base_bracket + j * gap

        p = float(
            pvals.loc[
                (pvals["dil_group"] == d)
                & (pvals["timing"] == t),
                "p_value"
            ].values[0]
        )

        add_bracket_with_p(ax, x1, x2, yb, h, format_p(p))

    ax.set_title(f"dil_group: {d}")
    ax.set_xticks([x_base[t] for t in order])
    ax.set_xticklabels(order)
    ax.set_xlabel("Timing")
    ax.set_ylim(bottom=0, top=y_base_bracket + (len(order) - 1) * gap + h * 3)
    ax.set_xlim(-0.5, x_base[order[-1]] + 0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Zap70 / CD19 (GM ± 95% t-CI on log)")

handles = [
    Line2D(
        [0], [0],
        marker=cart_markers[c],
        linestyle="None",
        color=color,
        label=c,
        markersize=7
    )
    for c in cart_order
]

axes[-1].legend(handles=handles, title="CAR", frameon=False, loc="upper center")

# summary.to_csv(os.path.join(SAVE_FOLDER, f'Fig2_intensity_Zap70overCD19_stage_matureOnly_{expression}_by_density.csv'), index=False)

plt.tight_layout()
# plt.savefig(os.path.join(SAVE_FOLDER, f'Fig2_intensity_Zap70overCD19_stage_matureOnly_{expression}_by_density.png'), dpi=600)
# plt.savefig(os.path.join(SAVE_FOLDER, f'Fig2_intensity_Zap70overCD19_stage_matureOnly_{expression}_by_density.pdf'), dpi=600)
plt.show()


#%% Zap70/CD19 ratio-of-ratios (paired change) — Δ panel with stats, separated by density

expression = "Low exp"
ycol = "median_intensity"

intensities = pd.read_csv(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\all_cotracks_velocities&directionality_stats_correctedtime.csv'
)

df = intensities[intensities["timing"].notna()].copy()
df = df[df["category"] == 1].copy()

df = add_common_columns(
    df,
    condition_col="condition",
    run_col="run",
    category_col="category"
)

df = df[
    (df["expr"] == expression)
    & df["dil_group"].notna()
].copy()

df["dil_group"] = (
    df["dil_group"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df = df[df["particle"].isin(["Zap70", "CD19"])].copy()

order = ["pre", "during"]
df = df[df["timing"].isin(order)].copy()

key_cols = ["run", "cell_id", "colocID", "timing"]

wide = (
    df.pivot_table(
        index=key_cols,
        columns="particle",
        values=ycol,
        aggfunc="first"
    )
    .reset_index()
)

wide = wide.dropna(subset=["Zap70", "CD19"]).copy()
wide = wide[wide["CD19"] > 0].copy()

wide["ratio"] = wide["Zap70"] / wide["CD19"]
wide = wide[wide["ratio"].notna() & (wide["ratio"] > 0)].copy()
wide["log_ratio"] = np.log(wide["ratio"])

meta = (
    df.loc[df["particle"] == "Zap70", key_cols + ["cart", "dil_group"]]
    .dropna(subset=["cart", "dil_group"])
    .drop_duplicates(subset=key_cols)
)

wide = wide.merge(meta, on=key_cols, how="left")

missing_cart = wide["cart"].isna()
if missing_cart.any():
    meta_cd19 = (
        df.loc[df["particle"] == "CD19", key_cols + ["cart", "dil_group"]]
        .dropna(subset=["cart", "dil_group"])
        .drop_duplicates(subset=key_cols)
    )

    recovered = (
        wide.loc[missing_cart, key_cols]
        .merge(meta_cd19, on=key_cols, how="left")
    )

    wide.loc[missing_cart, "cart"] = recovered["cart"].values
    wide.loc[missing_cart, "dil_group"] = recovered["dil_group"].values

wide = wide[
    wide["cart"].isin(["CART3", "CART4"])
    & wide["dil_group"].notna()
].copy()

pair_cols = ["run", "cell_id", "colocID", "cart", "dil_group"]

wide2 = (
    wide.pivot_table(
        index=pair_cols,
        columns="timing",
        values="log_ratio",
        aggfunc="first"
    )
    .reset_index()
)

wide2 = wide2.dropna(subset=["pre", "during"]).copy()
wide2["delta_log_ratio"] = wide2["during"] - wide2["pre"]
wide2["ratio_of_ratios"] = np.exp(wide2["delta_log_ratio"])


dil_order = ["sparse", "intermediate", "dense"]
dil_order = [
    d for d in dil_order
    if d in wide2["dil_group"].dropna().unique()
]

tmp = (
    wide2.groupby(["dil_group", "cart"])["ratio_of_ratios"]
    .apply(lambda s: geometric_mean_confidence_interval(s.values))
)

summary = tmp.apply(pd.Series)
summary.columns = ["gmean", "ci_low", "ci_high"]
summary = summary.reset_index()

p_within = []

for d in dil_order:
    for c in ["CART3", "CART4"]:
        vals = wide2.loc[
            (wide2["dil_group"] == d)
            & (wide2["cart"] == c),
            "delta_log_ratio"
        ].values

        if len(vals) < 2:
            p = np.nan
        else:
            _, p = stats.ttest_1samp(vals, 0)

        p_within.append({
            "dil_group": d,
            "cart": c,
            "p_within": p
        })

p_within = pd.DataFrame(p_within)

summary = summary.merge(
    p_within,
    on=["dil_group", "cart"],
    how="left"
)

p_between = []

for d in dil_order:
    g1 = wide2.loc[
        (wide2["dil_group"] == d)
        & (wide2["cart"] == "CART3"),
        "delta_log_ratio"
    ].values

    g2 = wide2.loc[
        (wide2["dil_group"] == d)
        & (wide2["cart"] == "CART4"),
        "delta_log_ratio"
    ].values

    if len(g1) < 2 or len(g2) < 2:
        p = np.nan
    else:
        _, p = stats.ttest_ind(g1, g2, equal_var=False)

    p_between.append({
        "dil_group": d,
        "p_between": p
    })

p_between = pd.DataFrame(p_between)

cart_order = ["CART3", "CART4"]
cart_markers = {"CART3": "s", "CART4": "o"}

fig, axes = plt.subplots(
    1,
    len(dil_order),
    figsize=(5.2 * len(dil_order), 3.8),
    sharey=True
)

if len(dil_order) == 1:
    axes = [axes]

color = "#2ecc71"
x_base = {c: i for i, c in enumerate(cart_order)}

all_ci_high = summary["ci_high"].max()
y_pad = all_ci_high * 0.06 if np.isfinite(all_ci_high) else 0.1
yb = all_ci_high * 1.18 if np.isfinite(all_ci_high) else 1.2
h = all_ci_high * 0.06 if np.isfinite(all_ci_high) else 0.1
top = all_ci_high * 1.35 if np.isfinite(all_ci_high) else 2

for ax, d in zip(axes, dil_order):

    sub_summary = summary[summary["dil_group"] == d].copy()
    pos = {}

    for _, r in sub_summary.iterrows():
        if r["cart"] not in x_base:
            continue

        x = x_base[r["cart"]]
        y = r["gmean"]

        ax.plot(
            x,
            y,
            marker=cart_markers[r["cart"]],
            linestyle="None",
            color=color,
            markersize=7
        )

        ax.errorbar(
            x,
            y,
            yerr=[[y - r["ci_low"]], [r["ci_high"] - y]],
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=3
        )

        pos[r["cart"]] = (x, r["ci_high"], r.get("p_within", np.nan))

    ax.axhline(1, color="gray", linestyle="--", linewidth=1)

    for c in cart_order:
        if c not in pos:
            continue

        x, ci_hi, p = pos[c]

        ax.text(
            x,
            ci_hi + y_pad,
            format_p(p),
            ha="center",
            va="bottom",
            fontsize=9,
            color="black"
        )

    if "CART3" in pos and "CART4" in pos:
        x1, _, _ = pos["CART3"]
        x2, _, _ = pos["CART4"]

        p = float(
            p_between.loc[
                p_between["dil_group"] == d,
                "p_between"
            ].values[0]
        )

        add_bracket_with_p(ax, x1, x2, yb, h, format_p(p))

    ax.set_title(f"dil_group: {d}")
    ax.set_xticks([x_base[c] for c in cart_order])
    ax.set_xticklabels(cart_order)
    ax.set_xlabel("CAR")
    ax.set_xlim(-0.5, len(cart_order) - 0.5)
    ax.set_ylim(bottom=0, top=top)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Fold change Zap70/CD19 ratio (during / pre)\n(GM ± 95% t-CI on log)")

handles = [
    Line2D(
        [0], [0],
        marker=cart_markers[c],
        linestyle="None",
        color=color,
        label=c,
        markersize=7
    )
    for c in cart_order
]

axes[-1].legend(handles=handles, title="CAR", frameon=False, loc="lower right")

summary.to_csv(os.path.join(SAVE_FOLDER, f'Fig2_Zap70overCD19Ratio_stage_matureOnly_{expression}_by_density.csv'), index=False)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_FOLDER, f'Fig2_Zap70overCD19Ratio_matureOnly_{expression}_by_density.png'), dpi=600)
plt.savefig(os.path.join(SAVE_FOLDER, f'Fig2_Zap70overCD19Ratio_matureOnly_{expression}_by_density.pdf'), dpi=600)
plt.show()

#%% intensity disorganized versus maturing cells + compare the 2 CARs within each group, separated by density

expression = "High exp"

title_dict = {
    "total_mean_cd": "Intensity CD19",
    "total_mean_zap": "Intensity Zap70",
    "total_mean_ratio": "Zap70 / CD19 intensity ratio"
}

name = {
    "total_mean_cd": "CD19",
    "total_mean_zap": "Zap70",
    "total_mean_ratio": "Zap70overCD19"
}

col_to_plot = "total_mean_zap"

intensities = pd.read_csv(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\Integratedintensity_non-matureVSMature_summary.csv'
)

# Keep the rest of the plotting code unchanged
intensities["mature"] = intensities["maturation_group"].map({
    "non-maturing": "not-mature",
    "maturing": "mature"
})

intensities["dil_group"] = (
    intensities["dil"]
    .astype(str)
    .str.strip()
    .map(DIL_GROUP_MAP)
)


int_to_plot = intensities[
    (intensities["expr"] == expression)
    & (intensities[col_to_plot] > 0)
    & intensities["dil_group"].notna()
    & (
        (col_to_plot != "total_mean_ratio")
        | (intensities["total_mean_cd"] > 20)
    )
].copy()

summary = (
    int_to_plot.dropna(subset=[col_to_plot, "CART", "mature", "dil_group"])
    .groupby(["dil_group", "CART", "mature"])[col_to_plot]
    .apply(lambda s: geometric_mean_confidence_interval(s.values))
    .apply(pd.Series)
    .reset_index()
)
summary.columns = ["dil_group", "CART", "mature", "gmean", "ci_low", "ci_high"]

n_summary = (
    int_to_plot.dropna(subset=[col_to_plot, "CART", "mature", "dil_group"])
    .groupby(["dil_group", "CART", "mature"])
    .size()
    .reset_index(name="n")
)

summary = summary.merge(
    n_summary,
    on=["dil_group", "CART", "mature"],
    how="left"
)

summary = summary[["dil_group", "CART", "mature", "n", "gmean", "ci_low", "ci_high"]]

dil_order = ["sparse", "intermediate", "dense"]
dil_order = [
    d for d in dil_order
    if d in summary["dil_group"].dropna().unique()
]

cart_order = sorted(summary["CART"].unique())
mature_order = ["not-mature", "mature"]

if len(cart_order) != 2:
    raise ValueError(f"This version expects exactly 2 CARs, but found {len(cart_order)}: {cart_order}")

car1, car2 = cart_order

pvals_between_cars = []

for d in dil_order:
    for mature_state in mature_order:
        sub = int_to_plot[
            (int_to_plot["dil_group"] == d)
            & (int_to_plot["mature"] == mature_state)
        ]

        g1 = np.log(sub.loc[sub["CART"] == car1, col_to_plot].values)
        g2 = np.log(sub.loc[sub["CART"] == car2, col_to_plot].values)

        if len(g1) < 2 or len(g2) < 2:
            p = np.nan
        else:
            _, p = stats.ttest_ind(g1, g2, equal_var=False)

        pvals_between_cars.append({
            "dil_group": d,
            "mature": mature_state,
            "p_value": p
        })

pvals_between_cars = pd.DataFrame(pvals_between_cars)

pvals_between_maturation = []

for d in dil_order:
    for cart in cart_order:
        sub = int_to_plot[
            (int_to_plot["dil_group"] == d)
            & (int_to_plot["CART"] == cart)
        ]

        g1 = np.log(sub.loc[sub["mature"] == "not-mature", col_to_plot].values)
        g2 = np.log(sub.loc[sub["mature"] == "mature", col_to_plot].values)

        if len(g1) < 2 or len(g2) < 2:
            p = np.nan
        else:
            _, p = stats.ttest_ind(g1, g2, equal_var=False)

        pvals_between_maturation.append({
            "dil_group": d,
            "CART": cart,
            "p_value": p
        })

pvals_between_maturation = pd.DataFrame(pvals_between_maturation)

fig, axes = plt.subplots(
    1,
    len(dil_order),
    figsize=(6.5 * len(dil_order), 4.8),
    sharey=True
)

if len(dil_order) == 1:
    axes = [axes]

x_base = {c: i for i, c in enumerate(cart_order)}
offset_step = 0.25
mature_to_i = {m: i for i, m in enumerate(mature_order)}

palette = {
    "not-mature": "#b23a8a",
    "mature": "#1A7A42"
}

box_width = 0.18

global_top_data = int_to_plot[col_to_plot].dropna().quantile(0.98)

h_global = global_top_data * 0.01

y_within_car_1_global = global_top_data * 0.8
y_within_car_2_global = global_top_data * 0.85
y_between_cars_notmat_global = global_top_data * 0.9
y_between_cars_mat_global = global_top_data * 0.95

global_ylim_top = global_top_data

for ax, d in zip(axes, dil_order):

    sub_data = int_to_plot[int_to_plot["dil_group"] == d].copy()
    pos = {}

    for cart in cart_order:
        for mature_state in mature_order:
            sub = sub_data[
                (sub_data["CART"] == cart)
                & (sub_data["mature"] == mature_state)
            ][col_to_plot].dropna()

            if len(sub) == 0:
                continue

            base = x_base[cart]
            offset = (mature_to_i[mature_state] - (len(mature_order) - 1) / 2) * offset_step
            x = base + offset

            bp = ax.boxplot(
                sub,
                positions=[x],
                widths=box_width,
                patch_artist=True,
                showfliers=False
            )

            for box in bp["boxes"]:
                box.set(facecolor=palette[mature_state], alpha=0.6)

            for element in ["whiskers", "caps", "medians"]:
                for item in bp[element]:
                    item.set(color="black")

            jitter = np.random.normal(0, 0.02, size=len(sub))

            ax.scatter(
                np.full(len(sub), x) + jitter,
                sub,
                color="black",
                s=15,
                alpha=0.4,
                zorder=3
            )

            pos.setdefault(cart, {})[mature_state] = x

    for i, row in pvals_between_maturation[
        pvals_between_maturation["dil_group"] == d
    ].reset_index(drop=True).iterrows():

        cart = row["CART"]
        p = row["p_value"]

        if pd.isna(p):
            continue
        if cart not in pos:
            continue
        if "not-mature" not in pos[cart] or "mature" not in pos[cart]:
            continue

        x1 = pos[cart]["not-mature"]
        x2 = pos[cart]["mature"]

        yb = y_within_car_1_global if i == 0 else y_within_car_2_global
        add_bracket_with_p(ax, x1, x2, yb, h_global, format_p(p))

    for _, row in pvals_between_cars[
        pvals_between_cars["dil_group"] == d
    ].iterrows():

        mature_state = row["mature"]
        p = row["p_value"]

        if pd.isna(p):
            continue
        if car1 not in pos or car2 not in pos:
            continue
        if mature_state not in pos[car1] or mature_state not in pos[car2]:
            continue

        x1 = pos[car1][mature_state]
        x2 = pos[car2][mature_state]

        if mature_state == "not-mature":
            yb = y_between_cars_notmat_global
        else:
            yb = y_between_cars_mat_global

        add_bracket_with_p(ax, x1, x2, yb, h_global, format_p(p))

    ax.set_title(f"dil_group: {d}")
    ax.set_xticks([x_base[c] for c in cart_order])
    ax.set_xticklabels(cart_order)
    ax.set_xlabel("CART")
    ax.set_xlim(-0.5, len(cart_order) - 0.5)
    ax.set_ylim(0, global_ylim_top)
    ax.ticklabel_format(style="plain", axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel(title_dict[col_to_plot])

handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        color=palette[m],
        label=str(m),
        markersize=7
    )
    for m in mature_order
]

axes[-1].legend(handles=handles, title="Maturation group", frameon=False)

summary.to_csv(os.path.join(SAVE_FOLDER, rf'1_integratedintensity_maturingVSnon-maturing_{name[col_to_plot]}_{expression}_allcomparisons_by_density.csv'), index=False)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_FOLDER, rf'1_integratedintensity_maturingVSnon-maturing_{name[col_to_plot]}_{expression}_allcomparisons_by_density.pdf'), dpi=600)
plt.savefig(os.path.join(SAVE_FOLDER, rf'1_integratedintensity_maturingVSnon-maturing_{name[col_to_plot]}_{expression}_allcomparisons_by_density.png'), dpi=600)

plt.show()