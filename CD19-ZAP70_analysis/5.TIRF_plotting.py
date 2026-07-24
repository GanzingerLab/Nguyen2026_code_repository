#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.lines import Line2D
from postSPIT import tirf_analysis as ta

#%% 
DIL_GROUP_MAP = {
    "50xdilutedCD19": "dense",
    "100xdilutedCD19": "dense",
    "500xdilutedCD19": "intermediate",
    "1000xdilutedCD19": "intermediate",
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

#%% Maturarion count plot - change expression on the first line
expression = "Low exp"
# Open the input data.
df_raw = pd.read_csv(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\maturation_count_withDil.csv')
raw = df_raw.copy()

# Grouping columns
raw = add_common_columns(raw, condition_col="condition", run_col="run", category_col="category")

# Keep only the selected expression level.
raw_low = raw[raw["expr"] == expression].copy()

# Count events per group and convert them to proportions.
counts = (
    raw_low.groupby(["cart", "dil_group", "category_label"], as_index=False)
           .size()
           .rename(columns={"size": "n"})
)
counts["proportion"] = (
    counts["n"] /
    counts.groupby(["cart", "dil_group"])["n"].transform("sum")
)
totals = (
    counts.groupby(["cart", "dil_group"], as_index=False)["n"]
          .sum()
)

# Build the plot.
fig, ax = plt.subplots(figsize=(8, 5))

group_order = ["sparse", "dense"]
cart_order = ["CART3", "CART4"]

x = np.arange(len(group_order))
width = 0.35
offsets = {"CART3": -width/2, "CART4": +width/2}

for cart in cart_order:
    #filter
    sub = counts[counts["cart"] == cart]
    #counts for the annotation
    sub_tot = (
        totals[totals["cart"] == cart]
        .set_index("dil_group")
        .reindex(group_order)
    )
    #make grouping table 
    pivot = (
        sub.pivot(index="dil_group", columns="category_label", values="proportion")
           .fillna(0)
           .reindex(group_order)
           .reindex(columns=CATEGORY_ORDER_ALL)
    )

    bottoms = np.zeros(len(group_order))

    for cat in ['matures', 'starts mature', 'never matures']:
        vals = pivot[cat].values
        ax.bar(
            x + offsets[cart],
            vals,
            width=width,
            bottom=bottoms,
            color=CATEGORY_COLORS[cat],
            edgecolor="black",
            linewidth=0.4,
            label=cat if cart == "CART3" else None
        )
        bottoms += vals

    # Add the CAR label and sample size above each bar.
    for i, dg in enumerate(group_order):
        n = sub_tot.loc[dg, "n"] if dg in sub_tot.index else np.nan
        if pd.notna(n):
            ax.text(
                x[i] + offsets[cart],
                1.05,
                f"{cart}\nn={int(n)}", # Show CAR type and sample size.
                # f"{cart}", # Alternative label showing only the CAR type.
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )

# Format axes and legend.
ax.set_xticks(x)
ax.set_xticklabels(group_order)
ax.set_ylabel("Proportion")
ax.set_ylim(0, 1.15)

ax.legend(title="Maturation category", bbox_to_anchor=(1.02, 0.5), loc="center left")

plt.tight_layout()
# plt.savefig(r'D:\Data\Chi_data\20250801_filtered\output\analysis2026\Fig1_maturation_count_LowExp.pdf', dpi=600)
plt.show()


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

# Exclude the intermediate dilution group.
plot_df = plot_df[plot_df["dil_group"] != "intermediate"].copy()
# group and make plotting table 
summary_wide = (
    plot_df.groupby(["cart", "transition", "category"])[ycol]
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
    col="cart",
    col_order=["CART3", "CART4"],
    sharey=True,
    height=4,
    aspect=1.2
)
g.map_dataframe(plot_points_with_ci)

# Add one legend for the full figure.
g.add_legend(title="category", label_order=CATEGORY_ORDER_ALL)

# Use simple facet titles.
g.set_titles(col_template="{col_name}")
plt.tight_layout()
# plt.savefig(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig1_directionality_HighExp.png', dpi=600)
# plt.savefig(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig1_directionality_HighExp.pdf', dpi=600)
plt.show()


#%% Directionality difference: permutatin test around 0 of the per-track difference of directionality
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

# Exclude the intermediate dilution group.
plot_df = plot_df[plot_df["dil_group"] != "intermediate"].copy()

# Calculate difference in directionality between during (loc1-loc2) and before localization (loc0-loc1)
needed_transitions = ["loc0-loc1", "loc1-loc2"]
plot_df = plot_df[plot_df["transition"].isin(needed_transitions)].copy()

plot_df["category_label"] = plot_df["category"].map(CATEGORY_MAP)

wide = (
    plot_df.pivot_table(
        index=["cart", "category_label", "colocID"],
        columns="transition",
        values=ycol,
        aggfunc="first",
    )
    .reset_index()
)

if not all(t in wide.columns for t in needed_transitions):
    raise KeyError(
        f"Missing one of {needed_transitions} in pivoted columns. "
        f"Found: {sorted([c for c in wide.columns if c not in ['cart','category_label','colocID']])}"
    )

wide["delta"] = wide["loc1-loc2"] - wide["loc0-loc1"]
delta_df = wide.dropna(subset=["delta"]).copy()

# Summarize with median and bootstrap confidence interval and make plotting table
summary_delta = (
    delta_df.groupby(["cart", "category_label"])["delta"]
    .apply(lambda s: bootstrap_ci_median(s.values))
    .reset_index(name="tmp")
)
summary_delta[["median", "ci_low", "ci_high"]] = pd.DataFrame(
    summary_delta["tmp"].tolist(), index=summary_delta.index
)
summary_delta = summary_delta.drop(columns="tmp")

# Test whether the per-track difference differs from zero for each CAR and category.
pvals = (
    delta_df.groupby(["cart", "category_label"])["delta"]
    .apply(lambda s: signflip_permutation_pvalue(
        s.values, n_perm=10000, seed=42, stat="median", sides="one-sided"
    ))
    .reset_index(name="p_value")
)
summary_delta = summary_delta.merge(pvals, on=["cart", "category_label"], how="left")
summary_delta["transition"] = "Δ (loc1-loc2 − loc0-loc1)"

# Plot the delta panel using the same visual style as above.
order = ["ΔDirectionality)"]

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

g = sns.FacetGrid(
    summary_delta,
    col="cart",
    col_order=["CART3", "CART4"],
    sharey=True,
    height=4,
    aspect=1.2
)
g.map_dataframe(plot_points_with_ci_delta)
g.add_legend(title="category", label_order=hue_order)
g.set_titles(col_template="{col_name}")
plt.tight_layout()
# plt.savefig(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig1_directionalityDIFF_HighExp.png',dpi=600)
# plt.savefig(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig1_directionalityDIFF_HighExp.pdf',dpi=600)
plt.show()


#%%velocities per maturation - Welch t-test
expression = "Low exp"
particle = 'Zap70'

# Load the tables and keep the same cleaning steps as before.
velocities = pd.read_csv(
    r'D:\Data\Chi_data\20250801_filtered\output\analysis2026\analysis_output\all_cotracks_velocities&directionality_correctedtime.csv'
)
velocities_stats = pd.read_csv(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\all_cotracks_velocities&directionality_stats_correctedtime.csv'
)

velocities_groups = velocities.groupby(['run', 'colocID', "particle"])
velocities_means = velocities_groups['velocity'].mean().reset_index()

right_unique = velocities_stats[['run','colocID','category', 'condition']].drop_duplicates(['run','colocID'])
velocities_means_maturation = velocities_means.merge(right_unique, on=['run','colocID'], how='left')

# Add grouping columns
velocities_means_maturation = add_common_columns(
    velocities_means_maturation,
    condition_col="condition",
    run_col="run",
    category_col="category"
)
#Filter the data (specific expression level, for cells that mature or never mature, fo a specific (defined above) protein, 
# and removing intermediate CD19 level)
plot_df = velocities_means_maturation[
    (velocities_means_maturation["expr"] == expression)
    & (velocities_means_maturation["category"] < 2)
    & (velocities_means_maturation["particle"] == particle)
    & (velocities_means_maturation["dil_group"] != "intermediate")
].copy()

ycol = "velocity"

#Filter out negative values, in case there is any, for geometric means.
plot_df = plot_df[plot_df[ycol].notna() & (plot_df[ycol] > 0)].copy()
#make plotting table
gm_ci = (
    plot_df.groupby(["cart", "category_label"])[ycol]
    .apply(lambda s: geometric_mean_confidence_interval(s.values))
)
summary = gm_ci.apply(pd.Series)
summary.columns = ["geo_mean", "ci_low", "ci_high"]
summary = summary.reset_index()

# Welch t-test on log-transformed velocity.
pvals = []
for cat in ["never matures", "matures"]:
    sub = plot_df[plot_df["category_label"] == cat].copy()
    g1 = np.log(sub.loc[sub["cart"] == "CART3", ycol].values)
    g2 = np.log(sub.loc[sub["cart"] == "CART4", ycol].values)

    if len(g1) < 2 or len(g2) < 2:
        p = np.nan
    else:
        _, p = stats.ttest_ind(g1, g2, equal_var=False)  # Welch
    pvals.append({"category_label": cat, "p_value": p})

pvals = pd.DataFrame(pvals)
summary = summary.merge(pvals, on="category_label", how="left")

# Plot the geometric means and confidence intervals, then add p-value brackets.
hue_order = ["never matures", "matures"]

cart_order = ["CART3", "CART4"]
cart_markers = {"CART3": "s", "CART4": "o"}  # square vs dot

plt.figure(figsize=(6, 4))
ax = plt.gca()

cart_to_i = {c: i for i, c in enumerate(cart_order)}
hue_to_j = {h: j for j, h in enumerate(hue_order)}

# Control the horizontal spacing between CARs and categories.
cart_step = 0.24
cat_step  = 0.10

pos = {}  

for _, r in summary.iterrows():
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
        x, y,
        marker=cart_markers.get(r["cart"], "o"),
        linestyle="None",
        markerfacecolor=CATEGORY_COLORS[r["category_label"]],
        markeredgecolor=CATEGORY_COLORS[r["category_label"]],
        markersize=7,
    )

    ax.errorbar(
        x, y,
        yerr=[[y - r["ci_low"]], [r["ci_high"] - y]],
        fmt="none", ecolor="black", elinewidth=1, capsize=3
    )

    pos.setdefault(r["category_label"], {})[r["cart"]] = (x, r["ci_high"])

cat_handles = [
    Line2D([0], [0], marker="o", linestyle="None",
           markerfacecolor=CATEGORY_COLORS[c], markeredgecolor=CATEGORY_COLORS[c],
           label=c, markersize=7)
    for c in hue_order
]

ax.set_xticks([
    (cart_to_i[c] - (len(cart_order) - 1) / 2) * cart_step
    for c in cart_order
])
ax.set_xticklabels(cart_order)
ax.set_xlabel("CAR")
ax.set_ylabel(f"Velocity {particle} (GM ± 95% t-CI on log)")

leg1 = ax.legend(handles=cat_handles, title="Category",
                 loc="lower right", frameon=False)
ax.add_artist(leg1)

# Draw one bracket and p-value per category, stacked to avoid overlap.
all_ci_high = summary["ci_high"].max()
y_base = all_ci_high * 0.95
h = all_ci_high * 0.05
gap = all_ci_high * 0.12

for j, cat in enumerate(hue_order):
    if cat not in pos or "CART3" not in pos[cat] or "CART4" not in pos[cat]:
        continue

    x1, _ = pos[cat]["CART3"]
    x2, _ = pos[cat]["CART4"]
    y_bracket = y_base + j * gap

    p = float(pvals.loc[pvals["category_label"] == cat, "p_value"].values[0])
    add_bracket_with_p(ax, x1, x2, y_bracket, h, format_p(p))

ax.set_ylim(
    bottom=0,
    top=y_base + (len(hue_order) - 1) * gap + h * 2.5
)

plt.tight_layout()
# plt.savefig(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig1_velocity_maturation_LowExp_Zap70.png',dpi=600)
# plt.savefig(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig1_velocity_maturation_LowExp_Zap70.pdf',dpi=600)
plt.show()


#%% velocities per stage - mature only
expression = "Low exp"
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
    & (velocities_clean["dil_group"] != "intermediate")
    & velocities_clean["cart"].notna()
].copy()

# Summarize by timing and CAR.
ycol = "avg_speed"
gm_ci = (
    plot_df.groupby(["timing", "cart"])[ycol]
    .apply(lambda s: geometric_mean_confidence_interval(s.values))
)
summary = gm_ci.apply(pd.Series)
summary.columns = ["geo_mean", "ci_low", "ci_high"]
summary = summary.reset_index()

# Compare CART3 and CART4 within each timing with a Welch t-test on log-transformed speed.
order = ["pre", "during"]

pvals = []
for t in order:
    sub = plot_df[plot_df["timing"] == t].copy()
    g1 = np.log(sub.loc[sub["cart"] == "CART3", ycol].values)
    g2 = np.log(sub.loc[sub["cart"] == "CART4", ycol].values)
    if len(g1) < 2 or len(g2) < 2:
        p = np.nan
    else:
        _, p = stats.ttest_ind(g1, g2, equal_var=False)  # Welch
    pvals.append({"timing": t, "p_value": p})
pvals = pd.DataFrame(pvals)

# Plot the summary values.
timing_spacing = 0.55
x_base = {t: i * timing_spacing for i, t in enumerate(order)}

cart_order = ["CART3", "CART4"]
cart_to_i = {c: i for i, c in enumerate(cart_order)}
cart_markers = {"CART3": "s", "CART4": "o"}

car_step = 0.18

plt.figure(figsize=(5.2, 3.8))
ax = plt.gca()
color = "#2ecc71"

pos = {} 

for _, r in summary.iterrows():
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
        fmt="none", ecolor='black', elinewidth=1, capsize=3
    )

    pos.setdefault(r["timing"], {})[r["cart"]] = (x, r["ci_high"])

# ----------------------------
# ADD: bracket + p-value per timing (pre and during)
# ----------------------------
all_ci_high = summary["ci_high"].max()
h = all_ci_high * 0.05
gap = all_ci_high * 0.08
y_base_bracket = all_ci_high * 1.10

for j, t in enumerate(order):
    if t not in pos or "CART3" not in pos[t] or "CART4" not in pos[t]:
        continue

    x1, _ = pos[t]["CART3"]
    x2, _ = pos[t]["CART4"]
    yb = y_base_bracket + j * gap

    p = float(pvals.loc[pvals["timing"] == t, "p_value"].values[0])
    add_bracket_with_p(ax, x1, x2, yb, h, format_p(p))

# Format the axes.
ax.set_xticks([x_base[t] for t in order])
ax.set_xticklabels(order)
ax.set_xlabel("Timing")
ax.set_ylabel("Velocity (GM ± 95% t-CI on log)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_ylim(bottom=0, top=y_base_bracket + (len(order) - 1) * gap + h * 3)

handles = [
    Line2D([0], [0], marker=cart_markers[c], linestyle="None",
           color=color, label=c, markersize=7)
    for c in cart_order
]
ax.legend(handles=handles, title="CAR", frameon=False, loc="lower right")

plt.tight_layout()
ax.set_xlim(-0.5, len(cart_order) - 1)
# plt.savefig(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig2_velocity_stage_LowExp_CD19.png', dpi=600)
# plt.savefig(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig2_velocity_stage_LowExp_CD19.pdf', dpi=600)
plt.show()


#%% paired difference of velocities (during − pre) per track for statsitics of decrase. T-test on log scale for between difference of 1, and welch t-test for between CARs.
expression = "Low exp"

# Load the input table.
velocities = pd.read_csv(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\all_cotracks_velocities&directionality_stats_correctedtime.csv'
)

# Filter
velocities = velocities[velocities.particle == 'CD19']

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
    & (velocities_clean["dil_group"] != "intermediate")
    & (velocities_clean["cart"].notna())
].copy()

ycol = "avg_speed"

index_cols = ["colocID", "cart"]
if "run" in plot_df.columns:
    index_cols = ["run", "colocID", "cart"]

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

# summarize (GM ± 95% t-CI on log)
gm_ci = (
    wide.groupby(["cart"])["fold_change"]
    .apply(lambda s: geometric_mean_confidence_interval(s.values))
)
summary = gm_ci.apply(pd.Series)
summary.columns = ["geo_mean", "ci_low", "ci_high"]
summary = summary.reset_index()

# Run the statistical tests.
#inside group
p_within = []
for c in ["CART3", "CART4"]:
    vals = wide.loc[wide["cart"] == c, "log_fc"].values
    if len(vals) < 2:
        p = np.nan
    else:
        _, p = stats.ttest_1samp(vals, 0)  
    p_within.append({"cart": c, "p_within": p})
p_within = pd.DataFrame(p_within)
summary = summary.merge(p_within, on="cart", how="left")
#between groups
g1 = wide.loc[wide["cart"] == "CART3", "log_fc"].values
g2 = wide.loc[wide["cart"] == "CART4", "log_fc"].values
if len(g1) < 2 or len(g2) < 2:
    p_between = np.nan
else:
    _, p_between = stats.ttest_ind(g1, g2, equal_var=False)  # Welch on log fold-change

# Plot 
cart_order = ["CART3", "CART4"]
cart_markers = {"CART3": "s", "CART4": "o"}

plt.figure(figsize=(5.2, 3.8))
ax = plt.gca()
color = "#2ecc71"

x_base = {c: i for i, c in enumerate(cart_order)}
pos = {}  

for _, r in summary.iterrows():
    if r["cart"] not in x_base:
        continue

    x = x_base[r["cart"]]
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
        fmt="none", ecolor="black", elinewidth=1, capsize=3
    )

    pos[r["cart"]] = (x, y, r["ci_high"], r.get("p_within", np.nan))

# Add a reference line for no change, corresponding to a fold change of 1.
ax.axhline(1, color="gray", linestyle="--", linewidth=1)

# Add the within-CAR p-values above each point.
if len(summary):
    all_ci_high = summary["ci_high"].max()
    y_pad = all_ci_high * 0.06

    for c in cart_order:
        if c not in pos:
            continue
        x, y, ci_hi, p = pos[c]
        ax.text(
            x, ci_hi + y_pad,
            format_p(p),
            ha="center", va="bottom",
            fontsize=9, color="black"
        )

# Add the bracket and p-value for the CAR comparison.
if "CART3" in pos and "CART4" in pos:
    x1, _, y1_hi, _ = pos["CART3"]
    x2, _, y2_hi, _ = pos["CART4"]

    all_ci_high = summary["ci_high"].max()
    yb = all_ci_high * 1.18
    h = all_ci_high * 0.06

    add_bracket_with_p(ax, x1, x2, yb, h, format_p(p_between))

# Format the axes.
ax.set_xticks([x_base[c] for c in cart_order])
ax.set_xticklabels(cart_order)
ax.set_xlabel("CAR")
ax.set_ylabel("Velocity fold-change (during / pre)\n(GM ± 95% t-CI on log)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Start the y-axis at 0 and leave room for labels and brackets.
top = summary["ci_high"].max() * 1.35 if len(summary) else 1
ax.set_ylim(bottom=0, top=top)

handles = [
    Line2D([0], [0], marker=cart_markers[c], linestyle="None",
           color=color, label=c, markersize=7)
    for c in cart_order
]
ax.legend(handles=handles, title="CAR", frameon=False, loc="lower right")

plt.tight_layout()
ax.set_xlim(-0.5, len(cart_order) - 0.5)
# plt.savefig(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig2_velocity_foldchange_LowExp_CD19.png', dpi=600)
# plt.savefig(r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig2_velocity_foldchange_LowExp_CD19.pdf', dpi=600)
plt.show()
#%% Intensities small clusters - mature over time only (with permutation test CART3 vs CART4 per timing)

expression = "Low exp"
particle = 'Zap70'

# Load the table and keep the same cleaning steps as before.
intensities = pd.read_csv(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\all_cotracks_velocities&directionality_stats_correctedtime.csv'
)
intensities_particle = intensities[intensities.particle == particle].copy()

plot_df = intensities_particle[intensities_particle["timing"].notna()].copy()
plot_df = plot_df[plot_df["category"] == 1].copy()

#grouping cols
plot_df = add_common_columns(
    plot_df,
    condition_col="condition",
    run_col="run",
    category_col="category"
)
#filter
plot_df = plot_df[plot_df["expr"] == expression].copy()
plot_df = plot_df[plot_df["dil_group"] != "intermediate"].copy()

# exclude post timing
plot_df = plot_df[plot_df["timing"] != "post"].copy()

ycol = "median_intensity"

# Summarize with median and bootstrap confidence interval.
summary_wide = (
    plot_df.groupby(["cart", "timing", "category"])[ycol]
    .apply(lambda s: bootstrap_ci_median(s.values))
    .reset_index(name="tmp")
)
summary_wide = expand_tuple_column(summary_wide, col="tmp")

order = ["pre", "during"]
summary_plot = summary_wide[summary_wide["timing"].isin(order)].copy()

#  permutation p-values: CART3 vs CART4 within each timing 
pvals = []
for t in order:
    sub = plot_df[plot_df["timing"] == t]
    g1 = sub.loc[sub["cart"] == "CART3", ycol].values
    g2 = sub.loc[sub["cart"] == "CART4", ycol].values
    p = permutation_test_between_groups(g1, g2, n_perm=10000, seed=42, stat="median")
    pvals.append({"timing": t, "p_value": p})
pvals = pd.DataFrame(pvals)

# Plot the summary values.
timing_spacing = 0.55
x_base = {t: i * timing_spacing for i, t in enumerate(order)}

cart_order = ["CART3", "CART4"]
cart_to_i = {c: i for i, c in enumerate(cart_order)}
cart_markers = {"CART3": "s", "CART4": "o"}

car_step = 0.18

plt.figure(figsize=(5.2, 3.8))
ax = plt.gca()

color = "#2ecc71"
pos = {}  

for _, r in summary_plot.iterrows():
    if r["timing"] not in x_base or r["cart"] not in cart_to_i:
        continue

    base = x_base[r["timing"]]
    cart_i = cart_to_i[r["cart"]]
    cart_offset = (cart_i - (len(cart_order) - 1) / 2) * car_step

    x = base + cart_offset
    y = r["median"]

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
        fmt="none", ecolor="black", elinewidth=1, capsize=3
    )

    pos.setdefault(r["timing"], {})[r["cart"]] = (x, r["ci_high"])

# Format the axes.
ax.set_xticks([x_base[t] for t in order])
ax.set_xticklabels(order)
ax.set_xlabel("Timing")
ax.set_ylabel(f"{ycol} (median ± 95% bootstrap CI)")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add permutation-test p-values for CART3 versus CART4 at each timing.
all_ci_high = summary_plot["ci_high"].max()
h = all_ci_high * 0.05
gap = all_ci_high * 0.08
y_base_bracket = all_ci_high * 1.10

for j, t in enumerate(order):
    if t not in pos or "CART3" not in pos[t] or "CART4" not in pos[t]:
        continue

    x1, _ = pos[t]["CART3"]
    x2, _ = pos[t]["CART4"]

    yb = y_base_bracket + j * gap
    p = float(pvals.loc[pvals["timing"] == t, "p_value"].values[0])
    add_bracket_with_p(ax, x1, x2, yb, h, format_p(p))

# Keep the lower axis limit at 1 and leave room for the brackets above.
ax.set_ylim(bottom=1, top=y_base_bracket + (len(order) - 1) * gap + h * 3)

# Add a legend for the CAR markers.
handles = [
    Line2D([0], [0], marker=cart_markers[c], linestyle="None",
           color=color, label=c, markersize=7)
    for c in cart_order
]
ax.legend(handles=handles, title="CAR", frameon=False, loc="upper center")

plt.tight_layout()

# plt.savefig(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig2_intensity_stage_matureOnly_{expression}_{particle}.png', dpi=600)
# plt.savefig(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig2_intensity_stage_matureOnly_{expression}_{particle}.pdf', dpi=600)

plt.show()


#%% Intensities fold-change panel (matures only) + bootstrap CI + permutation tests

expression = "Low exp"
particle = "CD19"
ycol = "median_intensity"

# Load the input table.
intensities = pd.read_csv(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\all_cotracks_velocities&directionality_stats_correctedtime.csv'
)
df = intensities[intensities.particle == particle].copy()

# Filter
df = df[df["timing"].notna()].copy()
df = df[df["category"] == 1].copy()

# Grouping cols
df = add_common_columns(
    df,
    condition_col="condition",
    run_col="run",
    category_col="category"
)
#more filter
df = df[df["expr"] == expression].copy()
df = df[df["dil_group"] != "intermediate"].copy()

# Keep only the pre and during time points.
order = ["pre", "during"]
df = df[df["timing"].isin(order)].copy()

# Filter out any (almost imposible) negative value for GM
df = df[df[ycol].notna() & (df[ycol] > 0)].copy()

# Pair measurements within each track and compute the during/pre fold change.
index_cols = ["colocID", "cart"]
if "run" in df.columns:
    index_cols = ["run", "colocID", "cart"]

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

# Summarize log fold change with the median and bootstrap confidence interval
summary = (
    wide.groupby(["cart"])["log_fc"]
    .apply(lambda s: bootstrap_ci_median(s.values))
)
summary = summary.apply(pd.Series)
summary.columns = ["median_log", "ci_low_log", "ci_high_log"]
summary = summary.reset_index()

# Convert the summary back to fold-change units for plotting.
summary["median"] = np.exp(summary["median_log"])
summary["ci_low"] = np.exp(summary["ci_low_log"])
summary["ci_high"] = np.exp(summary["ci_high_log"])

# Drop the intermediate log-scale summary columns.
summary = summary.drop(columns=["median_log", "ci_low_log", "ci_high_log"])

# Statistical tests
p_within = []
for c in ["CART3", "CART4"]:
    vals = wide.loc[wide["cart"] == c, "log_fc"].values
    p = signflip_permutation_pvalue(vals, n_perm=10000, seed=42, stat="median", sides="two-sided")
    p_within.append({"cart": c, "p_within": p})
p_within = pd.DataFrame(p_within)
summary = summary.merge(p_within, on="cart", how="left")

g1 = wide.loc[wide["cart"] == "CART3", "log_fc"].values
g2 = wide.loc[wide["cart"] == "CART4", "log_fc"].values
p_between = permutation_test_between_groups(g1, g2, n_perm=10000, seed=42, stat="median")

# Plot the fold-change summary using the same visual style.
cart_order = ["CART3", "CART4"]
cart_markers = {"CART3": "s", "CART4": "o"}

plt.figure(figsize=(5.2, 3.8))
ax = plt.gca()
color = "#2ecc71"

x_base = {c: i for i, c in enumerate(cart_order)}
pos = {} 

for _, r in summary.iterrows():
    if r["cart"] not in x_base:
        continue

    x = x_base[r["cart"]]
    y = r["median"]

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
        fmt="none", ecolor="black", elinewidth=1, capsize=3
    )

    pos[r["cart"]] = (x, r["ci_high"], r.get("p_within", np.nan))

# Add a reference line for no change, corresponding to a fold change of 1.
ax.axhline(1, color="gray", linestyle="--", linewidth=1)

# Add p-values for the within-CAR tests against no change.
all_ci_high = summary["ci_high"].max()
y_pad = all_ci_high * 0.06 if np.isfinite(all_ci_high) else 0.1

for c in cart_order:
    if c not in pos:
        continue
    x, ci_hi, p = pos[c]
    ax.text(
        x, ci_hi + y_pad,
        format_p(p),
        ha="center", va="bottom",
        fontsize=9, color="black"
    )

# Add the bracket and p-value comparing CART3 and CART4.
if "CART3" in pos and "CART4" in pos:
    x1, _, _ = pos["CART3"]
    x2, _, _ = pos["CART4"]

    yb = all_ci_high * 1.18 if np.isfinite(all_ci_high) else 1.2
    h = all_ci_high * 0.06 if np.isfinite(all_ci_high) else 0.1

    add_bracket_with_p(ax, x1, x2, yb, h, format_p(p_between))

# Format the axes.
ax.set_xticks([x_base[c] for c in cart_order])
ax.set_xticklabels(cart_order)
ax.set_xlabel("CAR")
ax.set_ylabel("Intensity fold-change (during / pre)\n(median ± 95% bootstrap CI)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlim(-0.5, len(cart_order) - 0.5)

top = all_ci_high * 1.35 if np.isfinite(all_ci_high) else 2
ax.set_ylim(bottom=1, top=top)

handles = [
    Line2D([0], [0], marker=cart_markers[c], linestyle="None",
           color=color, label=c, markersize=7)
    for c in cart_order
]
ax.legend(handles=handles, title="CAR", frameon=False, loc="upper right")

plt.tight_layout()
# plt.savefig(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig2_intensityRatio_stage_matureOnly_{expression}_{particle}.png', dpi=600)
# plt.savefig(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig2_intensityRatio_stage_matureOnly_{expression}_{particle}.pdf', dpi=600)
plt.show()

#%% Intensity zap70/CD19 ratio — Panel 1 with Welch t-test

expression = "Low exp"
ycol = "median_intensity"

# Load the input table.
intensities = pd.read_csv(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\all_cotracks_velocities&directionality_stats_correctedtime.csv'
)

# Apply the filtering choices used for this panel.
df = intensities[intensities["timing"].notna()].copy()
df = df[df["category"] == 1].copy()

#grouping cols
df = add_common_columns(
    df,
    condition_col="condition",
    run_col="run",
    category_col="category"
)
#filter part 2 
df = df[df["expr"] == expression].copy()
df = df[df["dil_group"] != "intermediate"].copy()

df = df[df["particle"].isin(["Zap70", "CD19"])].copy()

order = ["pre", "during"]
df = df[df["timing"].isin(order)].copy()

# Pair Zap70 and CD19 measurements for the same cell, track, and timing.
key_cols = ["cell_id", "colocID", "timing"]

wide = (
    df.pivot_table(index=key_cols, columns="particle", values=ycol)
    .reset_index()
)

wide = wide.dropna(subset=["Zap70", "CD19"]).copy()
wide = wide[wide["CD19"] > 0].copy()
#Calculate ratio
wide["zap70_over_cd19"] = wide["Zap70"] / wide["CD19"]
wide["log_ratio"] = np.log(wide["zap70_over_cd19"])

# Add the CAR identity back to the paired table.
meta = (
    df.loc[df["particle"] == "Zap70", key_cols + ["cart"]]
    .drop_duplicates(subset=key_cols)
)
wide = wide.merge(meta, on=key_cols, how="left")
wide = wide[wide["cart"].isin(["CART3", "CART4"])].copy()

# Summarize the ratio with geometric mean and confidence interval.
tmp = (
    wide.groupby(["timing", "cart"])["zap70_over_cd19"]
    .apply(lambda s: geometric_mean_confidence_interval(s.values))
)
summary = tmp.apply(pd.Series)
summary.columns = ["gmean", "ci_low", "ci_high"]
summary = summary.reset_index()

# Compare CART3 and CART4 at each timing with a Welch t-test on the log ratio.
pvals = []
for t in order:
    sub = wide[wide["timing"] == t]
    g1 = sub.loc[sub["cart"] == "CART3", "log_ratio"].values
    g2 = sub.loc[sub["cart"] == "CART4", "log_ratio"].values
    if len(g1) < 2 or len(g2) < 2:
        p = np.nan
    else:
        _, p = stats.ttest_ind(g1, g2, equal_var=False)
    pvals.append({"timing": t, "p_value": p})
pvals = pd.DataFrame(pvals)

# Plot the summary values.
timing_spacing = 0.55
x_base = {t: i * timing_spacing for i, t in enumerate(order)}

cart_order = ["CART3", "CART4"]
cart_to_i = {c: i for i, c in enumerate(cart_order)}
cart_markers = {"CART3": "s", "CART4": "o"}
car_step = 0.18

plt.figure(figsize=(5.2, 3.8))
ax = plt.gca()

color = "#2ecc71"
pos = {}

for _, r in summary.iterrows():
    base = x_base[r["timing"]]
    cart_i = cart_to_i[r["cart"]]
    cart_offset = (cart_i - (len(cart_order) - 1) / 2) * car_step

    x = base + cart_offset
    y = r["gmean"]

    ax.plot(x, y,
            marker=cart_markers[r["cart"]],
            linestyle="None",
            color=color,
            markersize=7)

    ax.errorbar(
        x, y,
        yerr=[[y - r["ci_low"]], [r["ci_high"] - y]],
        fmt="none", ecolor="black", elinewidth=1, capsize=3
    )

    pos.setdefault(r["timing"], {})[r["cart"]] = (x, r["ci_high"])

# Add p-value brackets
all_ci_high = summary["ci_high"].max()
h = all_ci_high * 0.05
gap = all_ci_high * 0.08
y_base_bracket = all_ci_high * 1.10

for j, t in enumerate(order):
    if t not in pos or "CART3" not in pos[t] or "CART4" not in pos[t]:
        continue

    x1, _ = pos[t]["CART3"]
    x2, _ = pos[t]["CART4"]

    yb = y_base_bracket + j * gap
    p = float(pvals.loc[pvals["timing"] == t, "p_value"].values[0])
    add_bracket_with_p(ax, x1, x2, yb, h, format_p(p))

# Format axes, limits, and legend.
ax.set_xticks([x_base[t] for t in order])
ax.set_xticklabels(order)
ax.set_xlabel("Timing")
ax.set_ylabel("Zap70 / CD19 (GM ± 95% t-CI on log)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_ylim(bottom=0, top=y_base_bracket + (len(order) - 1) * gap + h * 3)

handles = [
    Line2D([0], [0], marker=cart_markers[c], linestyle="None",
           color=color, label=c, markersize=7)
    for c in cart_order
]
ax.legend(handles=handles, title="CAR", frameon=False, loc="upper center")

plt.tight_layout()
# plt.savefig(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig2_intensity_Zap70overCD19_stage_matureOnly_{expression}.png', dpi=600)
# plt.savefig(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig2_intensity_Zap70overCD19_stage_matureOnly_{expression}.pdf', dpi=600)
plt.show()


#%% Zap70/CD19 ratio-of-ratios (paired change) — Δ panel with stats

expression = "Low exp"
ycol = "median_intensity"

# Load the input table.
intensities = pd.read_csv(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\all_cotracks_velocities&directionality_stats_correctedtime.csv'
)

#Filter
df = intensities[intensities["timing"].notna()].copy()
df = df[df["category"] == 1].copy()  # mature only

# Grouping cols
df = add_common_columns(
    df,
    condition_col="condition",
    run_col="run",
    category_col="category"
)
#more filtering
df = df[df["expr"] == expression].copy()
df = df[df["dil_group"] != "intermediate"].copy()

df = df[df["particle"].isin(["Zap70", "CD19"])].copy()

order = ["pre", "during"]
df = df[df["timing"].isin(order)].copy()

# Pair Zap70 and CD19 by cell, colocalization track, and timing.
key_cols = ["cell_id", "colocID", "timing"]

wide = (
    df.pivot_table(index=key_cols, columns="particle", values=ycol)
    .reset_index()
)

wide = wide.dropna(subset=["Zap70", "CD19"]).copy()
wide = wide[wide["CD19"] > 0].copy()
#clauclate ratio
wide["ratio"] = wide["Zap70"] / wide["CD19"]
wide = wide[wide["ratio"].notna() & (wide["ratio"] > 0)].copy()
wide["log_ratio"] = np.log(wide["ratio"])

# Add the CAR identity back using the same approach as in the first ratio panel.
meta = (
    df.loc[df["particle"] == "Zap70", key_cols + ["cart"]]
    .dropna(subset=["cart"])
    .drop_duplicates(subset=key_cols)
)
wide = wide.merge(meta, on=key_cols, how="left")

missing_cart = wide["cart"].isna()
if missing_cart.any():
    meta_cd19 = (
        df.loc[df["particle"] == "CD19", key_cols + ["cart"]]
        .dropna(subset=["cart"])
        .drop_duplicates(subset=key_cols)
    )
    wide.loc[missing_cart, "cart"] = (
        wide.loc[missing_cart, key_cols]
        .merge(meta_cd19, on=key_cols, how="left")["cart"]
        .values
    )

wide = wide[wide["cart"].isin(["CART3", "CART4"])].copy()

# Pair pre and during measurements within each track and compute the change in log ratio.
pair_cols = ["cell_id", "colocID", "cart"]

wide2 = (
    wide.pivot_table(index=pair_cols, columns="timing", values="log_ratio", aggfunc="first")
    .reset_index()
)
#make second ratio
wide2 = wide2.dropna(subset=["pre", "during"]).copy()
wide2["delta_log_ratio"] = wide2["during"] - wide2["pre"]

# Convert the log-scale change into a ratio of ratios for plotting.
wide2["ratio_of_ratios"] = np.exp(wide2["delta_log_ratio"])

# Summarize the ratio of ratios for each CAR.
tmp = (
    wide2.groupby(["cart"])["ratio_of_ratios"]
    .apply(lambda s: geometric_mean_confidence_interval(s.values))
)
summary = tmp.apply(pd.Series)
summary.columns = ["gmean", "ci_low", "ci_high"]
summary = summary.reset_index()

# Statistical tests
p_within = []
for c in ["CART3", "CART4"]:
    vals = wide2.loc[wide2["cart"] == c, "delta_log_ratio"].values
    if len(vals) < 2:
        p = np.nan
    else:
        _, p = stats.ttest_1samp(vals, 0)  # test mean Δlog = 0
    p_within.append({"cart": c, "p_within": p})
p_within = pd.DataFrame(p_within)
summary = summary.merge(p_within, on="cart", how="left")

g1 = wide2.loc[wide2["cart"] == "CART3", "delta_log_ratio"].values
g2 = wide2.loc[wide2["cart"] == "CART4", "delta_log_ratio"].values
if len(g1) < 2 or len(g2) < 2:
    p_between = np.nan
else:
    _, p_between = stats.ttest_ind(g1, g2, equal_var=False)  # Welch on Δlog

# Plot the fold-change summary using the same visual style.
cart_order = ["CART3", "CART4"]
cart_markers = {"CART3": "s", "CART4": "o"}

plt.figure(figsize=(5.2, 3.8))
ax = plt.gca()

color = "#2ecc71"
x_base = {c: i for i, c in enumerate(cart_order)}
pos = {}  # pos[cart] = (x, ci_high, p_within)

for _, r in summary.iterrows():
    if r["cart"] not in x_base:
        continue

    x = x_base[r["cart"]]
    y = r["gmean"]

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
        fmt="none", ecolor="black", elinewidth=1, capsize=3
    )

    pos[r["cart"]] = (x, r["ci_high"], r.get("p_within", np.nan))

# Add a reference line for no change, where the ratio of ratios equals 1.
ax.axhline(1, color="gray", linestyle="--", linewidth=1)

# Add p-values for the within-CAR tests against no change.
all_ci_high = summary["ci_high"].max()
y_pad = all_ci_high * 0.06 if np.isfinite(all_ci_high) else 0.1

for c in cart_order:
    if c not in pos:
        continue
    x, ci_hi, p = pos[c]
    ax.text(
        x, ci_hi + y_pad,
        format_p(p),
        ha="center", va="bottom",
        fontsize=9, color="black"
    )

# Add the bracket and p-value comparing CART3 and CART4.
if "CART3" in pos and "CART4" in pos:
    x1, _, _ = pos["CART3"]
    x2, _, _ = pos["CART4"]

    yb = all_ci_high * 1.18 if np.isfinite(all_ci_high) else 1.2
    h = all_ci_high * 0.06 if np.isfinite(all_ci_high) else 0.1

    add_bracket_with_p(ax, x1, x2, yb, h, format_p(p_between))

# Format the axes.
ax.set_xticks([x_base[c] for c in cart_order])
ax.set_xticklabels(cart_order)
ax.set_xlabel("CAR")
ax.set_ylabel("Fold change Zap70/CD19 ratio (during / pre)\n(GM ± 95% t-CI on log)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlim(-0.5, len(cart_order) - 0.5)

top = all_ci_high * 1.35 if np.isfinite(all_ci_high) else 2
ax.set_ylim(bottom=0, top=top)

handles = [
    Line2D([0], [0], marker=cart_markers[c], linestyle="None",
           color=color, label=c, markersize=7)
    for c in cart_order
]
ax.legend(handles=handles, title="CAR", frameon=False, loc="lower right")

plt.tight_layout()
# plt.savefig(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig2_Zap70overCD19Ratio_matureOnly_{expression}.png', dpi=600)
# plt.savefig(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig2_Zap70overCD19Ratio_matureOnly_{expression}.pdf', dpi=600)
plt.show()


#%% Colocalization and intensity example 
#D:\Data\Chi_data\20250801_filtered\output\CART4 CAT Low aff Low exp\100xdilutedCD19\20240826_142xdilutedCD19\R1_cont\Run00002 - colocID 9
#326 not bad
# Second example:
#D:\Data\Chi_data\20250801_filtered\output\CART4 CAT Low aff Low exp\50xdilutedCD19\20240826_53.25xdilutedCD19\R2\Run00003
#113
# Third example:
#P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\CART4 CAT Low aff High exp\1000xdilutedCD19\20241104_1000xdilutedCD19\R2\Run00002
#269
# 106
cID = 7
example_run = ta.Single_tracked_folder(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\CART4 CAT Low aff High exp\1000xdilutedCD19\20241104_1000xdilutedCD19\R2\Run00002'
).open_files()
a = example_run.coloc_stats
# Uncomment to inspect the available colocalization IDs.
c = example_run.plot_colocs([cID])
plt.show()
b = example_run.plot_intensity_coloc(cID, legend_0='CD19', legend_1='ZAP70')

plt.show()
# b.save_plot(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig2_Example_cotrack_intensityID{cID}.png', dpi=600)
# b.save_plot(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig2_Example_cotrack_intensityID{cID}.pdf', dpi=600)
# c.save_plot(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig2_Example_cotrack_CoLocalizationID{cID}.png', dpi=600)
# c.save_plot(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig2_Example_cotrack_CoLocalizationID{cID}.pdf', dpi=600)


#%% intensity before and after maturation
expression = "Low exp"
title_dict = {
    'total_mean_cd': "Intensity CD19",
    'total_mean_zap': "Intensity Zap70",
    "total_mean_ratio": "Zap70 / CD19 intensity ratio"
}
name = {
    'total_mean_cd': "CD19",
    'total_mean_zap': "Zap70",
    "total_mean_ratio": "Zap70overCD19"
}
col_to_plot = "total_mean_ratio"

intensities = pd.read_csv(
    r'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\analysis_output\intensity_maturation_summary.csv'
)

int_to_plot = intensities[
    (intensities["expr"] == expression) &
    (intensities[col_to_plot] > 0)
].copy()

# Summarize with geometric mean and confidence interval.
summary = (
    int_to_plot.dropna()
    .groupby(["CART", "mature"])[col_to_plot]
    .apply(lambda s: geometric_mean_confidence_interval(s.values))
    .apply(pd.Series)
    .reset_index()
)
summary.columns = ["CART", "mature", "gmean", "ci_low", "ci_high"]

# Compare mature and not-mature groups within each CAR using a Welch t-test on log-transformed intensity.
pvals = []
for cart in summary["CART"].unique():
    sub = int_to_plot[int_to_plot["CART"] == cart]
    g1 = np.log(sub.loc[sub["mature"] == "not-mature", col_to_plot].values)
    g2 = np.log(sub.loc[sub["mature"] == "mature", col_to_plot].values)

    if len(g1) < 2 or len(g2) < 2:
        p = np.nan
    else:
        _, p = stats.ttest_ind(g1, g2, equal_var=False)

    pvals.append({"CART": cart, "p_value": p})
pvals = pd.DataFrame(pvals)

# Set up the plot layout and color choices.
cart_order = sorted(summary["CART"].unique())
mature_order = ["not-mature", "mature"]

x_base = {c: i for i, c in enumerate(cart_order)}
offset_step = 0.25
mature_to_i = {m: i for i, m in enumerate(mature_order)}

palette = {
    mature_order[0]: "#b23a8a",
    mature_order[1]: "#1A7A42"
}

plt.figure(figsize=(6, 4))
ax = plt.gca()

pos = {} 

# Plot the summary points and their confidence intervals.
for _, r in summary.iterrows():
    base = x_base[r["CART"]]
    offset = (mature_to_i[r["mature"]] - (len(mature_order) - 1) / 2) * offset_step
    x = base + offset

    color = palette[r["mature"]]
    y = r["gmean"]

    ax.plot(x, y, marker="o", linestyle="None",
            color=color, markersize=7)

    ax.errorbar(
        x, y,
        yerr=[[y - r["ci_low"]], [r["ci_high"] - y]],
        fmt="none", ecolor="black", elinewidth=1, capsize=3
    )

    pos.setdefault(r["CART"], {})[r["mature"]] = (x, r["ci_high"])

# Add p-value brackets for the timing-specific comparisons.
all_ci_high = summary["ci_high"].max()
h = all_ci_high * 0.05
y_base_bracket = all_ci_high * 1.10

for i, cart in enumerate(cart_order):
    if cart not in pos:
        continue
    if "not-mature" not in pos[cart] or "mature" not in pos[cart]:
        continue

    x1, _ = pos[cart]["not-mature"]
    x2, _ = pos[cart]["mature"]
    yb = y_base_bracket

    p = float(pvals.loc[pvals["CART"] == cart, "p_value"].values[0])
    add_bracket_with_p(ax, x1, x2, yb, h, format_p(p))

# Format the axes.
ax.set_xticks([x_base[c] for c in cart_order])
ax.set_xticklabels(cart_order)
ax.set_xlabel("CART")
ax.set_ylabel(f"{title_dict[col_to_plot]} (geometric mean ± 95% CI)")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlim(-0.5, len(cart_order) - 0.5)
ax.set_ylim(bottom=0, top=y_base_bracket + h * 3)

# Add the legend.
handles = [
    Line2D([0], [0], marker='o', linestyle='None',
           color=palette[m], label=str(m), markersize=7)
    for m in mature_order
]
ax.legend(handles=handles, title="Mature", frameon=False)

plt.tight_layout()
# plt.savefig(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig1_intensity_maturation_{name[col_to_plot]}_LowExp.pdf', dpi=600)
# plt.savefig(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig1_intensity_maturation_{name[col_to_plot]}_LowExp.png', dpi=600)
plt.show()
#%% Colocalization and intensity example 
#D:\Data\Chi_data\20250801_filtered\output\CART4 CAT Low aff Low exp\100xdilutedCD19\20240826_142xdilutedCD19\R1_cont\Run00002 - colocID 9
# cIDs: 7 and 20181
cID = 218
example_run = ta.Single_tracked_folder(
    r'D:\Data\Chi_data\20250801_filtered\output\CART4 CAT Low aff Low exp\100xdilutedCD19\20240826_142xdilutedCD19\R1_cont\Run00002'
).open_files()
a = example_run.stats0
c = example_run.plot_tracks([cID])
# plt.show()
# c.save_plot(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig2_Example_track_CoLocalizationID{cID}.png', dpi=600)
# c.save_plot(rf'P:\10 CART Chi\6. All data\1. ZAP70 recruitment\20250801_filtered\output\Nguyen2026_analysis\figures\Fig2_Example_track_CoLocalizationID{cID}.pdf', dpi=600)