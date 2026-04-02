import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from _02_targets import plot_sector_region_fraction_heatmap_with_text

# ----------------------------
# Parameters and Data
# ----------------------------

first_year_available, last_year_available = 2019, 2024

df = pd.read_parquet(
    f"data/intermediate_data/df_merged_all_infos_{first_year_available}_{last_year_available}.parquet"
)
df_em_after_outliers_treatment = pd.read_parquet(
    "data/intermediate_data/df_em_after_outliers_treatment.parquet"
)

df_targets = pd.read_excel("data/intermediate_data/df_eq_target_ambitions.xlsx")

all_isin_index = df[df["is_relevant_scopes"] == True].set_index("isin").index
all_isin_rel_scope = (
    df[df["is_relevant_scopes"] == True].set_index(["isin", "scope"]).index
)

regions = list(df["region_0"].unique())
list_high_impact_sector = list(df["high_impact_sector"].dropna().sort_values().unique())
delta_years_str = [
    f"{int(y)}-{y+1}" for y in range(first_year_available, last_year_available)
]

# ----------------------------
# Alignment parameters
# ----------------------------


threshold = 0
target_needed = "is_target_sbt_or_ambitious_or_commited"


# ----------------------------
# Application of alignment rule
# ----------------------------

# eq with average rate inferior to threshold

eq_all_scope_pass = (
    df_em_after_outliers_treatment.set_index(["isin", "scope"])[delta_years_str].mean(
        axis=1
    )
    < threshold
)
eq_rel_scp_pass = (
    eq_all_scope_pass.loc[all_isin_rel_scope.intersection(eq_all_scope_pass.index)]
    .reset_index()
    .set_index("isin")[0]
)
eq_rel_scp_pass.name = "below_threshold"

# eq with specified target ambition

eq_target_pass = df_targets[df_targets[target_needed] == True][
    ["isin", target_needed]
].set_index("isin")
eq_target_pass = eq_target_pass.loc[
    all_isin_index.intersection(eq_target_pass.index)
].astype(bool)

# merge

df_all_info_for_alignment = pd.concat(
    [
        df[df["is_relevant_scopes"] == True].set_index("isin")[
            ["scope", "high_impact_sector", "region_0", "r_s_mc_weight"]
        ],
        eq_rel_scp_pass,
        eq_target_pass,
    ],
    axis=1,
)

# compute who is aligned

df_all_info_for_alignment["aligned"] = (
    df_all_info_for_alignment[
        ["below_threshold", "is_target_sbt_or_ambitious_or_commited"]
    ]
    .astype("boolean")
    .fillna(False)
    .all(axis=1)
)
df_all_info_for_alignment["challenger"] = (
    df_all_info_for_alignment[
        ["below_threshold", "is_target_sbt_or_ambitious_or_commited"]
    ]
    .astype("boolean")
    .fillna(False)
    .any(axis=1)
)

# ----------------------------
# Generate heatmaps
# ----------------------------

plot_sector_region_fraction_heatmap_with_text(
    df_all_info_for_alignment[df_all_info_for_alignment["aligned"]].reset_index(),
    df_all_info_for_alignment.reset_index(),
    "output/fraction_eq_aligned_heatmap.png",
    "Nbr aligned on total number by region and sectors",
    reverse_colors=False,
)

plot_sector_region_fraction_heatmap_with_text(
    df_all_info_for_alignment[df_all_info_for_alignment["challenger"]].reset_index(),
    df_all_info_for_alignment.reset_index(),
    "output/fraction_eq_challenger_heatmap.png",
    "Nbr challenger on total number by region and sectors",
    reverse_colors=False,
)
