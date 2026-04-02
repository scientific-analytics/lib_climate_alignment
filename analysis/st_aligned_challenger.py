import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ----------------------------
# Run Streamlit:
# streamlit run analysis/st_aligned_challenger.py
# ----------------------------


@st.cache_data
def load_data(first_year_available, last_year_available):

    df = pd.read_parquet(
        f"data/intermediate_data/df_merged_all_infos_{first_year_available}_{last_year_available}.parquet"
    )
    df_em_after_outliers_treatment = pd.read_parquet(
        "data/intermediate_data/df_em_after_outliers_treatment.parquet"
    )

    df_targets = pd.read_excel("data/intermediate_data/df_eq_target_ambitions.xlsx")

    return df, df_em_after_outliers_treatment, df_targets


first_year_available, last_year_available = 2019, 2024
df, df_em_after_outliers_treatment, df_targets = load_data(
    first_year_available, last_year_available
)


@st.cache_data
def prepare_base(df, first_year_available, last_year_available):

    df_relevant = df[df["is_relevant_scopes"] == True]
    all_isin_index = df_relevant.set_index("isin").index
    all_isin_rel_scope = df_relevant.set_index(["isin", "scope"]).index

    regions = list(df["region_0"].unique())
    list_high_impact_sector = list(
        df["high_impact_sector"].dropna().sort_values().unique()
    )
    delta_years_str = [
        f"{int(y)}-{y+1}" for y in range(first_year_available, last_year_available)
    ]
    return (
        all_isin_index,
        all_isin_rel_scope,
        regions,
        list_high_impact_sector,
        delta_years_str,
    )


(
    all_isin_index,
    all_isin_rel_scope,
    regions,
    list_high_impact_sector,
    delta_years_str,
) = prepare_base(df, first_year_available, last_year_available)

# ----------------------------
# Alignment parameters
# ----------------------------
st.write("**Number equity aligned or challenger by high impact sector and region**")

col1, col2 = st.columns([1, 1])

with col1:
    threshold = st.slider(
        "Select a threshold to determine the aligned and challenger",
        min_value=-1.0,
        max_value=1.0,
        step=0.1,
        value=0.0,
    )

dict_mapping_targets_ctgry = {
    "Ambitious Target": "is_ambitious_target",
    "Approved SBT": "is_approved_sbt_target",
    "Committed SBT": "is_committed_sbt_target",
    "Non-Ambitious Target": "is_target_non_ambitious",
    "No Target": "is_no_target",
}

with col2:
    target_ambitions_labels_selected = st.multiselect(
        "Target ambition category",
        options=list(dict_mapping_targets_ctgry.keys()),
        default=["Ambitious Target"],
        placeholder="Choose one or multiple target ambition category(ies)",
    )

target_ambitions_selected = [
    dict_mapping_targets_ctgry[lbl] for lbl in target_ambitions_labels_selected
]

aligned_or_challenger = st.radio(
    label="aligned_or_challenger",
    label_visibility="collapsed",
    options=["Aligned", "Challenger"],
    horizontal=True,
)
# ----------------------------
# Eq with average rate inferior to threshold
# ----------------------------

eq_all_scope_pass = (
    df_em_after_outliers_treatment.set_index(["isin", "scope"])[delta_years_str].mean(
        axis=1
    )
    < threshold
)
eq_rel_scp_pass = eq_all_scope_pass.reindex(all_isin_rel_scope).reset_index(
    level="scope", drop=True
)
eq_rel_scp_pass.name = "below_threshold"

# ----------------------------
# Eq with specified target ambition
# ----------------------------

eq_target_pass = (
    df_targets.set_index("isin")
    .reindex(all_isin_index)[target_ambitions_selected]
    .any(axis=1)
    .rename("has_target_needed")
)

# ----------------------------
# Compute aligned and challenger
# ----------------------------
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

df_all_info_for_alignment["Aligned"] = (
    df_all_info_for_alignment[["below_threshold", "has_target_needed"]]
    .astype("boolean")
    .fillna(False)
    .all(axis=1)
)
df_all_info_for_alignment["Challenger"] = (
    df_all_info_for_alignment[["below_threshold", "has_target_needed"]]
    .astype("boolean")
    .fillna(False)
    .any(axis=1)
)

# ----------------------------
# Create table
# ----------------------------


def create_alignment_heatmap(df, aligned_or_challenger):
    df_color = df.pivot_table(
        index="high_impact_sector",
        columns="region_0",
        values=aligned_or_challenger,
        aggfunc="mean",
    ).reindex(index=list_high_impact_sector, columns=regions)

    num = df.pivot_table(
        index="high_impact_sector",
        columns="region_0",
        values=aligned_or_challenger,
        aggfunc=lambda x: x.fillna(False).sum(),
    ).reindex(index=list_high_impact_sector, columns=regions)

    den = df.pivot_table(
        index="high_impact_sector",
        columns="region_0",
        values=aligned_or_challenger,
        aggfunc="count",
    ).reindex(index=list_high_impact_sector, columns=regions)

    df_frac = (
        num.fillna(0).astype(int).astype(str)
        + "/"
        + den.fillna(0).astype(int).astype(str)
    )

    return df_color, df_frac


df_color, df_frac = create_alignment_heatmap(
    df_all_info_for_alignment, aligned_or_challenger
)


def style_fraction_table(df_frac, df_color, reverse_colors=False):
    cmap = mpl.colormaps["RdYlGn_r" if reverse_colors else "RdYlGn"]

    def value_to_css(val):
        if pd.isna(val):
            return "background-color: lightgrey; color: black;"
        rgba = cmap(val)  # val supposé entre 0 et 1
        r, g, b, _ = [int(255 * x) for x in rgba]
        return f"background-color: rgb({r},{g},{b}); color: black;"

    styles = pd.DataFrame("", index=df_color.index, columns=df_color.columns)

    for i in df_color.index:
        for j in df_color.columns:
            styles.loc[i, j] = value_to_css(df_color.loc[i, j])

    return df_frac.style.apply(lambda _: styles, axis=None)


height = min(800, 40 + df_frac.shape[0] * 35)

st.dataframe(style_fraction_table(df_frac, df_color), width="stretch", height=height)
