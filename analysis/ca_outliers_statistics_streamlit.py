import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ----------------------------
# Run Streamlit:
# streamlit run analysis/ca_outliers_statistics_streamlit.py
# ----------------------------


# ----------------------------
# LOAD AND PREPARE DATA
# ----------------------------

first_year_available, last_year_available = 2019, 2024

df = pd.read_parquet(
    f"data/intermediate_data/df_merged_all_infos_{first_year_available}_{last_year_available}.parquet"
)
df_hist_abs_emissions = pd.read_parquet(
    f"data/intermediate_data/hist_abs_emissions_{first_year_available}_{last_year_available}.parquet"
)
df_abs_em_rate = pd.read_parquet(
    f"data/intermediate_data/hist_abs_emissions_growth_rate_{first_year_available}_{last_year_available}.parquet"
)
df_hist_intensities = pd.read_parquet(
    f"data/intermediate_data/hist_intensities_{first_year_available}_{last_year_available}.parquet"
)
df_intensities_rate = pd.read_parquet(
    f"data/intermediate_data/hist_intensities_growth_rate_{first_year_available}_{last_year_available}.parquet"
)
df_auto_intensities_rate = pd.read_parquet(
    f"data/intermediate_data/df_auto_intensities_rate_{first_year_available}_{last_year_available}.parquet"
)

scopes = ["s1", "s2", "s3_d", "s3_u"]
regions = list(df["region_0"].unique())
list_high_impact_sector = list(df["high_impact_sector"].dropna().sort_values().unique())
years_str = sorted(
    [c for c in df_hist_abs_emissions.columns if c.isdigit()], key=int, reverse=False
)
dict_year_delta_years = {y: f"{int(y)-1}-{y}" for y in years_str[1:]}
delta_years_str = list(dict_year_delta_years.values())

df_eq_abs_em = pd.merge(
    df[["isin", "scope", "high_impact_sector", "region_0"]],
    df_hist_abs_emissions,
    how="left",
    on=["isin", "scope"],
)
df_eq_abs_em_rate = pd.merge(
    df[["isin", "scope", "high_impact_sector", "region_0"]],
    df_abs_em_rate,
    how="left",
    on=["isin", "scope"],
)
df_eq_intensities = pd.merge(
    df[["isin", "scope", "high_impact_sector", "region_0"]],
    df_hist_intensities,
    how="left",
    on=["isin", "scope"],
)
df_eq_intensities_rate = pd.merge(
    df[["isin", "scope", "high_impact_sector", "region_0"]],
    df_intensities_rate[["isin", "scope"] + delta_years_str],
    how="left",
    on=["isin", "scope"],
)
df_eq_auto_intensities_rate = pd.merge(
    df[["isin", "scope", "high_impact_sector", "region_0"]],
    df_auto_intensities_rate[["isin", "scope"] + delta_years_str],
    how="left",
    on=["isin", "scope"],
)

# ----------------------------
# MISSING
# ----------------------------

st.subheader("Missing")
st.write(
    "nbr eq with missing sector:",
    f"{df[df['high_impact_sector'].isna()]['isin'].nunique()}/{df['isin'].nunique()}",
)
st.write(
    "percentage market cap with missing sector:",
    f"{df[df['high_impact_sector'].isna()]['company_free_float_market_cap'].sum()/df['company_free_float_market_cap'].sum():.1%}",
)


# ----------------------------
# TABLE 1 - Total equities
# ----------------------------

st.subheader("Universe / total number of equity")

df_one_scp = df[df["scope"] == "s1"]

df_groupby = (
    df_one_scp.groupby(["high_impact_sector", "region_0"])["isin"].count().reset_index()
)
total_by_sector = df_groupby.groupby("high_impact_sector")["isin"].sum().reset_index()
total_by_sector["region_0"] = "Total"
total_by_region = df_groupby.groupby("region_0")["isin"].sum().reset_index()
total_by_region["high_impact_sector"] = "Total"
df_groupby = pd.concat([df_groupby, total_by_sector, total_by_region], axis=0)
df_pivot = df_groupby.pivot(
    index="high_impact_sector",
    columns="region_0",
    values="isin",
)
df_pivot = df_pivot.loc[list_high_impact_sector + ["Total"]]
df_pivot.loc["Total", "Total"] = df_groupby[
    ~(
        (df_groupby["region_0"] == "Total")
        | (df_groupby["high_impact_sector"] == "Total")
    )
]["isin"].sum()

row_height = 35
st.dataframe(df_pivot, height=row_height * (len(df_pivot) + 1))


# ----------------------------
# PARAMETERS
# ----------------------------

st.subheader("Parameters")

col1, col2, col3 = st.columns(3)
with col1:
    selected_region_0 = st.multiselect(
        "Region(s)",
        options=regions,
        placeholder="Choose one or multiple options",
        key="region_0",
    )
with col2:
    selected_hisector_0 = st.multiselect(
        "High impact sector(s)",
        options=list_high_impact_sector,
        placeholder="Choose one or multiple options",
        key="hisector_0",
    )
with col3:
    selected_scope_0_list = st.multiselect(
        "Scope(s)",
        options=scopes,
        placeholder="Choose one or multiple scope(s)",
        key="scope_0",
    )

if not selected_region_0:
    selected_region_0 = regions
if not selected_hisector_0:
    selected_hisector_0 = list_high_impact_sector
if not selected_scope_0_list:
    selected_scope_0_list = scopes


def scope_label(scopes):
    scopes = set(scopes)
    base = "s" + "".join(
        x[1]
        for x in ["s1", "s2", "s3_u"]
        if x in scopes or (x == "s3_u" and "s3_d" in scopes)
    )
    if "s3_u" in scopes and "s3_d" not in scopes:
        return base + "_u"
    if "s3_d" in scopes and "s3_u" not in scopes:
        return base + "_d"
    return base


selected_scope_0 = scope_label(selected_scope_0_list)


# ----------------------------
# TABLE 2 - Stats on market cap
# ----------------------------

st.subheader("Initial statistitics")

df_selection = df_one_scp[
    (df_one_scp["region_0"].isin(selected_region_0))
    & (df_one_scp["high_impact_sector"].isin(selected_hisector_0))
]

st.write(
    "Block (region(s) and (sector(s)) market cap on total universe market cap :",
    f"{df_selection['company_free_float_market_cap'].sum() / df_one_scp['company_free_float_market_cap'].sum():.1%}",
)

df_selection = df_selection.copy()
df_selection["w_mc_selection"] = (
    df_selection["company_free_float_market_cap"]
    / df_selection["company_free_float_market_cap"].sum()
)
df_stats = df_selection["w_mc_selection"].describe()
df_fmt = df_stats.copy().astype(object)
df_fmt.loc["count"] = f"{int(df_stats.loc['count'])}"
df_fmt.loc["std"] = f"{df_stats.loc['std']:.2e}"
df_fmt.loc[~df_fmt.index.isin(["count", "std"])] = df_stats.loc[
    ~df_stats.index.isin(["count", "std"])
].map(lambda x: f"{x:.1%}")

st.write(
    "**Statistics on the weight (market cap weighted) of this block (region(s) and (sector(s))**"
)
st.dataframe(df_fmt.to_frame().T, hide_index=True)


# ----------------------------
# TABLE 3 - Absolute emissions - statistics
# ----------------------------

st.write("**Absolute emissions - statistics**")

df_selection_all_col_0 = df_eq_abs_em[
    (df_eq_abs_em["region_0"].isin(selected_region_0))
    & (df_eq_abs_em["high_impact_sector"].isin(selected_hisector_0))
    & (df_eq_abs_em["scope"] == selected_scope_0)
].copy()

df_selection_0 = df_selection_all_col_0[["scope"] + years_str].copy()
df_selection_0["last_available_year"] = (
    df_selection_0[years_str[::-1]].bfill(axis=1).iloc[:, 0]
)

selected_year_0 = st.multiselect(
    label="selected_year_0",
    label_visibility="collapsed",
    options=["last_available_year"] + years_str,
    placeholder="Choose one or multiple  year(s)",
    key="year_0",
)
if not selected_year_0:
    selected_year_0 = years_str

df_selection_0_mean = pd.concat(
    [
        df_selection_0["scope"],
        df_selection_0[selected_year_0].mean(axis=1).rename("mean_value"),
    ],
    axis=1,
)
df_stats_0 = df_selection_0_mean.groupby("scope")["mean_value"].describe()

df_fmt_0 = df_stats_0.copy()
df_fmt_0["count"] = df_stats_0["count"].astype(int)
df_fmt_0["std"] = df_stats_0["std"].map(lambda x: f"{x:.2e}")
cols_pct = df_fmt_0.columns.difference(["count", "std"])
df_fmt_0[cols_pct] = df_stats_0[cols_pct].map(lambda x: f"{x:.2e}")

st.dataframe(df_fmt_0, height=row_height * (len(df_fmt_0) + 1))

# ----------------------------
# TABLE 4 - Intensities - statistics
# ----------------------------

st.write("**Intensities - statistics**")

df_selection_all_col_2 = df_eq_intensities[
    (df_eq_intensities["region_0"].isin(selected_region_0))
    & (df_eq_intensities["high_impact_sector"].isin(selected_hisector_0))
    & (df_eq_intensities["scope"] == selected_scope_0)
]

df_selection_2 = df_selection_all_col_2[["scope"] + years_str].copy()
df_selection_2["last_available_year"] = (
    df_selection_2[years_str[::-1]].bfill(axis=1).iloc[:, 0]
)

selected_year_2 = st.multiselect(
    label="selected_year_2",
    label_visibility="collapsed",
    options=["last_available_year"] + years_str,
    placeholder="Choose one or multiple  year(s)",
    key="year_2",
)

if not selected_year_2:
    selected_year_2 = years_str

df_selection_2_mean = pd.concat(
    [
        df_selection_2["scope"],
        df_selection_2[selected_year_2].mean(axis=1).rename("mean_value"),
    ],
    axis=1,
)
df_stats_2 = df_selection_2_mean.groupby("scope")["mean_value"].describe()

df_fmt_2 = df_stats_2.copy()
df_fmt_2["count"] = df_stats_2["count"].astype(int)
df_fmt_2["std"] = df_stats_2["std"].map(lambda x: f"{x:.2e}")
cols_pct = df_fmt_2.columns.difference(["count", "std"])
df_fmt_2[cols_pct] = df_stats_2[cols_pct].map(lambda x: f"{x:.2e}")

st.dataframe(df_fmt_2, height=row_height * (len(df_fmt_2) + 1))

# ----------------------------
# TABLE 45 - statistics for adjusted data
# ----------------------------
df_eq_abs_em_rate_select = df_eq_abs_em_rate[
    (df_eq_abs_em_rate["region_0"].isin(selected_region_0))
    & (df_eq_abs_em_rate["high_impact_sector"].isin(selected_hisector_0))
    & (df_eq_abs_em_rate["scope"] == selected_scope_0)
].copy()

df_eq_intensities_rate_select = df_eq_intensities_rate[
    (df_eq_intensities_rate["region_0"].isin(selected_region_0))
    & (df_eq_intensities_rate["high_impact_sector"].isin(selected_hisector_0))
    & (df_eq_intensities_rate["scope"] == selected_scope_0)
].copy()

df_eq_auto_intensities_rate_select = df_eq_auto_intensities_rate[
    (df_eq_auto_intensities_rate["region_0"].isin(selected_region_0))
    & (df_eq_auto_intensities_rate["high_impact_sector"].isin(selected_hisector_0))
    & (df_eq_auto_intensities_rate["scope"] == selected_scope_0)
].copy()

st.subheader("Ouliers adjustments")

# choix du seuil
col1, col2, col3 = st.columns([3, 2, 2])

with col1:
    st.write("Choose the threshold for extreme values:")


with col2:
    threshold_1 = st.selectbox(
        label="threshold_1",
        options=[None, 0.1, 0.2, 0.3, 0.5],
        format_func=lambda x: (
            "Select the value" if x is None else f"{-x:.1%} / {x:.1%}"
        ),
        key="threshold_1",
        label_visibility="collapsed",
    )


def create_columns_number_outliers(
    df_emissions, threshold, delta_years_str=delta_years_str
):

    df_emissions = df_emissions.copy()
    if threshold is None:
        df_emissions[
            [
                "number_positive_outliers",
                "number_negative_outliers",
                "number_all_outliers",
            ]
        ] = 0
        return df_emissions

    df_emissions["number_positive_outliers"] = (
        df_emissions[delta_years_str] > threshold
    ).sum(axis=1)
    df_emissions["number_negative_outliers"] = (
        df_emissions[delta_years_str] < -threshold
    ).sum(axis=1)
    df_emissions["number_all_outliers"] = (
        (df_emissions[delta_years_str] > threshold)
        | (df_emissions[delta_years_str] < -threshold)
    ).sum(axis=1)

    return df_emissions


# ajouter les ajustements

adjustment_1 = st.toggle("Adjustment 1: keep only the 3 last observations")
adjustment_2 = st.toggle("Adjustment 2: remove unique outlier when enough observations")
adjustment_3 = st.toggle("Adjustment 3: replace outliers with financial intensities")
adjustment_4 = st.toggle("Adjustment 4: replace outliers with physical intensities")

# sans ajustement
df_adjusted = df_eq_abs_em_rate_select.copy()
df_adjusted = create_columns_number_outliers(df_adjusted, threshold_1)
df_adjusted["nbr_observations"] = df_adjusted[delta_years_str].notna().sum(axis=1)


def unique_outlier_or_not_if_nbr_obs_inf_3(
    df_emissions, threshold, delta_years_str=delta_years_str
):
    """
    don t consider one extreme observation (<-20% or >20%)  if the others are enough in number (>=3) or non-extreme. (replace with np.nan)
    """
    if threshold is None:
        return df_emissions

    df_emissions["can_remove_outliers"] = df_emissions.apply(
        lambda x: (
            True
            if (
                (x["nbr_observations"] >= 3 and (x["number_all_outliers"] < 2))
                or (x["nbr_observations"] in [1, 2] and (x["number_all_outliers"] == 0))
            )
            else False
        ),
        axis=1,
    )
    df_adjusted = df_emissions.copy()
    df_adjusted.loc[
        df_adjusted["can_remove_outliers"], delta_years_str
    ] = df_adjusted.loc[df_adjusted["can_remove_outliers"], delta_years_str].mask(
        (df_adjusted[delta_years_str] > threshold)
        | (df_adjusted[delta_years_str] < -threshold)
    )
    # df_adjusted = df_adjusted.drop('can_remove_outliers', axis=1)

    return df_adjusted


def keep_only_the_3_last_observations(df_emissions, delta_years_str=delta_years_str):

    df_adjusted = df_emissions.copy()
    df_adjusted[delta_years_str[:-3]] = np.nan

    return df_adjusted


def replace_emissions_with_intensities(
    df_emissions, df_itensities, threshold, delta_years_str=delta_years_str
):
    """Both df are already filtered for scopes, one row by equity"""

    df_adjusted = df_emissions.copy()
    df_adjusted = df_adjusted.set_index("isin")
    df_itensities = df_itensities.set_index("isin").reindex(df_adjusted.index)
    df_adjusted[delta_years_str] = df_adjusted[delta_years_str].mask(
        (
            (df_adjusted[delta_years_str] > threshold)
            | (df_adjusted[delta_years_str] < -threshold)
        )
        & (df_itensities[delta_years_str].notna()),
        df_itensities[delta_years_str],
    )

    df_adjusted = df_adjusted.reset_index()

    return df_adjusted


adjusted = False
if adjustment_1:

    df_adjusted = keep_only_the_3_last_observations(df_adjusted)
    df_adjusted = create_columns_number_outliers(df_adjusted, threshold_1)
    df_adjusted["nbr_observations"] = df_adjusted[delta_years_str].notna().sum(axis=1)
    adjusted = True

if adjustment_2:

    df_adjusted = unique_outlier_or_not_if_nbr_obs_inf_3(df_adjusted, threshold_1)
    df_adjusted = create_columns_number_outliers(df_adjusted, threshold_1)
    df_adjusted["nbr_observations"] = df_adjusted[delta_years_str].notna().sum(axis=1)
    adjusted = True

if adjustment_3:

    df_adjusted = replace_emissions_with_intensities(
        df_adjusted, df_eq_intensities_rate_select, threshold_1
    )
    df_adjusted = create_columns_number_outliers(df_adjusted, threshold_1)
    df_adjusted["nbr_observations"] = df_adjusted[delta_years_str].notna().sum(axis=1)
    adjusted = True

if adjustment_4:

    df_adjusted = replace_emissions_with_intensities(
        df_adjusted, df_eq_auto_intensities_rate_select, threshold_1
    )
    df_adjusted = create_columns_number_outliers(df_adjusted, threshold_1)
    df_adjusted["nbr_observations"] = df_adjusted[delta_years_str].notna().sum(axis=1)
    adjusted = True


# montrer le tableau des stats
st.write("**Emissions growth rate after adjustment - statistics**")

df_adjusted["average_rate"] = df_adjusted[delta_years_str].mean(axis=1)

df_adj_stat = df_adjusted.groupby("scope")["average_rate"].describe()
df_adj_stat_fmt = df_adj_stat.copy()
df_adj_stat_fmt["count"] = df_adj_stat_fmt["count"].astype(int)
df_adj_stat_fmt["std"] = df_adj_stat_fmt["std"].map(lambda x: f"{x:.2e}")
cols_pct = df_adj_stat_fmt.columns.difference(["count", "std"])
df_adj_stat_fmt[cols_pct] = df_adj_stat_fmt[cols_pct].map(lambda x: f"{x:.1%}")


st.dataframe(df_adj_stat_fmt, height=row_height * (len(df_adj_stat_fmt) + 1))

#  montrer le tableau description outliers
if adjusted:
    st.write(
        "**Number of outliers after adjustment (number of equity with at least one extreme observation)**"
    )
else:
    st.write(
        "**Number of outliers (number of equity with at least one extreme observation)**"
    )

df_adj_des = pd.DataFrame(
    {"negative outliers": None, "positive outliers": None, "both": None},
    index=["Number of equities"],
)

if threshold_1 is not None:

    df_adj_des = df_adjusted.copy()
    df_adj_des = pd.DataFrame(
        {
            f"inf to {-threshold_1:.1%}": [
                (df_adj_des["number_negative_outliers"] > 0).sum()
            ],
            f"sup to  {threshold_1:.1%}": [
                (df_adj_des["number_positive_outliers"] > 0).sum()
            ],
            "both": [(df_adj_des["number_all_outliers"] > 0).sum()],
        },
        index=["Number of equities"],
    )

st.dataframe(df_adj_des, height=row_height * (len(df_adj_des) + 1))

#  montrer le tableau avec le nombre d'observations et le nombre d'outliers
df_adj_nbr_outliers_by_nbr_obs = pd.DataFrame(index=range(5), columns=range(5))

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    if adjusted:
        st.write("**Number of outliers by number of observations after adjustment**")
    else:
        st.write("**Number of outliers by number of observations**")
with col2:
    with_negative_outliers = st.toggle(
        "negative outliers", key="with_negative_outliers", value=True
    )
with col3:
    with_positive_outliers = st.toggle(
        "positive outliers", key="with_positive_outliers", value=True
    )

if threshold_1 is not None:

    if with_negative_outliers & with_positive_outliers:
        df_adj_nbr_outliers_by_nbr_obs = pd.crosstab(
            [df_adjusted["nbr_observations"]], df_adjusted["number_all_outliers"]
        )
    elif with_negative_outliers:
        df_adj_nbr_outliers_by_nbr_obs = pd.crosstab(
            [df_adjusted["nbr_observations"]], df_adjusted["number_negative_outliers"]
        )
    elif with_positive_outliers:
        df_adj_nbr_outliers_by_nbr_obs = pd.crosstab(
            [df_adjusted["nbr_observations"]], df_adjusted["number_positive_outliers"]
        )

st.dataframe(
    df_adj_nbr_outliers_by_nbr_obs,
    height=row_height * (len(df_adj_nbr_outliers_by_nbr_obs) + 1),
)
