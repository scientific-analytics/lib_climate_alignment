import streamlit as st
import pandas as pd

# ----------------------------
# Run Streamlit:
# streamlit run analysis/climate_alignment_streamlit.py
# ----------------------------

# ----------------------------
# CODE
# ----------------------------
df = pd.read_parquet("output/df_merged_all_infos_2.parquet")
df_hist_trends = pd.read_csv(
    "data/intermediate_data/region_sector_historical_trends.csv"
)

print(df["isin"].nunique())
regions = list(df["region_0"].unique())

st.title("Climate alignment index - Block analysis")

st.subheader("Parameters")
# ----------------------------
# Manage missing
# ----------------------------
df["is_no_target"] = df["is_no_target"].fillna(True)

min_1_growth_rates = st.toggle(
    "At least 1 emission growth rate available (in the past 5 years)", value=False
)

if min_1_growth_rates:
    df = df[df["nbr_rate_available"] >= 1]

# ----------------------------
# Choice of scopes
# ----------------------------

choice_scopes = ["Only relevant scopes"] + [
    "s1",
    "s2",
    "s3",
    "s3_u",
    "s3_d",
    "s12",
    "s123",
    "s123_u",
    "s123_d",
    "s13",
    "s23",
    "s13_u",
    "s13_d",
    "s23_u",
    "s23_d",
]
selected_scope = st.selectbox("Select scope", options=choice_scopes, index=0)
print(df["isin"].nunique())
if selected_scope == "Only relevant scopes":
    df = df[df["is_relevant_scopes"] == True]
else:
    df = df[df["scope"] == selected_scope]

print(df["isin"].nunique())
print(df[df["is_relevant_scopes"].isna()]["isin"].nunique())
# ----------------------------
# Choice target ambition level
# ----------------------------

dict_mapping_targets_ctgry = {
    "Ambitious Target": "is_ambitious_target",
    "Approved SBT": "is_approved_sbt_target",
    "Committed SBT": "is_committed_sbt_target",
    "Non-Ambitious Target": "is_target_non_ambitious",
    "No Target": "is_no_target",
}

selected_labels = st.multiselect(
    "Target ambition category",
    options=list(dict_mapping_targets_ctgry.keys()),
    placeholder="Choose one or multiple options",
)

selected_targets = [dict_mapping_targets_ctgry[label] for label in selected_labels]
df_2 = df.copy()
df_2 = df_2.merge(
    df_hist_trends.rename(columns={"region": "region_0"})[
        [
            "region_0",
            "high_impact_sector",
            "average_trend_2014_2024",
            "average_trend_2021_2024",
        ]
    ],
    on=["region_0", "high_impact_sector"],
    how="left",
)

if (
    set(selected_labels) == set(list(dict_mapping_targets_ctgry.keys()))
    or not selected_targets
):  # no selection by default
    df_2["has_selected_target"] = True
else:
    df_2["has_selected_target"] = df_2[selected_targets].fillna(False).any(axis=1)

# ----------------------------
# Choice of annual average growth rate borns
# ----------------------------

col1, col2 = st.columns([3, 1])

# def set_2021_2024():
#     if st.session_state.thr_hist_trend_2021_2024:
#         st.session_state.thr_hist_trend_2014_2024 = False

# def set_2014_2024():
#     if st.session_state.thr_hist_trend_2014_2024:
#         st.session_state.thr_hist_trend_2021_2024 = False

with col1:
    is_threshold_based_on_hist_trend = st.toggle(
        "Max average annual growth threshold based on historical trends",
        key="thr_hist_trend",
    )
if is_threshold_based_on_hist_trend:
    with col2:
        selected_period = st.selectbox(
            "period_hist_trend",
            label_visibility="collapsed",
            options=["2021:2024", "2014:2024"],
            placeholder="Choose the period",
            key="period",
        )

# with col2:
#     is_threshold_based_on_hist_trend_2021_2024 = st.toggle(
#         "Max average annual growth threshold based on historical trends (2021–2024)",
#         key="thr_hist_trend_2021_2024",
#         on_change=set_2021_2024
#     )
# with col3:
#     is_threshold_based_on_hist_trend_2014_2024 = st.toggle(
#         "Threshold based on historical trends (2014–2024)",
#         key="thr_hist_trend_2014_2024",
#         on_change=set_2014_2024
#     )

col_average_rate = "average_rate"
st.caption(
    f"Minimum average annual growth rate= {df_2[col_average_rate].min():.1f}, \
        Maximum average annual growth rate = {df_2[col_average_rate].max():.2e}"
)

if is_threshold_based_on_hist_trend is False:

    col1, col2 = st.columns(2)
    with col1:
        min_rate = st.number_input(
            "Min average annual growth rate",
            value=None,
            step=0.05,
            format="%.2f",
            key="min_rate",
        )
    with col2:
        max_rate = st.number_input(
            "Max average annual growth rate",
            value=None,
            step=0.05,
            format="%.2f",
            key="max_rate",
        )

    df_2[col_average_rate] = pd.to_numeric(df_2[col_average_rate], errors="coerce")

    if min_rate is not None and (max_rate is not None):
        df_2["is_rate_in_range"] = df_2[col_average_rate].between(
            min_rate, max_rate, inclusive="both"
        )
    else:
        df_2["is_rate_in_range"] = True

else:

    if selected_period == "2021:2024":
        col_average_trend = "average_trend_2021_2024"

    elif selected_period == "2021:2024":
        col_average_trend = "average_trend_2014_2024"

    df_2["is_rate_in_range"] = df_2.apply(
        lambda x: True if x[col_average_rate] <= x[col_average_trend] else False, axis=1
    )

# ----------------------------
# Choice of annual average expected growth rate borns
# ----------------------------

col1, col2 = st.columns(2)
with col1:
    min_exp_rate = st.number_input(
        "Min expected average annual growth rate",
        value=None,
        step=0.05,
        format="%.2f",
        key="min_exp_rate",
    )
with col2:
    max_exp_rate = st.number_input(
        "Max expected average annual growth rate",
        value=None,
        step=0.05,
        format="%.2f",
        key="max_exp_rate",
    )

st.caption(
    f"Minimum expected average annual growth rate= {df_2["average_expected_rate"].min():.1f}, \
    Maximum average annual growth rate = {df_2["average_expected_rate"].max():.1f}"
)

df_2["average_expected_rate"] = pd.to_numeric(
    df_2["average_expected_rate"], errors="coerce"
)

if min_exp_rate is not None and (max_exp_rate is not None):
    df_2["is_exp_rate_in_range"] = df_2["average_expected_rate"].between(
        min_exp_rate, max_exp_rate, inclusive="both"
    )
else:
    df_2["is_exp_rate_in_range"] = True

print(df_2[df_2["is_exp_rate_in_range"]]["isin"].nunique())
st.caption(
    "(parfois on sait qu'il y a des targets set mais on a manqué d'info pour calculer le expected rate. / parfois on a des donnees d'emissions mais pas de targets)"
)

### All parameters
df_2["has_all_param"] = df_2[
    ["has_selected_target", "is_rate_in_range", "is_exp_rate_in_range"]
].all(axis=1)

# ----------------------------
# Final table
# ----------------------------

choice = st.radio(
    "Choose display option", options=["Number", "Weight"], horizontal=True
)

# Choice of the region and the sector
regions_order = [
    "North America",
    "Developed Europe",
    "Other Developed",
    "Emergings",
] + ["Total"]
regions_0_order = [
    "Dev Europe Other",
    "Dev Europe EU",
    "Dev Europe GB",
    "Dev America US",
    "Dev America CA",
    "Dev Asia-Pac Other",
    "Dev Asia-Pac JP",
    "Emg Europe",
    "Emg Asia-Pac IN",
    "Emg Asia-Pac CN",
    "Emg Asia-Pac Other",
    "Emg America",
] + ["Total"]

his_order = [
    "Agriculture, forestry and fishing",
    "Airlines",
    "Aluminium",
    "Automobiles",
    "Banking",
    "Cement",
    "Chemicals",
    "Coal mining",
    "Consumer goods & services",
    "Diversified mining",
    "Electric utilities",
    "Food producers",
    "Industrials",
    "Oil and gas",
    "Paper",
    "Real estate",
    "Shipping",
    "Steel",
    "Transportation",
    "non-HIMS",
] + ["Total"]

col1, col2 = st.columns(2)
with col1:
    selected_region_0 = st.multiselect(
        "Region(s)",
        options=["All"] + regions_0_order[:-1],
        placeholder="Choose one or multiple options",
        key="region_0",
    )
with col2:
    selected_hisector_0 = st.multiselect(
        "High impact sector(s)",
        options=["All"] + his_order,
        placeholder="Choose one or multiple options",
        key="hisector_0",
    )

all_regions = False
all_hisectors = False

if "All" in selected_region_0:
    all_regions = True
if "All" in selected_hisector_0:
    all_hisectors = True

if (not selected_region_0) or "All" in selected_region_0:
    selected_region_0 = regions_0_order
if (not selected_hisector_0) or "All" in selected_hisector_0:
    selected_hisector_0 = his_order


df_2_select = df_2[
    df_2["region_0"].isin(selected_region_0)
    & df_2["high_impact_sector"].isin(selected_hisector_0)
]

if choice == "Number":

    df_groupby_sum = (
        df_2_select.assign(has_all_param=lambda d: d["has_all_param"].fillna(False))
        .groupby(["high_impact_sector", "region_0"])
        .agg(n_has_all_param=("has_all_param", "sum"), n_total=("isin", "count"))
        .reset_index()
    )

    total_n = df_groupby_sum["n_total"].sum()
    total_has_all_param = df_groupby_sum["n_has_all_param"].sum()

    # create row and column "total"
    total_by_region = df_groupby_sum.groupby("region_0", as_index=False)[
        ["n_has_all_param", "n_total"]
    ].sum()
    total_by_region["high_impact_sector"] = "Total"

    total_by_sector = df_groupby_sum.groupby("high_impact_sector", as_index=False)[
        ["n_has_all_param", "n_total"]
    ].sum()
    total_by_sector["region_0"] = "Total"

    df_groupby_sum = pd.concat(
        [df_groupby_sum, total_by_region, total_by_sector], ignore_index=True
    )
    df_groupby_sum = (
        df_groupby_sum.set_index(["high_impact_sector", "region_0"])
        .unstack(fill_value=0)
        .stack(future_stack=True)
        .reset_index()
    )

    df_pivot = df_groupby_sum.assign(
        cell=lambda x: x["n_has_all_param"].astype(int).astype(str)
        + "/"
        + x["n_total"].astype(int).astype(str)
    ).pivot(
        index="high_impact_sector",
        columns="region_0",
        values="cell",
    )

    df_pivot.loc["Total", "Total"] = f"{int(total_has_all_param)}/{int(total_n)}"

if choice == "Weight":

    df_groupby_sum_w = (
        df_2_select.assign(
            w_has_all_param=lambda d: d["r_s_mc_weight"]
            * d["has_all_param"].fillna(False)
        )
        .groupby(["high_impact_sector", "region_0"], as_index=False)
        .agg(w_has_all_param=("w_has_all_param", "sum"))
    )

    df_groupby_sum_w = (
        df_groupby_sum_w.set_index(["high_impact_sector", "region_0"])
        .unstack(fill_value=0)
        .stack(future_stack=True)
        .reset_index()
    )

    df_groupby_region_sum_w = (
        df_2_select.assign(
            r_w_has_all_param=lambda d: d["r_mc_weight"]
            * d["has_all_param"].fillna(False)
        )
        .groupby(["region_0"], as_index=False)
        .agg(w_has_all_param=("r_w_has_all_param", "sum"))
    )
    df_groupby_region_sum_w["high_impact_sector"] = "Total"

    df_groupby_sector_sum_w = (
        df_2_select.assign(
            s_w_has_all_param=lambda d: d["s_mc_weight"]
            * d["has_all_param"].fillna(False)
        )
        .groupby(["high_impact_sector"], as_index=False)
        .agg(w_has_all_param=("s_w_has_all_param", "sum"))
    )
    df_groupby_sector_sum_w["region_0"] = "Total"

    df_groupby_sum_w = pd.concat(
        [df_groupby_sum_w, df_groupby_region_sum_w, df_groupby_sector_sum_w]
    )

    df_pivot = df_groupby_sum_w.assign(
        cell=lambda x: (x["w_has_all_param"] * 100).round(1).astype(str) + "%"
    ).pivot(
        index="high_impact_sector",
        columns="region_0",
        values="cell",
    )

    df_pivot.loc["Total", "Total"] = (
        f'{round((
        (df_2_select["company_free_float_market_cap"]* df_2_select["has_all_param"])/ \
            df_2_select["company_free_float_market_cap"].sum()).sum() * 100, 1)}%'
    )


df_pivot = (
    df_pivot.assign(
        _idx=pd.Categorical(df_pivot.index, categories=his_order, ordered=True)
    )
    .sort_values("_idx")
    .drop(columns="_idx")
)

df_pivot = df_pivot[[c for c in regions_0_order if c in df_pivot.columns]]

if all_regions:
    df_pivot = df_pivot[["Total"]]

if all_hisectors:
    df_pivot = df_pivot.loc[["Total"]]

if choice == "Number":
    st.subheader("Number of equity")
else:
    st.subheader("Market-cap weight of equity")

row_height = 35
st.dataframe(df_pivot, height=row_height * (len(df_pivot) + 1))


# --------------------
# A SUIVRE: Reprendre les regions initiales OK
# --------------------

# creer colonne r0_mc_weight OK
# modifier colonne rregion par region_0 OK
