import streamlit as st
import pandas as pd
import numpy as np

# ----------------------------
# Run Streamlit:
# streamlit run ca_data_statistics_streamlit.py
# ----------------------------


df = pd.read_parquet('output/df_merged_all_infos.parquet')
scopes = ["s1", "s2", "s3_d", "s3_u"]
df = df[df['scope'].isin(scopes)]
df["scope"] = pd.Categorical(df["scope"], categories=scopes, ordered=True)

regions = list(df['region_0'].unique())
list_high_impact_sector = list(df['high_impact_sector'].dropna().sort_values().unique())


# ----------------------------
# MISSING
# ----------------------------
st.subheader("Missing")
st.write(
    'nbr eq with missing sector:', 
    f"{df[df['high_impact_sector'].isna()]['isin'].nunique()}/{df['isin'].nunique()}")
st.write('percentage market cap with missing sector:', 
         f"{df[df['high_impact_sector'].isna()]['company_free_float_market_cap'].sum()/df['company_free_float_market_cap'].sum():.1%}")

# ----------------------------
# TABLE 1 - Total equities
# ----------------------------

st.header("Universe distribution")

choice = st.radio(
    "Choose display option",
    options=["Number", f"Market-cap (% of the total)"],
    horizontal=True
)

df_one_scp = df[df['scope']=="s1"]

if choice=="Number":

    st.subheader("Total number of equity")
    df_groupby = df_one_scp.groupby(['high_impact_sector','region_0'])['isin'].count().reset_index()
    total_by_sector = df_groupby.groupby('high_impact_sector')['isin'].sum().reset_index()
    total_by_sector["region_0"] = "Total"
    total_by_region = df_groupby.groupby('region_0')['isin'].sum().reset_index()
    total_by_region["high_impact_sector"] = "Total"
    df_groupby = pd.concat([df_groupby, total_by_sector,total_by_region], axis=0)
    df_pivot = df_groupby.pivot(
                index="high_impact_sector",
                columns="region_0",
                values="isin",
            )
    df_pivot = df_pivot.loc[list_high_impact_sector + ["Total"]]
    df_pivot.loc["Total", "Total"] = df_groupby[~((df_groupby['region_0']=='Total')|(df_groupby['high_impact_sector']=='Total'))]['isin'].sum()

    row_height = 35
    st.dataframe(df_pivot, height=row_height * (len(df_pivot) + 1))

if choice==f"Market-cap (% of the total)":

    st.subheader("Market cap")

    total_market_cap = df_one_scp['company_free_float_market_cap'].sum()
    df_2_groupby = df_one_scp.groupby(['high_impact_sector','region_0'])['company_free_float_market_cap'].sum().reset_index()
    total_by_sector = df_2_groupby.groupby('high_impact_sector')['company_free_float_market_cap'].sum().reset_index()
    total_by_sector["region_0"] = "Total"
    total_by_region = df_2_groupby.groupby('region_0')['company_free_float_market_cap'].sum().reset_index()
    total_by_region["high_impact_sector"] = "Total"
    df_2_groupby = pd.concat([df_2_groupby, total_by_sector,total_by_region], axis=0)

    df_2_groupby_perc = df_2_groupby.copy()
    df_2_groupby_perc['percentage_mc'] =  df_2_groupby['company_free_float_market_cap'] / total_market_cap
    df_2_groupby_perc = df_2_groupby_perc.drop('company_free_float_market_cap', axis=1)


    df_2_pivot = df_2_groupby_perc.pivot(
                index="high_impact_sector",
                columns="region_0",
                values="percentage_mc",
            )
    df_2_pivot = df_2_pivot.loc[list_high_impact_sector + ["Total"]]
    df_2_pivot.loc["Total", "Total"] = df_2_groupby[~((df_2_groupby['region_0']=='Total')|(df_2_groupby['high_impact_sector']=='Total'))]['company_free_float_market_cap'].sum() / total_market_cap

    row_height = 35
    st.dataframe(
        df_2_pivot.map(lambda x: f"{x:.1%}" if pd.notnull(x) else ""),
        height=row_height * (len(df_2_pivot) + 1)
    )


# ----------------------------
# TABLE 2 - Stats on market cap
# ----------------------------

st.header("Market cap - stat des. ")

selected_region_1 = st.multiselect(
    "Region", options=regions, placeholder="Choose one or multiple options", 
    key='region_1'
)
selected_hisector_1 = st.multiselect(
    "High impact sector", options=list_high_impact_sector, 
    placeholder="Choose one or multiple options",
    key='hisector_1'
)

if not selected_region_1:
    selected_region_1 = regions
if not selected_hisector_1:
    selected_hisector_1 = list_high_impact_sector

df_selection = df_one_scp[(df_one_scp['region_0'].isin(selected_region_1))&(
    df_one_scp['high_impact_sector'].isin(selected_hisector_1))]

st.write("market cap du bloc sur la market cap totale :", 
         f"{df_selection['company_free_float_market_cap'].sum() / df_one_scp['company_free_float_market_cap'].sum():.1%}")

df_selection = df_selection.copy()
df_selection['w_mc_selection'] = df_selection['company_free_float_market_cap'] / df_selection['company_free_float_market_cap'].sum()
df_stats = df_selection['w_mc_selection'].describe()
df_fmt = df_stats.copy()
df_fmt.loc["count"] = f"{int(df_stats.loc['count'])}"
df_fmt.loc["std"] = f"{df_stats.loc['std']:.2e}"
df_fmt.loc[~df_fmt.index.isin(["count","std"])] = df_stats.loc[~df_stats.index.isin(["count","std"])].map(lambda x: f"{x:.1%}")

st.subheader("Statistics des poids (en market cap) dans ce bloc")
st.dataframe(df_fmt.T, height=row_height * (len(df_fmt) + 1))

# ----------------------------
# TABLE 3 - Absolute emissions - growth rate statistics
# ----------------------------

df_abs_em = pd.read_parquet('data/transformed_data/hist_abs_emissions_growth_rate.parquet')
scopes = ["s1", "s2", "s3_d", "s3_u"]
df_abs_em = df_abs_em[df_abs_em['scope'].isin(scopes)]
df_abs_em["scope"] = df_abs_em["scope"].cat.set_categories(scopes)

df_eq_abs_em = pd.merge(df[['isin', 'scope','high_impact_sector', 'region_0']], df_abs_em, how='left', on=['isin', 'scope'])

years_str = sorted(
    [c for c in df_abs_em.columns if c.isdigit()],
    key=int, reverse=False)

st.header("Absolute emissions - stat des. ")

selected_region_2 = st.multiselect(
    "Region", options=regions, placeholder="Choose one or multiple options",
    key='region_2'
)
selected_hisector_2 = st.multiselect(
    "High impact sector", options=list_high_impact_sector, 
    placeholder="Choose one or multiple options",
    key='hisector_2'
)

if not selected_region_2:
    selected_region_2 = regions
if not selected_hisector_2:
    selected_hisector_2 = list_high_impact_sector

df_selection = df_eq_abs_em[(df_eq_abs_em['region_0'].isin(selected_region_2))&(
    df_eq_abs_em['high_impact_sector'].isin(selected_hisector_2))]

df_selection = df_selection[['scope'] + years_str]
df_selection['last_available_year'] = df_selection[years_str[::-1]].bfill(axis=1).iloc[:, 0]

df_selection_melt = df_selection.melt(
    id_vars="scope", var_name="year", value_name="growth_rate")

st.subheader("Statistics des annual growth rate dans ce bloc et pour la combinaison des années séléctionnées")

selected_year = st.multiselect(
    "Year", options=['last_available_year'] + years_str, 
    placeholder="Choose one or multiple options", key='year_abs'
)
if not selected_year:
    selected_year = years_str

df_selection_melt_filtered = df_selection_melt[df_selection_melt['year'].isin(selected_year)]
df_stats_2 = df_selection_melt_filtered.groupby('scope')['growth_rate'].describe()
df_fmt_2 = df_stats_2.copy()
df_fmt_2["count"] = df_stats_2["count"].astype(int)
df_fmt_2["std"] = df_stats_2["std"].map(lambda x: f"{x:.2e}")
cols_pct = df_fmt_2.columns.difference(["count", "std"])
df_fmt_2[cols_pct] = df_stats_2[cols_pct].applymap(lambda x: f"{x:.1%}")


st.dataframe(df_fmt_2, height=row_height * (len(df_fmt_2) + 1))

# ----------------------------
# TABLE 4 - Absolute emissions - delta growth rate statistics
# ----------------------------
df_selection_delta = df_selection.copy()
col_delta_years = []
for y, y_pre in zip(years_str[1:], years_str[:-1]):
    name_col_delta = f"{y}-{y_pre}"
    col_delta_years.append(name_col_delta)
    df_selection_delta[name_col_delta] = df_selection_delta[y] - df_selection_delta[y_pre]
df_selection_delta = df_selection_delta.drop(['last_available_year'] + years_str, axis=1)

df_selection_delta_melt = df_selection_delta.melt(
    id_vars="scope", var_name="delta_years", value_name="delta_growth_rate")

st.subheader("Statistics des annual delta growth rate dans ce bloc et pour la combinaison des années séléctionnées")

selected_delta_years = st.multiselect(
    "Delta years", options=col_delta_years, 
    placeholder="Choose one or multiple options", key='delta_years_abs'
)
if not selected_delta_years:
    selected_delta_years = col_delta_years

df_selection_delta_melt_filtered = df_selection_delta_melt[df_selection_delta_melt['delta_years'].isin(selected_delta_years)]
df_stats_3 = df_selection_delta_melt_filtered.groupby('scope')['delta_growth_rate'].describe()
df_fmt_3 = df_stats_3.copy()
df_fmt_3["count"] = df_stats_3["count"].astype(int)
df_fmt_3["std"] = df_stats_3["std"].map(lambda x: f"{x:.2e}")
cols_pct = df_fmt_3.columns.difference(["count", "std"])
df_fmt_3[cols_pct] = df_stats_3[cols_pct].applymap(lambda x: f"{x:.1%}")

st.dataframe(df_fmt_3, height=row_height * (len(df_fmt_3) + 1))

# ----------------------------
# TABLE 5 - Absolute emissions - nbr outlier
# ----------------------------
df_nmr_outliers_1 = df_selection[years_str].groupby(df_selection["scope"]).count()

st.subheader("Number of outliers")
st.write("choose the threshold that defines the outliers")

comparison = st.radio(
    "Comparison",
    options=["Superior", "Inferior"],
    horizontal=True
)

threshold = st.selectbox(
    "Threshold value",
    options=[None, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
    format_func=lambda x: "Select a value" if x is None else str(x)
)

if threshold is not None:
    if comparison == 'Superior':
        df_nmr_outliers_1 = (df_selection[years_str] >= threshold).groupby(df_selection["scope"]).sum()
    elif comparison == "Inferior":
        df_nmr_outliers_1 = (df_selection[years_str] <= threshold).groupby(df_selection["scope"]).sum()

st.dataframe(df_nmr_outliers_1, height=row_height * (len(df_nmr_outliers_1) + 1))

# ----------------------------
# TABLE 6 - Intensities - growth rate statistics
# ----------------------------
df_intensities = pd.read_parquet('data/transformed_data/hist_intensities_growth_rate.parquet')
scopes = ["s1", "s2", "s3_d", "s3_u"]
df_intensities = df_intensities[df_intensities['scope'].isin(scopes)]
df_intensities["scope"] = df_intensities["scope"].cat.set_categories(scopes)

years_str_int = sorted(
    [c for c in df_intensities.columns if c.isdigit()],
    key=int, reverse=False)

df_eq_intensities = pd.merge(df[['isin', 'scope','high_impact_sector', 'region_0']], 
                             df_intensities[['isin', 'scope'] + years_str_int], how='left', 
                             on=['isin', 'scope'])

st.header("Intensities - stat des. ")

selected_region_3 = st.multiselect(
    "Region", options=regions, placeholder="Choose one or multiple options",
    key='region_3'
)
selected_hisector_3 = st.multiselect(
    "High impact sector", options=list_high_impact_sector, 
    placeholder="Choose one or multiple options",
    key='hisector_3'
)

if not selected_region_3:
    selected_region_3 = regions
if not selected_hisector_3:
    selected_hisector_3 = list_high_impact_sector

df_selection_int = df_eq_intensities[(df_eq_intensities['region_0'].isin(selected_region_3))&(
    df_eq_intensities['high_impact_sector'].isin(selected_hisector_3))]

df_selection_int = df_selection_int[['scope'] + years_str_int]
df_selection_int['last_available_year'] = df_selection_int[years_str_int[::-1]].bfill(axis=1).iloc[:, 0]

df_selection_int_melt = df_selection_int.melt(
    id_vars="scope", var_name="year", value_name="growth_rate")

st.subheader("Statistics des annual growth rate des intensities dans ce bloc et pour la combinaison des années séléctionnées")

selected_year_int = st.multiselect(
    "Year", options=['last_available_year'] + years_str_int, 
    placeholder="Choose one or multiple options", key='year_int'
)
if not selected_year_int:
    selected_year_int = years_str_int

df_selection_int_melt_filtered = df_selection_int_melt[df_selection_int_melt['year'].isin(selected_year_int)]
df_stats_4 = df_selection_int_melt_filtered.groupby('scope')['growth_rate'].describe()
df_fmt_4 = df_stats_4.copy()
df_fmt_4["count"] = df_stats_4["count"].astype(int)
df_fmt_4["std"] = df_stats_4["std"].map(lambda x: f"{x:.2e}")
cols_pct = df_fmt_4.columns.difference(["count", "std"])
df_fmt_4[cols_pct] = df_stats_4[cols_pct].applymap(lambda x: f"{x:.1%}")


st.dataframe(df_fmt_4, height=row_height * (len(df_fmt_4) + 1))

# ----------------------------
# TABLE 7 - Intensities - delta growth rate statistics
# ----------------------------
df_selection_int_delta = df_selection_int.copy()
col_delta_years_int = []
for y, y_pre in zip(years_str_int[1:], years_str_int[:-1]):
    name_col_delta = f"{y}-{y_pre}"
    col_delta_years_int.append(name_col_delta)
    df_selection_int_delta[name_col_delta] = df_selection_int_delta[y] - df_selection_int_delta[y_pre]
df_selection_int_delta = df_selection_int_delta.drop(['last_available_year'] + years_str_int, axis=1)

df_selection_int_delta_melt = df_selection_int_delta.melt(
    id_vars="scope", var_name="delta_years", value_name="delta_growth_rate")

st.subheader("Statistics des annual delta growth rate dans ce bloc et pour la combinaison des années séléctionnées")

selected_delta_years_int = st.multiselect(
    "Delta years", options=col_delta_years_int, 
    placeholder="Choose one or multiple options", key='delta_years_int'
)
if not selected_delta_years_int:
    selected_delta_years_int = col_delta_years_int

df_selection_int_delta_melt_filtered = df_selection_int_delta_melt[df_selection_int_delta_melt['delta_years'].isin(selected_delta_years_int)]
df_stats_5 = df_selection_int_delta_melt_filtered.groupby('scope')['delta_growth_rate'].describe()
df_fmt_5 = df_stats_5.copy()
df_fmt_5["count"] = df_stats_5["count"].astype(int)
df_fmt_5["std"] = df_stats_5["std"].map(lambda x: f"{x:.2e}")
cols_pct = df_fmt_5.columns.difference(["count", "std"])
df_fmt_5[cols_pct] = df_stats_5[cols_pct].applymap(lambda x: f"{x:.1%}")

st.dataframe(df_fmt_5, height=row_height * (len(df_fmt_5) + 1))

# ----------------------------
# TABLE 7 - Ratio - abs emissions
# ----------------------------

st.header("Ratio - stat des. ")

selected_region_4 = st.multiselect(
    "Region", options=regions, placeholder="Choose one or multiple options",
    key='region_4'
)
selected_hisector_4 = st.multiselect(
    "High impact sector", options=list_high_impact_sector, 
    placeholder="Choose one or multiple options",
    key='hisector_4'
)

if not selected_region_4:
    selected_region_4 = regions
if not selected_hisector_4:
    selected_hisector_4 = list_high_impact_sector

df_selection_ratio = df_eq_abs_em[(df_eq_abs_em['region_0'].isin(selected_region_4))&(
    df_eq_abs_em['high_impact_sector'].isin(selected_hisector_4))]
df_selection_ratio = df_selection_ratio.copy()
df_selection_ratio[years_str] = df_selection_ratio[years_str].replace(0, np.nan) 

df_ratio_s2_over_s1 = (
    df_selection_ratio.groupby("isin")[["scope"] + years_str]
    .apply(lambda g: pd.Series(
            g.loc[g["scope"] == "s2", years_str ].values[0]
            / g.loc[g["scope"] == "s1", years_str ].values[0],
            index=years_str)))
df_ratio_s3_d_over_s1 = (
    df_selection_ratio.groupby("isin")[["scope"] + years_str]
    .apply(lambda g: pd.Series(
            g.loc[g["scope"] == "s3_d", years_str ].values[0]
            / g.loc[g["scope"] == "s1", years_str ].values[0],
            index=years_str)))
df_ratio_s3_u_over_s1 = (
    df_selection_ratio.groupby("isin")[["scope"] + years_str]
    .apply(lambda g: pd.Series(
            g.loc[g["scope"] == "s3_u", years_str ].values[0]
            / g.loc[g["scope"] == "s1", years_str ].values[0],
            index=years_str)))
df_ratio_s2_over_s1['last_available_year'] = df_ratio_s2_over_s1[years_str[::-1]].bfill(axis=1).iloc[:, 0]
df_ratio_s3_d_over_s1['last_available_year'] = df_ratio_s3_d_over_s1[years_str[::-1]].bfill(axis=1).iloc[:, 0]
df_ratio_s3_u_over_s1['last_available_year'] = df_ratio_s3_u_over_s1[years_str[::-1]].bfill(axis=1).iloc[:, 0]

selected_year_ratio = 'last_available_year'

df_ratio_final = pd.concat([df_ratio_s2_over_s1[selected_year_ratio].describe().rename('s2/s1'), 
           df_ratio_s3_d_over_s1[selected_year_ratio].describe().rename('s3_d/s1'), 
           df_ratio_s3_u_over_s1[selected_year_ratio].describe().rename('s3_u/s1')], axis=1)

df_ratio_final = df_ratio_final.map(lambda x: f"{x:.1f}")

st.dataframe(df_ratio_final, height=row_height * (len(df_ratio_final) + 1))