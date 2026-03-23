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

df_one_scp = df[df['scope']=="s1"]

st.subheader("Universe / total number of equity")
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

# ----------------------------
# PARAMETERS - 
# ----------------------------
st.subheader("Parameters")

col1, col2, col3 = st.columns(3)
with col1:
    selected_region_0 = st.multiselect(
        "Region", options=regions, placeholder="Choose one or multiple options", 
        key='region_0'
    )
with col2:
    selected_hisector_0 = st.multiselect(
        "High impact sector", options=list_high_impact_sector, 
        placeholder="Choose one or multiple options",
        key='hisector_0'
    )

with col3:
    selected_scopes_0 = st.multiselect(
        options=scopes, 
        placeholder="Choose one or multiple scope(s)", key='scope_0'
    )


if not selected_region_0:
    selected_region_0 = regions
if not selected_hisector_0:
    selected_hisector_0 = list_high_impact_sector
if not selected_scopes_0:
    selected_scopes_0 = scopes

st.subheader("Results")

# ----------------------------
# TABLE 2 - Stats on market cap
# ----------------------------

df_selection = df_one_scp[(df_one_scp['region_0'].isin(selected_region_0))&(
    df_one_scp['high_impact_sector'].isin(selected_hisector_0))]

st.write("market cap du bloc sur la market cap totale :", 
         f"{df_selection['company_free_float_market_cap'].sum() / df_one_scp['company_free_float_market_cap'].sum():.1%}")

df_selection = df_selection.copy()
df_selection['w_mc_selection'] = df_selection['company_free_float_market_cap'] / df_selection['company_free_float_market_cap'].sum()
df_stats = df_selection['w_mc_selection'].describe()
df_fmt = df_stats.copy()
df_fmt.loc["count"] = f"{int(df_stats.loc['count'])}"
df_fmt.loc["std"] = f"{df_stats.loc['std']:.2e}"
df_fmt.loc[~df_fmt.index.isin(["count","std"])] = df_stats.loc[~df_stats.index.isin(["count","std"])].map(lambda x: f"{x:.1%}")

st.write("**Statistics des poids (en market cap) dans ce bloc**")
st.dataframe(df_fmt.to_frame().T,hide_index=True) #, height=row_height * (len(df_fmt) + 1))

# ----------------------------
# TABLE 3 - Absolute emissions - statistics
# ----------------------------
df_hist_abs_emissions = pd.read_parquet('data/transformed_data/hist_abs_emissions.parquet')
years_str = sorted(
    [c for c in df_hist_abs_emissions.columns if c.isdigit()],
    key=int, reverse=False)
df_eq_abs_em = pd.merge(df[['isin', 'scope','high_impact_sector', 'region_0']], df_hist_abs_emissions, how='left', on=['isin', 'scope'])

st.write("**Absolute emissions - statistics**")

df_selection_0 = df_eq_abs_em[(df_eq_abs_em['region_0'].isin(selected_region_0))&(
    df_eq_abs_em['high_impact_sector'].isin(selected_hisector_0))]
df_selection_0 = df_selection_0[['scope'] + years_str]
df_selection_0['last_available_year'] = df_selection_0[years_str[::-1]].bfill(axis=1).iloc[:, 0]

df_selection_0_melt = df_selection_0.melt(
    id_vars="scope", var_name="year", value_name="growth_rate")

selected_year_0 = st.multiselect(
    label='', label_visibility="collapsed", options=['last_available_year'] + years_str, 
    placeholder="Choose one or multiple  year(s)", key='year_0'
)
if not selected_year_0:
    selected_year_0 = years_str

df_selection_0_melt_filtered = df_selection_0_melt[df_selection_0_melt['year'].isin(selected_year_0)]
df_stats_0 = df_selection_0_melt_filtered.groupby('scope')['growth_rate'].describe()
df_fmt_0 = df_stats_0.copy()
df_fmt_0["count"] = df_stats_0["count"].astype(int)
df_fmt_0["std"] = df_stats_0["std"].map(lambda x: f"{x:.2e}")
cols_pct = df_fmt_0.columns.difference(["count", "std"])
df_fmt_0[cols_pct] = df_stats_0[cols_pct].applymap(lambda x: f"{x:.2e}")


st.dataframe(df_fmt_0, height=row_height * (len(df_fmt_0) + 1))
# ----------------------------
# TABLE 4 - Absolute emissions - growth rate statistics
# ----------------------------
df_abs_em_rate = pd.read_parquet('data/transformed_data/hist_abs_emissions_growth_rate.parquet')
df_abs_em_rate = df_abs_em_rate[df_abs_em_rate['scope'].isin(scopes)]
df_abs_em_rate["scope"] = df_abs_em_rate["scope"].cat.set_categories(scopes)


delta_years_str = [f"{int(y)-1}-{y}" for y in years_str[1:]]
df_abs_em_rate = df_abs_em_rate.rename(columns={y: f"{int(y)-1}-{y}" for y in years_str[1:]})

df_eq_abs_em_rate = pd.merge(df[['isin', 'scope','high_impact_sector', 'region_0']], df_abs_em_rate, how='left', on=['isin', 'scope'])

st.write("**Absolute emissions growth rate - statistics**")

df_selection_1 = df_eq_abs_em_rate[(df_eq_abs_em_rate['region_0'].isin(selected_region_0))&(
    df_eq_abs_em_rate['high_impact_sector'].isin(selected_hisector_0))]

df_selection_1 = df_selection_1[['scope'] + delta_years_str]
df_selection_1['last_available_year'] = df_selection_1[delta_years_str[::-1]].bfill(axis=1).iloc[:, 0]

df_selection_1_melt = df_selection_1.melt(
    id_vars="scope", var_name="year", value_name="growth_rate")

selected_year_1 = st.multiselect(
    "Year", options=['last_available_year'] + delta_years_str, 
    placeholder="Choose one or multiple options", key='year_1'
)

if not selected_year_1:
    selected_year_1 = delta_years_str

df_selection_1_melt_filtered = df_selection_1_melt[df_selection_1_melt['year'].isin(selected_year_1)]
df_stats_1 = df_selection_1_melt_filtered.groupby('scope')['growth_rate'].describe()
df_fmt_1 = df_stats_1.copy()
df_fmt_1["count"] = df_stats_1["count"].astype(int)
df_fmt_1["std"] = df_stats_1["std"].map(lambda x: f"{x:.2e}")
cols_pct = df_fmt_1.columns.difference(["count", "std"])
df_fmt_1[cols_pct] = df_stats_1[cols_pct].applymap(lambda x: f"{x:.1%}")


st.dataframe(df_fmt_1, height=row_height * (len(df_fmt_1) + 1))


# ----------------------------
# TABLE 5 - Absolute emissions - nbr outlier
# ----------------------------
df_nmr_outliers_0 = df_selection_1[delta_years_str].groupby(df_selection_1["scope"]).count()

st.write("**Absolute emissions growth rate - number of outliers**")

col1, col2, col3, col4 = st.columns([3, 2, 1, 3])

with col1:
    st.write("Choose the threshold that defines the outliers")

with col2:
    comparison = st.radio(label="", options=["superior", "inferior"], horizontal=True)

with col3:
    st.write("to:")

with col4:
    threshold = st.selectbox(
        label="", options=[None, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
        format_func=lambda x: "Select a value" if x is None else f"{x:.1%}")

if threshold is not None:
    if comparison == 'superior':
        df_nmr_outliers_0 = (df_selection_1[delta_years_str] >= threshold).groupby(df_selection_1["scope"]).sum()
    elif comparison == "inferior":
        df_nmr_outliers_0 = (df_selection_1[delta_years_str] <= threshold).groupby(df_selection_1["scope"]).sum()

st.dataframe(df_nmr_outliers_0, height=row_height * (len(df_nmr_outliers_0) + 1))

# ----------------------------
# TABLE 6 - Intensities - statistics
# ----------------------------
st.write("**Intensities - statistics**")

df_hist_intensities = pd.read_parquet('data/transformed_data/hist_intensities.parquet')
df_eq_intensities = pd.merge(df[['isin', 'scope','high_impact_sector', 'region_0']], df_hist_intensities, how='left', on=['isin', 'scope'])

df_selection_2 = df_eq_intensities[(df_eq_intensities['region_0'].isin(selected_region_0))&(
    df_eq_intensities['high_impact_sector'].isin(selected_hisector_0))]
df_selection_2 = df_selection_2[['scope'] + years_str]
df_selection_2['last_available_year'] = df_selection_2[years_str[::-1]].bfill(axis=1).iloc[:, 0]

df_selection_2_melt = df_selection_2.melt(
    id_vars="scope", var_name="year", value_name="growth_rate")

selected_year_2 = st.multiselect(
    label='', label_visibility="collapsed", options=['last_available_year'] + years_str, 
    placeholder="Choose one or multiple  year(s)", key='year_2'
)
if not selected_year_2:
    selected_year_2 = years_str

df_selection_2_melt_filtered = df_selection_2_melt[df_selection_2_melt['year'].isin(selected_year_2)]
df_stats_2 = df_selection_2_melt_filtered.groupby('scope')['growth_rate'].describe()
df_fmt_2 = df_stats_2.copy()
df_fmt_2["count"] = df_stats_2["count"].astype(int)
df_fmt_2["std"] = df_stats_2["std"].map(lambda x: f"{x:.2e}")
cols_pct = df_fmt_2.columns.difference(["count", "std"])
df_fmt_2[cols_pct] = df_stats_2[cols_pct].applymap(lambda x: f"{x:.2e}")


st.dataframe(df_fmt_2, height=row_height * (len(df_fmt_2) + 1))

# ----------------------------
# TABLE 7 - Intensities - growth rate statistics
# ----------------------------
df_intensities_rate = pd.read_parquet('data/transformed_data/hist_intensities_growth_rate.parquet')
df_intensities_rate = df_intensities_rate[df_intensities_rate['scope'].isin(scopes)]
df_intensities_rate["scope"] = df_intensities_rate["scope"].cat.set_categories(scopes)
df_intensities_rate = df_intensities_rate.rename(columns={y: f"{int(y)-1}-{y}" for y in years_str[1:]})


df_eq_intensities_rate = pd.merge(df[['isin', 'scope','high_impact_sector', 'region_0']], 
                             df_intensities_rate[['isin', 'scope'] + delta_years_str], how='left', 
                             on=['isin', 'scope'])

st.write("**Intensities growth rate - statistics**")

df_selection_3 = df_eq_intensities_rate[(df_eq_intensities_rate['region_0'].isin(selected_region_0))&(
    df_eq_intensities_rate['high_impact_sector'].isin(selected_hisector_0))]

df_selection_3 = df_selection_3[['scope'] + delta_years_str]
df_selection_3['last_available_year'] = df_selection_3[delta_years_str[::-1]].bfill(axis=1).iloc[:, 0]

df_selection_3_melt = df_selection_3.melt(
    id_vars="scope", var_name="year", value_name="growth_rate")

selected_year_3 = st.multiselect(
    "Year", options=['last_available_year'] + delta_years_str, 
    placeholder="Choose one or multiple options", key='year_3'
)
if not selected_year_3:
    selected_year_3 = delta_years_str

df_selection_3_melt_filtered = df_selection_3_melt[df_selection_3_melt['year'].isin(selected_year_3)]
df_stats_3 = df_selection_3_melt_filtered.groupby('scope')['growth_rate'].describe()
df_fmt_3 = df_stats_3.copy()
df_fmt_3["count"] = df_stats_3["count"].astype(int)
df_fmt_3["std"] = df_stats_3["std"].map(lambda x: f"{x:.2e}")
cols_pct = df_fmt_3.columns.difference(["count", "std"])
df_fmt_3[cols_pct] = df_stats_3[cols_pct].applymap(lambda x: f"{x:.1%}")


st.dataframe(df_fmt_3, height=row_height * (len(df_fmt_3) + 1))



# ----------------------------
# TABLE 8 - Ratio - delta abs emissions
# ----------------------------

st.write("**Absolute emisssions growth rate - scopes ratio - stat des.**")


df_selection_ratio = df_eq_abs_em_rate[(df_eq_abs_em_rate['region_0'].isin(selected_region_0))&(
    df_eq_abs_em_rate['high_impact_sector'].isin(selected_hisector_0))]
df_selection_ratio = df_selection_ratio.copy()
df_selection_ratio[delta_years_str] = df_selection_ratio[delta_years_str].replace(0, np.nan) 

df_ratio_s2_over_s1 = (
    df_selection_ratio.groupby("isin")[["scope"] + delta_years_str]
    .apply(lambda g: pd.Series(
            g.loc[g["scope"] == "s2", delta_years_str ].values[0]
            / g.loc[g["scope"] == "s1", delta_years_str ].values[0],
            index=delta_years_str)))
df_ratio_s3_d_over_s1 = (
    df_selection_ratio.groupby("isin")[["scope"] + delta_years_str]
    .apply(lambda g: pd.Series(
            g.loc[g["scope"] == "s3_d", delta_years_str ].values[0]
            / g.loc[g["scope"] == "s1", delta_years_str ].values[0],
            index=delta_years_str)))
df_ratio_s3_u_over_s1 = (
    df_selection_ratio.groupby("isin")[["scope"] + delta_years_str]
    .apply(lambda g: pd.Series(
            g.loc[g["scope"] == "s3_u", delta_years_str ].values[0]
            / g.loc[g["scope"] == "s1", delta_years_str ].values[0],
            index=delta_years_str)))
df_ratio_s2_over_s1['last_available_year'] = df_ratio_s2_over_s1[delta_years_str[::-1]].bfill(axis=1).iloc[:, 0]
df_ratio_s3_d_over_s1['last_available_year'] = df_ratio_s3_d_over_s1[delta_years_str[::-1]].bfill(axis=1).iloc[:, 0]
df_ratio_s3_u_over_s1['last_available_year'] = df_ratio_s3_u_over_s1[delta_years_str[::-1]].bfill(axis=1).iloc[:, 0]

selected_year_ratio = 'last_available_year'

df_ratio_final = pd.concat([df_ratio_s2_over_s1[selected_year_ratio].describe().rename('s2/s1'), 
           df_ratio_s3_d_over_s1[selected_year_ratio].describe().rename('s3_d/s1'), 
           df_ratio_s3_u_over_s1[selected_year_ratio].describe().rename('s3_u/s1')], axis=1)

df_ratio_final = df_ratio_final.map(lambda x: f"{x:.1f}")

st.dataframe(df_ratio_final, height=row_height * (len(df_ratio_final) + 1))