import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ----------------------------
# Run Streamlit:
# streamlit run ca_data_statistics_streamlit.py
# ----------------------------


# ----------------------------
# LOAD AND PREPARE DATA
# ----------------------------

df = pd.read_parquet('output/df_merged_all_infos.parquet')
df_hist_abs_emissions = pd.read_parquet('data/transformed_data/hist_abs_emissions.parquet')
df_abs_em_rate = pd.read_parquet('data/transformed_data/hist_abs_emissions_growth_rate.parquet')
df_hist_intensities = pd.read_parquet('data/transformed_data/hist_intensities.parquet')
df_intensities_rate = pd.read_parquet('data/transformed_data/hist_intensities_growth_rate.parquet')

scopes = ["s1", "s2", "s3_d", "s3_u"]
regions = list(df['region_0'].unique())
list_high_impact_sector = list(df['high_impact_sector'].dropna().sort_values().unique())
years_str = sorted([c for c in df_hist_abs_emissions.columns if c.isdigit()], key=int, reverse=False)
dict_year_delta_years = {y: f"{int(y)-1}-{y}" for y in years_str[1:]}
delta_years_str = list(dict_year_delta_years.values())

df_eq_abs_em = pd.merge(df[['isin', 'scope','high_impact_sector', 'region_0']], df_hist_abs_emissions, how='left', on=['isin', 'scope'])
df_eq_abs_em_rate = pd.merge(df[['isin', 'scope','high_impact_sector', 'region_0']], df_abs_em_rate, how='left', on=['isin', 'scope'])
df_eq_intensities = pd.merge(df[['isin', 'scope','high_impact_sector', 'region_0']], df_hist_intensities, how='left', on=['isin', 'scope'])
df_eq_intensities_rate = pd.merge(df[['isin', 'scope','high_impact_sector', 'region_0']], 
                             df_intensities_rate[['isin', 'scope'] + delta_years_str], how='left', 
                             on=['isin', 'scope'])


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

st.subheader("Universe / total number of equity")

df_one_scp = df[df['scope']=="s1"]

df_groupby = df_one_scp.groupby(['high_impact_sector','region_0'])['isin'].count().reset_index()
total_by_sector = df_groupby.groupby('high_impact_sector')['isin'].sum().reset_index()
total_by_sector["region_0"] = "Total"
total_by_region = df_groupby.groupby('region_0')['isin'].sum().reset_index()
total_by_region["high_impact_sector"] = "Total"
df_groupby = pd.concat([df_groupby, total_by_sector, total_by_region], axis=0)
df_pivot = df_groupby.pivot(index="high_impact_sector", columns="region_0", values="isin",)
df_pivot = df_pivot.loc[list_high_impact_sector + ["Total"]]
df_pivot.loc["Total", "Total"] = df_groupby[~((df_groupby['region_0']=='Total')|(df_groupby['high_impact_sector']=='Total'))]['isin'].sum()

row_height = 35
st.dataframe(df_pivot, height=row_height * (len(df_pivot) + 1))


# ----------------------------
# PARAMETERS
# ----------------------------

st.subheader("Parameters")

col1, col2, col3 = st.columns(3)
with col1:
    selected_region_0 = st.multiselect(
        "Region(s)", options=regions, placeholder="Choose one or multiple options", 
        key='region_0')
with col2:
    selected_hisector_0 = st.multiselect(
        "High impact sector(s)", options=list_high_impact_sector, 
        placeholder="Choose one or multiple options",
        key='hisector_0')
with col3:
    selected_scope_0_list = st.multiselect(
        "Scope(s)", options=scopes, 
        placeholder="Choose one or multiple scope(s)", key='scope_0')

if not selected_region_0:
    selected_region_0 = regions
if not selected_hisector_0:
    selected_hisector_0 = list_high_impact_sector
if not selected_scope_0_list:
    selected_scope_0_list = scopes

def scope_label(scopes):
    scopes = set(scopes)
    base = "s" + "".join(x[1] for x in ["s1","s2","s3_u"] if x in scopes or (x=="s3_u" and "s3_d" in scopes))
    if "s3_u" in scopes and "s3_d" not in scopes:
        return base + "_u"
    if "s3_d" in scopes and "s3_u" not in scopes:
        return base + "_d"
    return base

selected_scope_0 = scope_label(selected_scope_0_list)


# ----------------------------
# TABLE 2 - Stats on market cap
# ----------------------------

st.subheader("Results")

df_selection = df_one_scp[(df_one_scp['region_0'].isin(selected_region_0))&(
    df_one_scp['high_impact_sector'].isin(selected_hisector_0))]

st.write("Block (region(s) and (sector(s)) market cap on total universe market cap :", 
         f"{df_selection['company_free_float_market_cap'].sum() / df_one_scp['company_free_float_market_cap'].sum():.1%}")

df_selection = df_selection.copy()
df_selection['w_mc_selection'] = df_selection['company_free_float_market_cap'] / df_selection['company_free_float_market_cap'].sum()
df_stats = df_selection['w_mc_selection'].describe()
df_fmt = df_stats.copy().astype(object)
df_fmt.loc["count"] = f"{int(df_stats.loc['count'])}"
df_fmt.loc["std"] = f"{df_stats.loc['std']:.2e}"
df_fmt.loc[~df_fmt.index.isin(["count","std"])] = df_stats.loc[~df_stats.index.isin(["count","std"])].map(lambda x: f"{x:.1%}")

st.write("**Statistics on the weight (market cap weighted) of this block (region(s) and (sector(s))**")
st.dataframe(df_fmt.to_frame().T,hide_index=True) 


# ----------------------------
# TABLE 3 - Absolute emissions - statistics
# ----------------------------

st.write("**Absolute emissions - statistics**")

df_selection_all_col_0 = df_eq_abs_em[(df_eq_abs_em['region_0'].isin(selected_region_0))&(
    df_eq_abs_em['high_impact_sector'].isin(selected_hisector_0))&(
        df_eq_abs_em['scope']==selected_scope_0)].copy()

df_selection_0 = df_selection_all_col_0[['scope'] + years_str].copy()
df_selection_0['last_available_year'] = df_selection_0[years_str[::-1]].bfill(axis=1).iloc[:, 0]

selected_year_0 = st.multiselect(
    label='selected_year_0', label_visibility="collapsed", options=['last_available_year'] + years_str, 
    placeholder="Choose one or multiple  year(s)", key='year_0'
)
if not selected_year_0:
    selected_year_0 = years_str

df_selection_0_mean = pd.concat([df_selection_0['scope'], df_selection_0[selected_year_0].mean(axis=1).rename("mean_value")], axis=1)
df_stats_0 = df_selection_0_mean.groupby('scope')['mean_value'].describe()

df_fmt_0 = df_stats_0.copy()
df_fmt_0["count"] = df_stats_0["count"].astype(int)
df_fmt_0["std"] = df_stats_0["std"].map(lambda x: f"{x:.2e}")
cols_pct = df_fmt_0.columns.difference(["count", "std"])
df_fmt_0[cols_pct] = df_stats_0[cols_pct].map(lambda x: f"{x:.2e}")

st.dataframe(df_fmt_0, height=row_height * (len(df_fmt_0) + 1))


# ----------------------------
# TABLE 4 - Absolute emissions - growth rate statistics
# ----------------------------

st.write("**Absolute emissions growth rate - statistics**")

df_selection_all_col_1 = df_eq_abs_em_rate[(df_eq_abs_em_rate['region_0'].isin(selected_region_0))&(
    df_eq_abs_em_rate['high_impact_sector'].isin(selected_hisector_0))&(
        df_eq_abs_em_rate['scope']==selected_scope_0)].copy()

df_selection_all_col_1['last_available_year'] = df_selection_all_col_1[delta_years_str[::-1]].bfill(axis=1).iloc[:, 0]

df_selection_1 = df_selection_all_col_1[['scope'] + delta_years_str + ['last_available_year']].copy()

selected_year_1 = st.multiselect(
    label='selected_year_1', label_visibility="collapsed", options=['last_available_year'] + delta_years_str, 
    placeholder="Choose one or multiple years range(s)", key='year_1')

if not selected_year_1:
    selected_year_1 = delta_years_str

df_selection_1_mean = pd.concat([df_selection_1['scope'], df_selection_1[selected_year_1].mean(axis=1).rename("mean_value")], axis=1)
df_stats_1 = df_selection_1_mean.groupby('scope')['mean_value'].describe()

df_fmt_1 = df_stats_1.copy()
df_fmt_1["count"] = df_stats_1["count"].astype(int)
df_fmt_1["std"] = df_stats_1["std"].map(lambda x: f"{x:.2e}")
cols_pct = df_fmt_1.columns.difference(["count", "std"])
df_fmt_1[cols_pct] = df_stats_1[cols_pct].map(lambda x: f"{x:.1%}")

st.dataframe(df_fmt_1, height=row_height * (len(df_fmt_1) + 1))


# ----------------------------
# TABLE 5 - Absolute emissions - nbr outlier
# ----------------------------
st.write("**Absolute emissions growth rate - number of outliers**")

df_nmr_outliers_0 = pd.DataFrame(index=[selected_scope_0], columns=delta_years_str)

col1, col2, col3, col4 = st.columns([3, 2, 1, 3])

with col1:
    st.write("Number of equity with absolute emissions growth rate")

with col2:
    comparison_1 = st.radio(label="comparison_1", options=["superior", "inferior"], 
                            horizontal=True, key='comparison_1', label_visibility="collapsed")

with col3:
    st.write("to:")

with col4:
    threshold_1 = st.selectbox(
        label="threshold_1", options=[None, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
        format_func=lambda x: "Select a value" if x is None else f"{x:.1%}", 
        key='threshold_1', label_visibility="collapsed")

if threshold_1 is not None:
    if comparison_1 == 'superior':
        df_nmr_outliers_0 = (df_selection_1[delta_years_str] >= threshold_1).groupby(df_selection_1["scope"]).sum()
    elif comparison_1 == "inferior":
        df_nmr_outliers_0 = (df_selection_1[delta_years_str] <= threshold_1).groupby(df_selection_1["scope"]).sum()

st.dataframe(df_nmr_outliers_0, height=row_height * (len(df_nmr_outliers_0) + 1))


# ----------------------------
# TABLE 6 - Among abs emitter outlier
# ----------------------------

df_selection_1_threshold = (df_selection_all_col_1.set_index('isin')[delta_years_str]>= threshold_1)

df_selection_all_col_3 = df_eq_intensities_rate[(df_eq_intensities_rate['region_0'].isin(selected_region_0))&(
    df_eq_intensities_rate['high_impact_sector'].isin(selected_hisector_0))&(
        df_eq_intensities_rate['scope']==selected_scope_0)].copy()

df_selection_all_col_3['last_available_year'] = df_selection_all_col_3[delta_years_str[::-1]].bfill(axis=1).iloc[:, 0]

df_nmr_outliers_2 = df_nmr_outliers_0.copy()

col1, col2, col3, col4 = st.columns([3, 2, 1, 3])

with col1:
    st.write("Among these outliers, number of them with intensities growth rate")

with col2:
    comparison_2 = st.radio(label="comparison_2", options=["superior", "inferior"], 
                            horizontal=True, key='comparison_2', label_visibility="collapsed")

with col3:
    st.write("to:")

with col4:
    threshold_2 = st.selectbox(
        label="threshold_2", options=[None, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
        format_func=lambda x: "Select a value" if x is None else f"{x:.1%}", 
        key='threshold_2', label_visibility="collapsed")

if threshold_2 is not None:
    if comparison_2 == 'superior':
        df_selection_3_threshold = (df_selection_all_col_3.set_index('isin')[delta_years_str]>= threshold_2)
    elif comparison_2 == "inferior":
        df_selection_3_threshold = (df_selection_all_col_3.set_index('isin')[delta_years_str] <= threshold_2)
    if (threshold_1 is not None) and (comparison_1 is not None) and (comparison_2 is not None): 
        df_nmr_outliers_2 = (df_selection_1_threshold & df_selection_3_threshold).sum().rename(selected_scope_0)
        df_nmr_outliers_2 = df_nmr_outliers_2.to_frame().T

st.dataframe(df_nmr_outliers_2)


# ----------------------------
# TABLE 7 - Intensities - statistics
# ----------------------------

st.write("**Intensities - statistics**")

df_selection_all_col_2 = df_eq_intensities[(df_eq_intensities['region_0'].isin(selected_region_0))&(
    df_eq_intensities['high_impact_sector'].isin(selected_hisector_0)) & (
        df_eq_intensities['scope']==selected_scope_0)]

df_selection_2 = df_selection_all_col_2[['scope'] + years_str].copy()
df_selection_2['last_available_year'] = df_selection_2[years_str[::-1]].bfill(axis=1).iloc[:, 0]

selected_year_2 = st.multiselect(
    label='selected_year_2', label_visibility="collapsed", options=['last_available_year'] + years_str, 
    placeholder="Choose one or multiple  year(s)", key='year_2')

if not selected_year_2:
    selected_year_2 = years_str

df_selection_2_mean = pd.concat([df_selection_2['scope'], df_selection_2[selected_year_2].mean(axis=1).rename("mean_value")], axis=1)
df_stats_2 = df_selection_2_mean.groupby('scope')['mean_value'].describe()

df_fmt_2 = df_stats_2.copy()
df_fmt_2["count"] = df_stats_2["count"].astype(int)
df_fmt_2["std"] = df_stats_2["std"].map(lambda x: f"{x:.2e}")
cols_pct = df_fmt_2.columns.difference(["count", "std"])
df_fmt_2[cols_pct] = df_stats_2[cols_pct].map(lambda x: f"{x:.2e}")

st.dataframe(df_fmt_2, height=row_height * (len(df_fmt_2) + 1))


# ----------------------------
# TABLE 8 - Intensities - growth rate statistics
# ----------------------------

st.write("**Intensities growth rate - statistics**")

df_selection_3 = df_selection_all_col_3[['scope'] + delta_years_str + ['last_available_year']].copy()

selected_year_3 = st.multiselect(
    label='selected_year_3', label_visibility="collapsed", options=['last_available_year'] + delta_years_str, 
    placeholder="Choose one or multiple years range(s)", key='year_3'
)
if not selected_year_3:
    selected_year_3 = delta_years_str

df_selection_3_mean = pd.concat([df_selection_3['scope'], df_selection_3[selected_year_3].mean(axis=1).rename("mean_value")], axis=1)
df_stats_3 = df_selection_3_mean.groupby('scope')['mean_value'].describe()

df_fmt_3 = df_stats_3.copy()
df_fmt_3["count"] = df_stats_3["count"].astype(int)
df_fmt_3["std"] = df_stats_3["std"].map(lambda x: f"{x:.2e}")
cols_pct = df_fmt_3.columns.difference(["count", "std"])
df_fmt_3[cols_pct] = df_stats_3[cols_pct].map(lambda x: f"{x:.1%}")

st.dataframe(df_fmt_3, height=row_height * (len(df_fmt_3) + 1))


# ----------------------------
# GRAPH 1 - Intensities / Absolute growth rate
# ----------------------------

st.write("**Intensities / Absolute emissions**")

selected_year_4 = st.multiselect(
    label="selected_year_4", options=["last_available_year"] + delta_years_str,
    placeholder="Choose a year range", key="year_4", label_visibility="collapsed")

if not selected_year_4:
    selected_year_4 = delta_years_str

df_selection_all_col_1 = df_selection_all_col_1.copy()
df_selection_all_col_3 = df_selection_all_col_3.copy()
df_selection_all_col_1['mean_selected_year'] = df_selection_all_col_1[selected_year_4].mean(axis=1)
df_selection_all_col_3['mean_selected_year'] = df_selection_all_col_3[selected_year_4].mean(axis=1)

df_selection_all_col_4 = pd.merge(df_selection_all_col_1[['isin', 'mean_selected_year']].rename(columns={'mean_selected_year': 'abs'}),
    df_selection_all_col_3[['isin', 'mean_selected_year']].rename(columns={'mean_selected_year': 'int'}), on='isin')

fig = plt.figure(figsize=(10, 10)) 
plt.scatter(df_selection_all_col_4["abs"]*100, df_selection_all_col_4["int"]*100)

plt.xlim(-100, 200)
plt.ylim(-100, 200)

plt.xlabel("Absolute emissions growth rate (%)")
plt.ylabel("Intensities growth rate (%)")

if threshold_1 is not None:
    plt.axvline(x=threshold_1*100, linestyle='--', color='r')
if threshold_2 is not None:
    plt.axhline(y=threshold_2*100, linestyle='--', color='grey')

st.pyplot(fig)

st.write("Note: This graph shows the average growth rates for the selected years. For visual clarity, the axes are truncated at 200%, although certain observations lie above this threshold.")


# ----------------------------
# TABLE 9 - Ratio - delta abs emissions
# ----------------------------

st.write("**Absolute emissions - growth rate - scopes ratio - last year available**")

df_selection_ratio = df_eq_abs_em[(df_eq_abs_em['region_0'].isin(selected_region_0))&(
    df_eq_abs_em['high_impact_sector'].isin(selected_hisector_0))]
df_selection_ratio = df_selection_ratio.copy()
df_selection_ratio[years_str] = df_selection_ratio[years_str].replace(0, np.nan) 

selected_year_5 = st.multiselect(
    label='selected_year_5', label_visibility="collapsed", options=['last_available_year'] + years_str, 
    placeholder="Choose one or multiple  year(s)", key='year_5')

if not selected_year_5:
    selected_year_5 = years_str

df_selection_ratio['mean_selected_year'] = df_selection_ratio[selected_year_5].mean(axis=1)

list_stats_ratios = []
list_ratios = [("s2","s1"), ("s3_u","s1"), ("s3_d","s1")]

for scope_nu, scope_de in list_ratios: 
    df_ratio = (
        df_selection_ratio[["isin", "scope", "mean_selected_year"]]
        .groupby("isin")
        .apply(lambda g: 
                g.loc[g["scope"] == scope_nu, 'mean_selected_year' ].iloc[0]
                / g.loc[g["scope"] == scope_de, 'mean_selected_year' ].iloc[0],
                include_groups=False)
                .rename("mean_selected_year")
    )

    df_ratio_stats = df_ratio.describe().rename(f'{scope_nu}/{scope_de}')
    list_stats_ratios.append(df_ratio_stats)

df_ratio_final = pd.concat(list_stats_ratios, axis=1)  

df_ratio_final = df_ratio_final.map(lambda x: f"{x:.1f}")

st.dataframe(df_ratio_final, height=row_height * (len(df_ratio_final) + 1))


