import pandas as pd
import numpy as np

first_year_available, last_year_available = 2019, 2024
df_eq_all_infos = pd.read_parquet("data/intermediate_data/df_eq_all_infos.parquet")
df_exp_abs_emissions_growth_rate_stats = pd.read_parquet(f"data/intermediate_data/df_expected_emissions_rate_stats_{last_year_available}.parquet")
df_hist_abs_emissions_growth_rate_stats = pd.read_parquet(f"data/intermediate_data/hist_abs_emissions_growth_rate_stats_{first_year_available}_{last_year_available}.parquet")
df_hist_intensities_growth_rate_stats = pd.read_parquet(f"data/intermediate_data/hist_intensities_growth_rate_stats_{first_year_available}_{last_year_available}.parquet")

# ----------------------------
# Initialisation
# ----------------------------

# on garde le max d'isin fourni par ISS avant de merger avec notre univers d'equities
df_exp_abs_emissions_growth_rate_stats['isin'] = df_exp_abs_emissions_growth_rate_stats['isin'].astype(str).str.strip()
df_hist_abs_emissions_growth_rate_stats['isin'] = df_hist_abs_emissions_growth_rate_stats['isin'].astype(str).str.strip()
list_isin = list(set(df_exp_abs_emissions_growth_rate_stats['isin'].unique())) + list(set(df_hist_abs_emissions_growth_rate_stats['isin'].unique()))
print(len(list_isin))
df_eq_all_infos = df_eq_all_infos[df_eq_all_infos['isin'].isin(list_isin)]
print(len(df_eq_all_infos['isin']))
print(df_eq_all_infos['isin'].nunique())
df_eq_all_infos = df_eq_all_infos.drop_duplicates(subset='isin', keep='first').copy()

# ----------------------------
# Add weight derived from market cap
# ----------------------------
def create_mc_weight(df, col_region='region', col_sector='high_impact_sector'):

    df_2 = df.drop_duplicates(subset='isin', keep='first').copy()
    col_eq_mc_weight = 'company_free_float_market_cap'
    # for one region AND sector block
    df_2['r_s_mc_weight'] = df_2[col_eq_mc_weight] / \
        df_2.groupby([col_region, col_sector])[col_eq_mc_weight].transform('sum')    
    # for one region
    df_2['r_mc_weight'] = df_2[col_eq_mc_weight] / \
        df_2.groupby([col_region])[col_eq_mc_weight].transform('sum')
    # for one sector
    df_2['s_mc_weight'] = df_2[col_eq_mc_weight] / \
        df_2.groupby([col_sector])[col_eq_mc_weight].transform('sum')        
    df = pd.merge(df, df_2[['isin', 'r_s_mc_weight', 'r_mc_weight', 's_mc_weight']], on='isin', how='left')
    
    return df

df_eq_all_infos = create_mc_weight(df_eq_all_infos, col_region='region_0')
print(df_eq_all_infos["isin"].duplicated().sum())

# ----------------------------
# Duplicate rows for each scope
# ----------------------------
scopes = ['s1', 's2', 's3', 's3_u', 's3_d', 's12', 's123', 's123_u', 's123_d', 's13', 's23', 's13_u', 's13_d', 's23_u','s23_d' ]
df_eq_info_all_scopes = df_eq_all_infos.merge(
    pd.DataFrame({"scope": scopes}), how="cross")


# ----------------------------
# Merge datasets
# ----------------------------
df_merged = pd.merge(df_exp_abs_emissions_growth_rate_stats, df_hist_abs_emissions_growth_rate_stats, on=['isin', 'scope'], how='outer')
df_merged = pd.merge(df_merged, df_hist_intensities_growth_rate_stats.rename(
    columns={'average_rate': 'intensity_average_rate', 'nbr_rate_available':'intensity_nbr_rate_available'}), on=['isin', 'scope'], how='outer')
df_merged = pd.merge(df_merged, df_eq_info_all_scopes, on=['isin', 'scope'], how='right')
df_merged['is_no_target'] = df_merged['is_no_target'].fillna(True)

# ----------------------------
# Last adjustments
# ----------------------------

# Add boolean column for relevant scope
df_merged['is_relevant_scopes'] = df_merged.apply(
    lambda x: np.nan if pd.isna(x['relevant_scopes']) else (
        True if x['scope'] == x['relevant_scopes'] else False
    ), axis=1)
df_merged = df_merged.drop('relevant_scopes', axis=1)

# Delete rows without isin
df_merged = df_merged[df_merged['isin'].notna()]

# ----------------------------
# Download
# ----------------------------
df_merged.to_parquet(f'data/intermediate_data/df_merged_all_infos_{first_year_available}_{last_year_available}.parquet')