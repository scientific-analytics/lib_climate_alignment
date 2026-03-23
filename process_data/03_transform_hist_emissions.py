import pandas as pd
import numpy as np

# ----------------------------
# Functions to create df for historical emissions
# ----------------------------

def load_files_hist_emissions():

    emissions_col_map = {"ghg_emissions_scope_1": "s1", 
                        "ghg_emissions_scope_2": "s2", 
                        "ghg_emissions_scope_3": "s3",
                        "ghg_emissions_scope_3_upstream": "s3_u",
                        "ghg_emissions_scope_3_downstream": "s3_d",}

    columns_to_keep = ['isin', 'name', 'fiscal_year', 'ghg_emissions_scope_1', 'ghg_emissions_scope_2', 
            'ghg_emissions_scope_3', 'ghg_emissions_scope_3_upstream', 'ghg_emissions_scope_3_downstream']

    years = list(range(2019, last_year_available+1))

    dict_df_emissions = {}
    for year in years:

        try:
            df = pd.read_parquet(f"data/raw_data/iss_raw/iss_emissions_{year}.parquet")
        except FileNotFoundError:
            print(f"File for year {year} not found, skipping.")
            continue

        dict_df_emissions[year] = df[columns_to_keep].dropna(how='all').rename(columns=emissions_col_map)

    return dict_df_emissions


def create_s12_and_s123(df_year):
    if df_year is None:
        return None

    df_year = df_year.copy()
    df_year['s12'] = df_year[['s1', 's2']].sum(axis=1, skipna=False) # put nan if any of the two is nan
    df_year['s123'] = df_year[['s1', 's2', 's3']].sum(axis=1, skipna=False)

    return df_year    


def create_other_scp_combination(df_year):
    if df_year is None:
        return None

    df_year = df_year.copy()
    df_year['s13'] = df_year[['s1', 's3']].sum(axis=1, skipna=False) # put nan if any of the two is nan
    df_year['s23'] = df_year[['s2', 's3']].sum(axis=1, skipna=False)
    df_year['s13_u'] = df_year[['s1', 's3_u']].sum(axis=1, skipna=False)
    df_year['s13_d'] = df_year[['s1', 's3_d']].sum(axis=1, skipna=False)
    df_year['s23_u'] = df_year[['s2', 's3_u']].sum(axis=1, skipna=False)
    df_year['s23_d'] = df_year[['s2', 's3_d']].sum(axis=1, skipna=False)
    df_year['s123_u'] = df_year[['s1', 's2', 's3_u']].sum(axis=1, skipna=False)
    df_year['s123_d'] = df_year[['s1', 's2', 's3_d']].sum(axis=1, skipna=False)

    return df_year


def create_table_one_year(df_year, fiscal_year):
    """ return the good formatted dataframe for one year """

    if df_year is None:
        print(f"No data available for fiscal year {fiscal_year}.")
        return pd.DataFrame()  # Return empty DataFrame if year not found

    scope_order = ['s1', 's2', 's3', 's3_u', 's3_d', 's12', 's123', 's123_u', 's123_d', 's13', 's23', 's13_u', 's13_d', 's23_u','s23_d' ]

    df_year = df_year.copy()
    df_year = df_year[df_year['fiscal_year'] == fiscal_year].drop(columns=['fiscal_year'])
    df_year = (df_year.melt(
        id_vars=["isin", "name"], 
        var_name="scope", 
        value_name=fiscal_year
        )
        .assign(scope=lambda d: pd.Categorical(d["scope"], categories=scope_order, ordered=True))
        .sort_values(by=["isin", "scope"])  
    )

    return df_year


def create_all_years_emissions_table(dict_df_emissions):

    fiscal_years = list(dict_df_emissions.keys())
    if not fiscal_years:
        print("No emissions data available.")
        return pd.DataFrame()  # Return empty DataFrame if no data

    df_merged = pd.DataFrame()

    for year in fiscal_years:
        df_year = dict_df_emissions.get(year)
        df_year = create_s12_and_s123(df_year)
        df_year = create_other_scp_combination(df_year)
        df_year = create_table_one_year(df_year, year)
        if df_merged.empty:
            df_merged = df_year
        else:
            df_merged = pd.merge(
                df_merged, 
                df_year.drop('name', axis=1), 
                on=['isin', 'scope'], 
                how='outer'
            )

    return df_merged


def create_df_average_hist_growth_rate(df_hist_emissions, years_hist_emissions=list(range(2019,2025))):
    
    first_year = years_hist_emissions[0]
    first_year_str = str(first_year)

    if (first_year not in df_hist_emissions) & (first_year_str not in df_hist_emissions):
        print('the range of years is incorrect')

    elif (first_year not in df_hist_emissions) & (first_year_str in df_hist_emissions):

        years_hist_emissions_str = {str(year): year for year in years_hist_emissions}
        df_hist_emissions = df_hist_emissions.rename(columns=years_hist_emissions_str)
    
    df_emissions_rate = df_hist_emissions.copy()
    for year in years_hist_emissions[1:]:

        df_emissions_rate[f'{year-1}-{year}'] = df_hist_emissions[year] / df_hist_emissions[year-1] - 1

    df_emissions_rate = df_emissions_rate.replace([np.inf, -np.inf], np.nan)#fais legerement varier le nombre de rate available
    df_emissions_rate = df_emissions_rate.drop(years_hist_emissions, axis=1)

    df_emissions_stats = df_emissions_rate.copy()
    delta_years_rate = [f'{year-1}-{year}' for year in years_hist_emissions[1:]]
    df_emissions_stats['average_rate'] = df_emissions_stats[delta_years_rate].mean(axis=1)
    df_emissions_stats['nbr_rate_available'] = df_emissions_stats[delta_years_rate].notna().sum(axis=1)
    df_emissions_stats = df_emissions_stats[['isin', 'scope', 'average_rate', 'nbr_rate_available']] 

    return df_emissions_stats , df_emissions_rate # if needed

# ----------------------------
# Functions to create df for intensities
# ----------------------------

def create_merged_df_emissions_and_revenues(df_revenues, df_data_gov):

    df_merge = pd.merge(df_data_gov[['companyid', 'isin']], df_revenues, on='companyid', how='left')
    df_merge = df_merge.drop_duplicates(subset='isin', keep='first').copy()
    df_merge_em_rev = pd.merge(df_merge, df_hist_emissions, on="isin", how='right')

    return df_merge_em_rev


def compute_intensities(df_merge_em_rev, years=list(range(2019,2025))):
    # set good format for year columns in historical emissions
    first_year = years[0]
    first_year_str = str(first_year)

    # prevent if the format of the name of the year columns is string
    if (first_year not in df_merge_em_rev) & (first_year_str not in df_merge_em_rev):
        print('the range of years is incorrect')

    elif (first_year not in df_merge_em_rev) & (first_year_str in df_merge_em_rev):
        dict_years_str_int = {str(year): year for year in years}
        df_merge_em_rev = df_merge_em_rev.rename(columns=dict_years_str_int)

    # compute intensities    
    df_intensities = df_merge_em_rev.copy()
    for year in years:
        df_intensities[year] = df_intensities.apply(
            lambda x: np.nan if (x[f'revenue_{year}'] <= 0) else (
                x[year] / x[f'revenue_{year}'])
            , axis=1)
        
    return df_intensities


# ----------------------------
# Create dataframes
# ----------------------------
first_year_available = 2019
last_year_available = 2024 #(abs emissions)
last_folder_data_gov = "2026-03-20"
# Absolute emissions
dict_df_emissions = load_files_hist_emissions()
df_hist_emissions = create_all_years_emissions_table(dict_df_emissions)
df_emissions_growth_rate_stats, df_emissions_growth_rate = create_df_average_hist_growth_rate(
    df_hist_emissions, list(range(first_year_available,last_year_available+1)))

# Intensities
df_revenues = pd.read_parquet(f'data/raw_data/df_revenues_data_gov_{first_year_available}_{last_year_available}.parquet')
df_data_gov = pd.read_parquet(f'data/raw_data/df_eq_info_data_gov_{last_folder_data_gov}.parquet') 
df_merge_em_rev = create_merged_df_emissions_and_revenues(df_revenues, df_data_gov)
df_hist_intensities = compute_intensities(df_merge_em_rev, list(range(2019,2025)))
df_intensities_growth_rate_stats, df_intensities_growth_rate = create_df_average_hist_growth_rate(
    df_hist_intensities, list(range(first_year_available,last_year_available+1)))

# ----------------------------
# Downloads dataframes
# ----------------------------
df_hist_emissions.to_parquet(f'data/intermediate_data/hist_abs_emissions_{first_year_available}_{last_year_available}.parquet')
df_hist_intensities.to_parquet(f'data/intermediate_data/hist_intensities_{first_year_available}_{last_year_available}.parquet')

df_emissions_growth_rate.to_parquet(f'data/intermediate_data/hist_abs_emissions_growth_rate_{first_year_available}_{last_year_available}.parquet')
df_intensities_growth_rate.to_parquet(f'data/intermediate_data/hist_intensities_growth_rate_{first_year_available}_{last_year_available}.parquet')

df_emissions_growth_rate_stats.to_parquet(f'data/intermediate_data/hist_abs_emissions_growth_rate_stats_{first_year_available}_{last_year_available}.parquet')
df_intensities_growth_rate_stats.to_parquet(f'data/intermediate_data/hist_intensities_growth_rate_stats_{first_year_available}_{last_year_available}.parquet')
