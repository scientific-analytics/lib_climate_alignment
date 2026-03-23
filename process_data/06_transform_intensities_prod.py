import pandas as pd
import numpy as np


def clean_df_for_auto_intensity_em_by_km(
        df_raw, sector_moodys='Automobiles', 
        intensity_unit='Emission Intensity by Product (passenger vehicles) in gCO2/vkm', 
        years_str=[str(y) for y in range(2019,2024)]):

    df_auto_intensity_clean = df_raw[df_raw['Generic Sector']==sector_moodys].query(f"Unit=='{intensity_unit}'")[
        ['ISIN', 'Data Type', 'Scope'] + years_str].rename(columns={'ISIN': 'isin'})
    # ne garder que les data type qui sont historical (chaque equity qui a une info a au moins celle des historical data et elle semble plus fiable que l'extrapolation)
    df_auto_intensity_clean = df_auto_intensity_clean[df_auto_intensity_clean['Data Type']=='Historical Data']
    df_auto_intensity_clean = df_auto_intensity_clean.drop(['Data Type'], axis=1)
    # je garde le scope meme s'il n'y a que scpope 3
    df_auto_intensity_clean['Scope'] = df_auto_intensity_clean['Scope'].replace('Scope 3', 's3')
    df_auto_intensity_clean = df_auto_intensity_clean.rename(columns = {"Scope": 'scope'})
    # ajouter le scope 3d
    df_auto_intensity_clean = pd.concat([df_auto_intensity_clean, df_auto_intensity_clean.assign(scope="s3_d")], ignore_index=True)
    # je remplace les 'NI" par des NaN
    df_auto_intensity_clean = df_auto_intensity_clean.replace("NI", np.nan)
    # supprimer les lignes dont toutes les data sont nan
    df_auto_intensity_clean = df_auto_intensity_clean.set_index('isin').dropna(how='all').reset_index()

    return df_auto_intensity_clean


def create_df_average_hist_growth_rate(df_hist_emissions, years_hist_emissions=list(range(2019,2024))):
    
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

    df_emissions_rate = df_emissions_rate.replace([np.inf, -np.inf], np.nan) # fais legerement varier le nombre de rate available
    df_emissions_rate = df_emissions_rate.drop(years_hist_emissions, axis=1)

    df_emissions_stats = df_emissions_rate.copy()
    delta_years_rate = [f'{year-1}-{year}' for year in years_hist_emissions[1:]]
    df_emissions_stats['average_rate'] = df_emissions_stats[delta_years_rate].mean(axis=1)
    df_emissions_stats['nbr_rate_available'] = df_emissions_stats[delta_years_rate].notna().sum(axis=1)
    df_emissions_stats = df_emissions_stats[['isin', 'scope', 'average_rate', 'nbr_rate_available']] 

    return df_emissions_stats , df_emissions_rate


# ----------------------------
# Create final dataframes
# ----------------------------
first_year_available = 2019
last_year_available = 2024 #(abs emissions)

df_intensities_prod = pd.read_excel("data/raw_data/moodys_raw/Mra_Rl_Temperature_Alignment_2050_2024_10.xlsx", sheet_name="Full")
df_auto_intensity_clean = clean_df_for_auto_intensity_em_by_km(df_intensities_prod, years_str=[str(y) for y in range(first_year_available,last_year_available+1)])
df_auto_intensity_stats , df_auto_intensity_rate = create_df_average_hist_growth_rate(
    df_auto_intensity_clean, list(range(first_year_available,last_year_available+1)))

# ----------------------------
# Downloads final dataframes
# ----------------------------

df_auto_intensity_rate.to_parquet(f"data/intermediate_data/df_auto_intensities_rate_{first_year_available}_{last_year_available}.parquet", index=False)