import pandas as pd
import numpy as np

path_file_mappings = 'data/climate_alignment_index_parameters.xlsx'
sheet_mapping_code_nace_sectors, sheet_mapping_sector_rel_scopes = \
    'mapping_nace_nzif_sectors', 'mapping_sector_rel_scopes'


# ----------------------------
# Functions for mapping NACE code with high impact sectors
# ----------------------------

def set_good_format_nace(code_nace):
    """
    Prepare nace code format
    """
    if pd.isnull(code_nace):
        return np.nan
    prefix = code_nace.split('.')[0]
    if (len(prefix) == 1) & (prefix.isdigit()):
        code_nace = '0' + code_nace
    code_nace = '"' + code_nace + '"'
    return code_nace

def prepare_mapping_table(df_mapping, col_code_nace='nace_rev_21_code', col_sector='nzif_nace_21'):
    if (col_code_nace not in df_mapping) or (col_sector not in df_mapping) :
        return print('columns for mapping nace code with sectors are missing')
    
    df_mapping = df_mapping[[col_code_nace, col_sector]]
    df_mapping = df_mapping.drop_duplicates()
    if len(df_mapping) != df_mapping[col_code_nace].nunique():
        return print('nace codea have different nzif values, plz fix the table', 
                     df_mapping[df_mapping.duplicated(col_code_nace, keep=False)])

    df_mapping = df_mapping.rename(
        columns={col_code_nace: 'nace', col_sector: 'high_impact_sector'})
    
    return df_mapping

def add_high_impact_sector(df_eq, df_mapping, col_nace_df='nace', col_code_nace_mapping='nace_rev_21_code', col_sector_mapping='nzif_nace_21'):

    if col_nace_df not in df_eq:
        return 'nace column missing from the dataframe with the equity information'
    df_eq = df_eq.rename(columns={col_nace_df: 'nace'})
    df_eq["nace"] = df_eq["nace"].astype(str)
    df_eq['nace'] = df_eq['nace'].apply(set_good_format_nace)    

    df_mapping = prepare_mapping_table(df_mapping, col_code_nace_mapping, col_sector_mapping)

    df_eq_with_high_impact_sector = pd.merge(df_eq, df_mapping, on='nace', how='left')

    return df_eq_with_high_impact_sector

def mapping_test(df_mapping, col_code_nace='nace_rev_21_code'):
    """
    for each two-digit number, there should exist at least one value of the form “xx.x”, and for each “xx.x”, at least one value of the form “xx.xx”.
    """
    list_int = list(df_mapping[df_mapping[col_code_nace].str.len()==4][col_code_nace].unique())
    for nb_int in list_int:
        list_one_decimal = list(df_mapping[(df_mapping[col_code_nace].str.len().eq(6)) & (
            df_mapping[col_code_nace].str.split('.').str[0].eq(nb_int[:3]))][col_code_nace].unique())
        if len(list_one_decimal)==0:
            print('there are missing nace-codes with one decimal for code', nb_int)
            continue
        for nb_one_dec in list_one_decimal:
            list_two_decimal = list(df_mapping[(df_mapping[col_code_nace].str.len().eq(7)) & (
                df_mapping[col_code_nace].str[:5].eq(nb_one_dec[:5]))][col_code_nace].unique())
            if len(list_two_decimal)==0:
                print('there are missing nace-codes with two decimal for code', nb_one_dec)


# ----------------------------
# Functions to add additional information
# ----------------------------

def add_sectors_relevant_scopes_column(df):

    df_mapping_sector_scopes = pd.read_excel(path_file_mappings, sheet_name=sheet_mapping_sector_rel_scopes)
    mapping_sector_scopes = df_mapping_sector_scopes.set_index('high_impact_sector').to_dict()['scopes']

    if 'high_impact_sector' not in df.columns:
        print("'high_impact_sector' if missing from the dataframe to add relevant scopes")
        return df
    df = df.copy()
    df['relevant_scopes'] = df['high_impact_sector'].map(mapping_sector_scopes)

    return df

def add_regions(df):
    if 'region' not in df.columns:
        print('"region" not in columns')
        return df
    df = df.rename(columns={'region': 'region_0'})
    dict_regions = {'Dev Europe Other': 'Developed Europe', 
                    'Dev Europe EU': 'Developed Europe', 
                    'Dev Europe GB': 'Developed Europe',
                    'Dev America US': 'North America', 
                    'Dev America CA': 'North America', 
                    'Dev Asia-Pac Other': 'Other Developed',
                    'Dev Asia-Pac JP': 'Other Developed',
                    'Emg Europe': 'Emergings',
                    'Emg Asia-Pac IN': 'Emergings',
                    'Emg Asia-Pac CN': 'Emergings', 
                    'Emg Asia-Pac Other': 'Emergings', 
                    'Emg America': 'Emergings'}
    df = df.copy()
    df['region'] = df['region_0'].map(dict_regions)
    # df = df.drop('region_0', axis=1)

    return df


# ----------------------------
# Add info in equity information dataframe
# ----------------------------
last_folder_data_gov = "2026-03-20"
df_mapping = pd.read_excel(path_file_mappings, sheet_mapping_code_nace_sectors)[['nace_rev_21_code', 'nzif_nace_21']]
df_eq_info_data_gov = pd.read_parquet(f"data/raw_data/df_eq_info_data_gov_{last_folder_data_gov}.parquet")
mapping_test(df_mapping)
df_eq_all_infos = add_high_impact_sector(df_eq_info_data_gov, df_mapping)
df_eq_all_infos = add_sectors_relevant_scopes_column(df_eq_all_infos)
df_eq_all_infos = add_regions(df_eq_all_infos)


# ----------------------------
# Download dataframe
# ----------------------------

df_eq_all_infos.to_parquet("data/intermediate_data/df_eq_all_infos.parquet")