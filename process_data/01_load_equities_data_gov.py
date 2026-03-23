import pandas as pd
import boto3
# .venv\Scripts\Activate 

# ----------------------------
# Functions to load data
# ----------------------------

def get_universe(date_folder_aws):

    # recuperer les isin
    universe = pd.read_csv(f"s3://sp-data-governance/{date_folder_aws}/_universe/reference_data.csv")[['companyid', 'company', 'nace', 'company_free_float_market_cap']]
    print('companyid in universe', universe['companyid'].nunique())
    isin_cousins = pd.read_csv(f"s3://sp-data-governance/{date_folder_aws}/_universe/isin_cousins.csv")[['companyid', 'isin', 'primary_isin']]
    print('isin in isincousins', isin_cousins['isin'].nunique())
    universe = pd.merge(isin_cousins, universe, on='companyid', how='left')
    print('isin in universe', universe['isin'].nunique())

    # recuperer les regions
    data_region = pd.read_csv(f"s3://sp-data-governance/{date_folder_aws}/_universe/auc.csv")
    data_region = data_region[['companyid', 'region']]    
    df_eq_info_data_gov = pd.merge(universe, data_region, on='companyid', how='left')

    return df_eq_info_data_gov


def is_folder_in_folder_aws(name_folder_to_check, path_folder, bucket="sp-data-governance"):
    ### path_folder if the path after bucket

    if "s3://sp-data-governance" in path_folder:
        path_folder = path_folder[len("s3://sp-data-governance")+1:]
    if path_folder[-1:] == '/':
        path_folder = path_folder[:-1]

    s3 = boto3.client("s3")
    bucket = "sp-data-governance"
    prefix= f'{path_folder}/{name_folder_to_check}'
    resp = s3.list_objects_v2(
        Bucket=bucket,
        Prefix=prefix,
        MaxKeys=1
    )

    exist = "Contents" in resp
    return exist


def load_revenues():
    # Take the last available data in the last folders.

    file_name="clean.parquet"
    list_data = []
    years = [str(y) for y in range(last_year_available, first_year_available-1, -1)]
    
    i=0
    for year in years:
        i=0
        prefolder = list_prefolders[i]
        path_folder = f"s3://sp-data-governance/{prefolder}/ciq/fundamentals"
        while not is_folder_in_folder_aws(year, path_folder):
            i=i+1
            if i == len(list_prefolders):
                raise FileNotFoundError(f"Aucun prefolder ne contient l'année {year}")
            
            prefolder = list_prefolders[i]
            path_folder = f"s3://sp-data-governance/{prefolder}/ciq/fundamentals"
        
        data = (pd.read_parquet(f"{path_folder}/{year}/{file_name}")
                [['companyid','revenues']]
                .rename(columns={'revenues': f'revenue_{year}'}))
        list_data.append(data)

    df_revenues = pd.DataFrame(columns = ['companyid'])

    for data in list_data:
        df_revenues = pd.merge(df_revenues, data, on='companyid', how='outer') 

    return df_revenues


# ----------------------------
# Load data
# ----------------------------
first_year_available = 2019
last_year_available = 2024
last_folder_data_gov = "2026-03-20"
list_prefolders = ["2026-03-20", "2025-12-19", "2025-03-21", "2024-12-20"]

df_eq_info_data_gov = get_universe(last_folder_data_gov)
df_revenues = load_revenues()

# ----------------------------
# Download data
# ----------------------------
df_eq_info_data_gov.to_parquet(f"data/raw_data/df_eq_info_data_gov_{last_folder_data_gov}.parquet")
df_revenues.to_parquet(f'data/raw_data/df_revenues_data_gov_{first_year_available}_{last_year_available}.parquet')