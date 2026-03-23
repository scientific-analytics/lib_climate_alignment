import sys
print("PYTHON EXECUTABLE =", sys.executable)
import pandas as pd
import numpy as np

# ----------------------------
# Configuration
# ----------------------------
YEARS = list(range(1990, 2061)) 
SCOPES = ["s1", "s2", "s3", "s12", "s123"]

DEFAULT_COLMAP = {
    "s1":   {"base":"climate_target_scope_1_base_year",      "target":"climate_target_scope_1_target_year",
             "quantity":"climate_target_scope_1_target_quantity",       "spec":"climate_target_scope_1_spec"},
    "s2":   {"base":"climate_target_scope_2_base_year",      "target":"climate_target_scope_2_target_year",
             "quantity":"climate_target_scope_2_target_quantity",       "spec":"climate_target_scope_2_spec"},
    "s3":   {"base":"climate_target_scope_3_base_year",      "target":"climate_target_scope_3_target_year",
             "quantity":"climate_target_scope_3_target_quantity",       "spec":"climate_target_scope_3_spec"},
    "s12":  {"base":"climate_target_scope_12_base_year",   "target":"climate_target_scope_12_target_year",
             "quantity":"climate_target_scope_12_target_quantity",    "spec":"climate_target_scope_12_spec"},
    "s123": {"base":"climate_target_scope_123_base_year","target":"climate_target_scope_123_target_year",
             "quantity":"climate_target_scope_123_target_quantity", "spec":"climate_target_scope_123_spec"},
}


# ----------------------------
# Functions to build the factors for expected emissions
# ----------------------------

# Small parsing helpers
def parse_year(x):
    try:
        return int(float(str(x).replace(",", "").strip()))
    except Exception:
        return None

def parse_float(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None

def is_absolute_target(spec):
    return isinstance(spec, str) and ("absolute" in spec.lower())


# Series construction + normalization
def make_target_path_series(base_year, target_year, quantity, years):
    """
    Build a trajectory on 'years' with value 1.0 at base_year
    and 'quantity' at target_year, linearly interpolated,
    then forward- and back-filled.
    """
    idx = pd.Index(years, name="year")
    s = pd.Series(np.nan, index=idx, dtype=float)

    if base_year is None or target_year is None or quantity is None:
        return s

    # Ensure chronological order
    if target_year < base_year:
        base_year, target_year = target_year, base_year

    # Clamp to index range
    base_year = min(max(base_year, idx[0]), idx[-1])
    target_year = min(max(target_year, idx[0]), idx[-1])

    # Handling on cases where base_year == target_year
    if base_year == target_year:
        if quantity == 1:
            s.loc[base_year] = 1.0
        else:  
            if target_year + 1 in idx:  
                s.loc[target_year+1] = 1 - quantity
            elif base_year - 1 in idx:
                s.loc[base_year-1] = 1 - quantity
    
    else: 
        if base_year in idx: # "if" is not necessary because of raw 64
            s.loc[base_year] = 1.0
        if target_year in idx:
            s.loc[target_year] = 1 - quantity

    # Interpolate inside endpoints; fill outside
    s = s.interpolate(limit_area="inside").bfill().ffill()
    return s


def normalize_to_first_year(s): # seem not necessary as v0 would always be =1 after interpolation (if nonna)
    """
    Divide by the value at the first year in the index, if defined and non-zero.
    """
    if s.isna().all():
        return s
    first_year = s.index[0]
    v0 = s.loc[first_year]
    if pd.isna(v0) or v0 == 0:
        return s
    return s / v0


# Scope-level builder
def build_scope_series_for_row(row, mapping, years, abs_only=True, normalize_first_year=True):
    """
    Build a normalized series for one scope, given a row and column mapping.
    Respects 'abs_only' and normalizes on the first year if requested.
    """
    # If required, skip non-absolute targets
    if abs_only and not is_absolute_target(row.get(mapping["spec"])): #considering one eq can't have both (intensity ou absolute))
        return pd.Series(np.nan, index=pd.Index(years, name="year"), dtype=float)

    by = parse_year(row.get(mapping["base"]))
    ty = parse_year(row.get(mapping["target"]))
    q  = parse_float(row.get(mapping["quantity"]))

    s = make_target_path_series(by, ty, q, years)
    if normalize_first_year:
        s = normalize_to_first_year(s)
    return s


# Fallback rules across scopes
def apply_scope_fallbacks(series_by_scope):
    """
    Fill missing scopes using broader scopes in a pragmatic order.
    """
    def valid(s): return (s is not None) and (not s.isna().all())

    # s1, s2 <- s12 <- s123
    if not valid(series_by_scope.get("s1")):
        series_by_scope["s1"] = series_by_scope.get("s12") if valid(series_by_scope.get("s12")) else series_by_scope.get("s123")
    if not valid(series_by_scope.get("s2")):
        series_by_scope["s2"] = series_by_scope.get("s12") if valid(series_by_scope.get("s12")) else series_by_scope.get("s123")
    # s12 <- s123
    if not valid(series_by_scope.get("s12")):
        series_by_scope["s12"] = series_by_scope.get("s123")
    # s3 <- s123
    if not valid(series_by_scope.get("s3")):
        series_by_scope["s3"] = series_by_scope.get("s123")

    return series_by_scope


# ----------------------------
# Functions to create final dataframes (Main entry points)
# ----------------------------

# Final df with factors (Main entry point)
def build_company_scope_factors(
    df,
    colmap=DEFAULT_COLMAP,
    years=YEARS,
    abs_only=True,
    use_fallbacks=True,
    id_cols=("isin",), # we can add the name column later if needed
    rename_columns=None,
):
    """
    Returns a wide DataFrame with columns:
    ['isin','scope', '2020', '2021', ..., '2050'].
    Each row corresponds to a (isin, scope).
    """
    years_str = [str(y) for y in years]
    out_rows = []

    # Group by unique issuer (ISIN as key)
    for _, g in df.groupby(id_cols[0], as_index=False):
        row0 = g.iloc[0]
        isin = row0[id_cols[0]]

        # Build per-scope series
        scope_series = {}
        for sc in SCOPES:
            m = colmap.get(sc)
            # Skip if mapping incomplete or columns missing
            if (m is None) or any(k not in row0.index for k in m.values()):
                scope_series[sc] = pd.Series(np.nan, index=pd.Index(years, name="year"), dtype=float)
                continue
            scope_series[sc] = build_scope_series_for_row(
                row=row0, mapping=m, years=years, abs_only=abs_only, normalize_first_year=True
            )

        if use_fallbacks:
            scope_series = apply_scope_fallbacks(scope_series)

        # Collect rows (create row for each scope, use 1.0 if no data)
        for sc, s in scope_series.items():
            if s is None or s.isna().all():
                # Create series with 1.0 for all years when no target data
                s = pd.Series(1.0, index=pd.Index(years, name="year"), dtype=float)
            row_dict = {id_cols[0]: isin, "scope": sc}
            row_dict.update({str(y): s.loc[y] if y in s.index else np.nan for y in years}) #y always in years by construction
            out_rows.append(row_dict)

    # Assemble final wide frame; ensure column order
    if not out_rows:
        cols = [id_cols[0], "scope", *[str(y) for y in years]]
        return pd.DataFrame(columns=cols)

    wide = pd.DataFrame(out_rows)
    # order columns: ids, scope, years
    wide = wide[[id_cols[0], "scope", *years_str]]
    
    # Rename columns if specified
    if rename_columns:
        wide = wide.rename(columns=rename_columns)
    
    return wide


# Final df with average expected growth rate
def create_df_average_expected_rate(df_targets_emissions, years_projected_emissions=list(range(2023,2028))):
    
    first_year = years_projected_emissions[0]
    first_year_str = str(first_year)

    # prevent if the format of the name of the year columns is string
    if (first_year not in df_targets_emissions) & (first_year_str not in df_targets_emissions):
        print('the range of years is incorrect')

    elif (first_year not in df_targets_emissions) & (first_year_str in df_targets_emissions):

        years_projected_emissions_str = {str(year): year for year in years_projected_emissions}
        df_targets_emissions = df_targets_emissions.rename(columns=years_projected_emissions_str)
    
    # create df with growth rate    
    df_projected_emissions_rate = df_targets_emissions.copy()
    years_projected_emissions_rate = years_projected_emissions[1:]
    
    for year in years_projected_emissions[1:]:

        df_projected_emissions_rate[year] = df_targets_emissions[year] / df_targets_emissions[year-1] - 1

    df_projected_emissions_rate = df_projected_emissions_rate.replace([np.inf, -np.inf], np.nan)# lead to small variation of the number of eq covered
    df_projected_emissions_rate = df_projected_emissions_rate.drop(first_year, axis=1)

    # compute average growth expected rate
    df_projected_emissions_stats = df_projected_emissions_rate.copy()
    df_projected_emissions_stats['average_expected_rate'] = df_projected_emissions_stats[years_projected_emissions_rate].mean(axis=1)
    df_projected_emissions_stats['nbr_expected_rate_available'] = df_projected_emissions_stats[years_projected_emissions_rate].notna().sum(axis=1) 

    df_projected_emissions_stats = df_projected_emissions_stats[['isin', 'scope', 'average_expected_rate', 'nbr_expected_rate_available']] 
    
    return df_projected_emissions_stats #,df_projected_emissions_rate if needed


def add_columns_for_level_of_ambition(df, df_raw_targets, col_ambition_target='climate_target_ambition'):

    dict_mapping_targets_ctgry = {
        'is_ambitious_target' : ["Ambitious Target"],
        'is_approved_sbt_target' : ["Approved SBT"],
        'is_committed_sbt_target' : ["Committed SBT"],
        "is_target_sbt_or_ambitious": ["Ambitious Target", "Approved SBT"],
        "is_target_sbt_or_ambitious_or_commited": ["Ambitious Target", "Approved SBT", "Committed SBT"],
        "is_target_non_ambitious": ["Non-Ambitious Target"],
        "is_no_target": ["No Target"]}
    
    df_targets_ambition = df_raw_targets.copy()
    if col_ambition_target not in df_raw_targets.columns:
        print("the columns with the target level of ambition is missing")
        return df_targets_ambition
    
    for target_ambition in dict_mapping_targets_ctgry.keys():
        df_targets_ambition[target_ambition] = df_targets_ambition[col_ambition_target].apply(
            lambda x: np.nan if pd.isna(x) else x in dict_mapping_targets_ctgry[target_ambition])
    
    df_targets_ambition = df_targets_ambition[['isin'] + list(dict_mapping_targets_ctgry.keys())]
    # in case of duplicates
    df_targets_ambition = df_targets_ambition.drop_duplicates(subset='isin', keep="first")
    # merge with dataframe
    # df_merged_targets_ambition = pd.merge(df, df_targets_ambition, on='isin', how='left')
    print(df['isin'].nunique())
    print(df_targets_ambition['isin'].nunique())
    df_merged_targets_ambition = pd.merge(df, df_targets_ambition, on='isin', how='outer')
    print(df_targets_ambition['isin'].nunique())
    return df_merged_targets_ambition


# ----------------------------
# Create final dataframes
# ----------------------------
last_available_year = 2024
raw_targets = pd.read_parquet(f"data/raw_data/iss_raw/iss_targets_{last_available_year}.parquet")

# Check if all columns from DEFAULT_COLMAP exist in the raw file
required_columns = set()
for scope_mapping in DEFAULT_COLMAP.values():
    required_columns.update(scope_mapping.values())

missing_columns = required_columns - set(raw_targets.columns)
if missing_columns:
    print(f"Warning: Missing columns in raw data: {sorted(missing_columns)}")
else:
    print("All required columns found in raw data")

# creation of the dataframes
df_expected_abs_growth_factors = build_company_scope_factors(
    raw_targets, colmap=DEFAULT_COLMAP, abs_only=True,
    rename_columns={"ISIN": "isin"})

df_projected_emissions_stats = create_df_average_expected_rate(df_expected_abs_growth_factors)

df_projected_emissions_stats = add_columns_for_level_of_ambition(df_projected_emissions_stats, raw_targets)


# ----------------------------
# Downloads final dataframes
# ----------------------------

df_expected_abs_growth_factors.to_csv(f"data/intermediate_data/exp_abs_emissions_growth_factors_{last_available_year}.csv", index=False)
df_projected_emissions_stats.to_parquet(f"data/intermediate_data/exp_abs_emissions_growth_rate_stats_{last_available_year}.parquet", index=False)