import sys

print("PYTHON EXECUTABLE =", sys.executable)
import pandas as pd
import numpy as np


# ----------------------------
# Functions
# ----------------------------


def create_df_with_real_emissions_on_expected_factors(
    df_hist_abs_emissions,
    df_expected_abs_growth_factors,
    years_str_hist_abs_em=None,
    years_str_exp_grwth=None,
):
    """
    creer le dataframe avec les dernieres emissions reelles observees multipliees par les facteurs expected
    """
    if not years_str_hist_abs_em:
        years_str_hist_abs_em = sorted(
            [c for c in df_hist_abs_emissions.columns if c.isdigit()],
            key=int,
            reverse=False,
        )

    if not years_str_exp_grwth:
        years_str_exp_grwth = [
            c for c in df_expected_abs_growth_factors.columns if c.isdigit()
        ]

    df_hist_abs_emissions["last_available_year_value"] = (
        df_hist_abs_emissions[years_str_hist_abs_em].bfill(axis=1).iloc[:, 0]
    )

    df_hist_abs_emissions["last_available_year"] = (
        df_hist_abs_emissions[years_str_hist_abs_em].notna().idxmax(axis=1)
    )

    df_expected_abs = df_expected_abs_growth_factors.merge(
        df_hist_abs_emissions[
            ["isin", "scope", "last_available_year", "last_available_year_value"]
        ],
        on=["isin", "scope"],
        how="left",
    )

    mask = df_expected_abs["last_available_year"].notna()

    # serie avec les facteurs a la last_available_date pour chaque ligne
    growth_at_last = pd.Series(np.nan, index=df_expected_abs.index)
    growth_at_last.loc[mask] = df_expected_abs.loc[mask].apply(
        lambda row: row[row["last_available_year"]], axis=1
    )
    growth_at_last.loc[~mask] = np.nan

    # la c'est donc la valeur a laquelle il faut multiplier chaque ligne
    scale = df_expected_abs["last_available_year_value"] / growth_at_last
    df_expected_abs[years_str_exp_grwth] = df_expected_abs[years_str_exp_grwth].mul(
        scale, axis=0
    )
    df_expected_abs = df_expected_abs.drop(
        ["last_available_year", "last_available_year_value"], axis=1
    )

    return df_expected_abs


def add_s3_u_and_s3_d(df_expected_abs):
    """
    rajouter les scopes s3_u et s3_d,  on s'en fiche un peu des valeurs, c'est plus pour
    la trajectoires donc on peut dire que s3_u et s3_d = s3
    """

    s3_u = df_expected_abs[df_expected_abs["scope"] == "s3"].copy()
    s3_d = df_expected_abs[df_expected_abs["scope"] == "s3"].copy()
    s3_u["scope"] = "s3_u"
    s3_d["scope"] = "s3_d"
    df_expected_abs = pd.concat([df_expected_abs, s3_u], ignore_index=True)
    df_expected_abs = pd.concat([df_expected_abs, s3_d], ignore_index=True)

    return df_expected_abs


def add_other_scope_combination(df_expected_abs, years_str_exp_grwth=None):
    """creer les autres cominaisons de scope (qui sont la sommes des emissions des scopes de la combinaision)"""

    if not years_str_exp_grwth:
        years_str_exp_grwth = [
            c for c in df_expected_abs_growth_factors.columns if c.isdigit()
        ]

    scope_map = {
        "s13": ["s1", "s3"],
        "s23": ["s2", "s3"],
        "s13_u": ["s1", "s3_u"],
        "s13_d": ["s1", "s3_d"],
        "s23_u": ["s2", "s3_u"],
        "s23_d": ["s2", "s3_d"],
        "s123_u": ["s1", "s2", "s3_u"],
        "s123_d": ["s1", "s2", "s3_d"],
    }

    rows = []
    for new_scope, scopes in scope_map.items():
        tmp = (
            df_expected_abs[df_expected_abs["scope"].isin(scopes)]
            .groupby("isin")[years_str_exp_grwth]
            .sum(min_count=1)
            .reset_index()
        )
        tmp["scope"] = new_scope
        rows.append(tmp)

    df_expected_abs = pd.concat([df_expected_abs] + rows)
    df_expected_abs = df_expected_abs.sort_values(["isin", "scope"], ignore_index=True)

    return df_expected_abs


def create_df_average_hist_growth_rate(
    df_hist_emissions, years_hist_emissions=list(range(2019, 2024))
):

    first_year = years_hist_emissions[0]
    first_year_str = str(first_year)

    if (first_year not in df_hist_emissions) & (
        first_year_str not in df_hist_emissions
    ):
        print("the range of years is incorrect")

    elif (first_year not in df_hist_emissions) & (first_year_str in df_hist_emissions):

        years_hist_emissions_str = {str(year): year for year in years_hist_emissions}
        df_hist_emissions = df_hist_emissions.rename(columns=years_hist_emissions_str)

    df_emissions_rate = df_hist_emissions.copy()
    for year in years_hist_emissions[1:]:

        df_emissions_rate[f"{year-1}-{year}"] = (
            df_hist_emissions[year] / df_hist_emissions[year - 1] - 1
        )

    df_emissions_rate = df_emissions_rate.replace(
        [np.inf, -np.inf], np.nan
    )  # fais legerement varier le nombre de rate available
    df_emissions_rate = df_emissions_rate.drop(years_hist_emissions, axis=1)

    df_emissions_stats = df_emissions_rate.copy()
    delta_years_rate = [f"{year-1}-{year}" for year in years_hist_emissions[1:]]
    df_emissions_stats["average_expected_rate"] = df_emissions_stats[
        delta_years_rate
    ].mean(axis=1)
    df_emissions_stats["nbr_expected_rate_available"] = (
        df_emissions_stats[delta_years_rate].notna().sum(axis=1)
    )
    df_emissions_stats = df_emissions_stats[
        ["isin", "scope", "average_expected_rate", "nbr_expected_rate_available"]
    ]

    return df_emissions_stats, df_emissions_rate


def add_columns_for_level_of_ambition(
    df, df_raw_targets, col_ambition_target="climate_target_ambition"
):

    dict_mapping_targets_ctgry = {
        "is_ambitious_target": ["Ambitious Target"],
        "is_approved_sbt_target": ["Approved SBT"],
        "is_committed_sbt_target": ["Committed SBT"],
        "is_target_sbt_or_ambitious": ["Ambitious Target", "Approved SBT"],
        "is_target_sbt_or_ambitious_or_commited": [
            "Ambitious Target",
            "Approved SBT",
            "Committed SBT",
        ],
        "is_target_non_ambitious": ["Non-Ambitious Target"],
        "is_no_target": ["No Target"],
    }

    df_targets_ambition = df_raw_targets.copy()
    if col_ambition_target not in df_raw_targets.columns:
        print("the columns with the target level of ambition is missing")
        return df_targets_ambition

    for target_ambition in dict_mapping_targets_ctgry.keys():
        df_targets_ambition[target_ambition] = df_targets_ambition[
            col_ambition_target
        ].apply(
            lambda x: (
                np.nan
                if pd.isna(x)
                else x in dict_mapping_targets_ctgry[target_ambition]
            )
        )

    df_targets_ambition = df_targets_ambition[
        ["isin"] + list(dict_mapping_targets_ctgry.keys())
    ]
    # in case of duplicates
    df_targets_ambition = df_targets_ambition.drop_duplicates(
        subset="isin", keep="first"
    )
    # merge with dataframe
    # df_merged_targets_ambition = pd.merge(df, df_targets_ambition, on='isin', how='left')
    print(df["isin"].nunique())
    print(df_targets_ambition["isin"].nunique())
    df_merged_targets_ambition = pd.merge(
        df, df_targets_ambition, on="isin", how="outer"
    )
    print(df_targets_ambition["isin"].nunique())
    return df_merged_targets_ambition


# ----------------------------
# Create dataframes
# ----------------------------

first_year_available = 2019
last_year_available = 2024
df_expected_abs_growth_factors = pd.read_csv(
    f"data/intermediate_data/exp_abs_emissions_growth_factors_{last_year_available}.csv"
)
df_hist_abs_emissions = pd.read_parquet(
    f"data/intermediate_data/hist_abs_emissions_{first_year_available}_{last_year_available}.parquet"
)
raw_targets = pd.read_parquet(
    f"data/raw_data/iss_raw/iss_targets_{last_year_available}.parquet"
)

df_expected_abs = create_df_with_real_emissions_on_expected_factors(
    df_hist_abs_emissions, df_expected_abs_growth_factors
)
df_expected_abs = add_s3_u_and_s3_d(df_expected_abs)
df_expected_abs = add_other_scope_combination(df_expected_abs)

years_str_exp_grwth = [c for c in df_expected_abs_growth_factors.columns if c.isdigit()]
df_expected_emissions_rate_stats, df_expected_emissions_rate = (
    create_df_average_hist_growth_rate(
        df_expected_abs, [int(y) for y in years_str_exp_grwth]
    )
)
df_expected_emissions_rate_stats = add_columns_for_level_of_ambition(
    df_expected_emissions_rate_stats, raw_targets
)

df_expected_abs.to_parquet(
    f"data/intermediate_data/df_expected_emissions_{last_year_available}.parquet"
)
df_expected_emissions_rate.to_parquet(
    f"data/intermediate_data/df_expected_emissions_rate_{last_year_available}.parquet"
)
df_expected_emissions_rate_stats.to_parquet(
    f"data/intermediate_data/df_expected_emissions_rate_stats_{last_year_available}.parquet"
)
