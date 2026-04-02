import pandas as pd
import numpy as np

# ----------------------------
# Functions
# ----------------------------


def adj_1_keep_only_the_3_last_years(
    df_emissions_rate, delta_years_str=[f"{int(y)}-{y+1}" for y in range(2019, 2024)]
):

    df_adjusted = df_emissions_rate.copy()
    df_adjusted[delta_years_str[:-3]] = np.nan

    return df_adjusted


def adj_2_unique_outlier_or_not_if_nbr_obs_inf_3(
    df_emissions_rate,
    threshold,
    delta_years_str=[f"{int(y)}-{y+1}" for y in range(2019, 2024)],
):
    """
    don t consider one extreme observation (<-20% or >20%)  if the others are enough in number (>=3) or non-extreme. (replace with np.nan)
    """
    if threshold is None:
        return df_emissions_rate

    df_emissions_rate["can_remove_outliers"] = df_emissions_rate.apply(
        lambda x: (
            True
            if (
                (x["nbr_observations"] >= 3 and (x["number_all_outliers"] < 2))
                or (x["nbr_observations"] in [1, 2] and (x["number_all_outliers"] == 0))
            )
            else False
        ),
        axis=1,
    )
    df_adjusted = df_emissions_rate.copy()
    df_adjusted.loc[
        df_adjusted["can_remove_outliers"], delta_years_str
    ] = df_adjusted.loc[df_adjusted["can_remove_outliers"], delta_years_str].mask(
        (df_adjusted[delta_years_str] > threshold)
        | (df_adjusted[delta_years_str] < -threshold)
    )
    # df_adjusted = df_adjusted.drop('can_remove_outliers', axis=1)

    return df_adjusted


def adj_3_replace_emissions_with_intensities(
    df_emissions_rate,
    df_intensities_rate,
    threshold,
    delta_years_str=[f"{int(y)}-{y+1}" for y in range(2019, 2024)],
):
    """Both df are already filtered for scopes, one row by equity"""

    df_adjusted = df_emissions_rate.copy()
    df_adjusted = df_adjusted.set_index("isin")
    df_intensities_rate = df_intensities_rate.set_index("isin").reindex(
        df_adjusted.index
    )
    df_adjusted[delta_years_str] = df_adjusted[delta_years_str].mask(
        (
            (df_adjusted[delta_years_str] > threshold)
            | (df_adjusted[delta_years_str] < -threshold)
        )
        & (df_intensities_rate[delta_years_str].notna())
        & (
            (df_intensities_rate[delta_years_str] < threshold)
            & (df_intensities_rate[delta_years_str] > -threshold)
        ),
        df_intensities_rate[delta_years_str],
    )

    df_adjusted = df_adjusted.reset_index()

    return df_adjusted


def create_columns_number_outliers(
    df_emissions,
    threshold,
    delta_years_str=[f"{int(y)}-{y+1}" for y in range(2019, 2024)],
):

    df_emissions = df_emissions.copy()
    if threshold is None:
        df_emissions[
            [
                "number_positive_outliers",
                "number_negative_outliers",
                "number_all_outliers",
            ]
        ] = 0
        return df_emissions

    df_emissions["number_positive_outliers"] = (
        df_emissions[delta_years_str] > threshold
    ).sum(axis=1)
    df_emissions["number_negative_outliers"] = (
        df_emissions[delta_years_str] < -threshold
    ).sum(axis=1)
    df_emissions["number_all_outliers"] = (
        (df_emissions[delta_years_str] > threshold)
        | (df_emissions[delta_years_str] < -threshold)
    ).sum(axis=1)

    return df_emissions


def treat_outlier_by_sector(
    df_eq_infos,
    df_abs_emissions_rate,
    df_intensities_rate,
    sectors,
    regions,
    list_adjustements,
    threshold_outliers,
    col_sector="high_impact_sector",
    col_region="region_0",
    delta_years_str=[f"{int(y)}-{y+1}" for y in range(2019, 2024)],
):
    """
    Parameters:
    df_eq_infos: pd.DataFrame
        must contain "isin", "is_relevant_scopes", one column for the sectors and one for the region. Name of the columns must be mentionned in the parameters. There are defaults name
    df_abs_emissions_rate, df_intensities_rate: pd.DataFrames
    sector, region: str
        must be in the col_sector or col_region categories mentionned
    list_adjustements: list
        contains "adj_1", "adj_2" etc. depending on the sector and region selected?
    threshold_outlier: float
        positive number between 0 and 1
    col_sector, col_region: str
        name of the columns that give the equity sector or region selectionned
    delta_years_str: list
        years intervals like ["2019-2020", "2020-2021"]
    """

    if sectors[0] not in list(df_eq_infos[col_sector].unique()):
        print("parameter sector or col_sector incorrect")
        return df_abs_emissions_rate

    if regions[0] not in list(df_eq_infos[col_region].unique()):
        print("parameter region or col_region incorrect")
        return df_abs_emissions_rate

    df_selected = df_eq_infos[
        (df_eq_infos[col_sector].isin(sectors))
        & (df_eq_infos[col_region].isin(regions))
    ]
    df_selected = df_selected[df_selected["is_relevant_scopes"]][["isin", "scope"]]

    df_abs_em_rate_selected = pd.merge(
        df_selected, df_abs_emissions_rate, how="left", on=["isin", "scope"]
    )

    df_adjusted = df_abs_em_rate_selected.copy()

    if "adj_3" in list_adjustements:
        df_int_rate_selected = pd.merge(
            df_selected, df_intensities_rate, how="left", on=["isin", "scope"]
        )

    for adj in list_adjustements:

        if adj == "adj_1":
            df_adjusted = adj_1_keep_only_the_3_last_years(
                df_adjusted, delta_years_str=delta_years_str
            )

        if adj == "adj_2":
            df_adjusted = adj_2_unique_outlier_or_not_if_nbr_obs_inf_3(
                df_adjusted, threshold_outliers, delta_years_str=delta_years_str
            )

        if adj == "adj_3":
            df_adjusted = adj_3_replace_emissions_with_intensities(
                df_adjusted,
                df_int_rate_selected,
                threshold_outliers,
                delta_years_str=delta_years_str,
            )

        df_adjusted = create_columns_number_outliers(df_adjusted, threshold_outliers)
        df_adjusted["nbr_observations"] = (
            df_adjusted[delta_years_str].notna().sum(axis=1)
        )

    df_adjusted = df_adjusted.drop(
        [
            "number_positive_outliers",
            "number_negative_outliers",
            "number_all_outliers",
            "nbr_observations",
            "can_remove_outliers",
        ],
        axis=1,
    )

    return df_adjusted


# ----------------------------
# Parameters
# ----------------------------

first_year_available, last_year_available = 2019, 2024
delta_years_str = [
    f"{int(y)}-{y+1}" for y in range(first_year_available, last_year_available)
]

# ----------------------------
# Load data
# ----------------------------

df = pd.read_parquet(
    f"data/intermediate_data/df_merged_all_infos_{first_year_available}_{last_year_available}.parquet"
)
df_abs_em_rate = pd.read_parquet(
    f"data/intermediate_data/hist_abs_emissions_growth_rate_{first_year_available}_{last_year_available}.parquet"
)
df_intensities_rate = pd.read_parquet(
    f"data/intermediate_data/hist_intensities_growth_rate_{first_year_available}_{last_year_available}.parquet"
)

regions = list(df["region_0"].unique())
list_high_impact_sector = list(df["high_impact_sector"].dropna().sort_values().unique())

dict_sectors_adjustements = {
    sector: ["adj_1", "adj_2"] for sector in list_high_impact_sector
}

dict_thresholds = {}
for region in regions:
    dict_thresholds[region] = {}
    for sector in list_high_impact_sector:
        dict_thresholds[region][sector] = 0.5


list_df_adjusted_sector = []
for sector in list_high_impact_sector:
    df_adjusted = treat_outlier_by_sector(
        df,
        df_abs_em_rate,
        df_intensities_rate,
        [sector],
        regions,
        dict_sectors_adjustements[sector],
        0.5,
        col_sector="high_impact_sector",
        delta_years_str=delta_years_str,
    )
    list_df_adjusted_sector.append(df_adjusted)

df_emissions_adjusted = pd.concat(list_df_adjusted_sector)
df_emissions_adjusted.to_csv(
    f"data/intermediate_data/df_emissions_rate_adjusted_{first_year_available}_{last_year_available}.csv"
)
