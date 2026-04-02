import pandas as pd
import numpy as np


# ----------------------------
# Functions
# ----------------------------


def create_boolean_table_is_valid_value(df, threshold):
    return (df.notna()) & (df.abs() <= threshold)


def are_3_valid_values(boolean_table_is_valid_value):
    return boolean_table_is_valid_value.sum(axis=1) >= 3


def are_3_positive_values(df):
    return (df > 0).sum(axis=1) > 3


def are_3_negative_values(df):
    return (df < 0).sum(axis=1) > 3


def treat_outliers_vVB(
    df_absolute_emissions_rate, df_intensities_rate, delta_years_str, threshold
):
    """
    Parameters:
    df_absolute_emissions_rate, df_intensities_rate: pd.DataFrame
        must contain a column 'isin' and the columns in the list delta_years_str
    delta_years_str: list
        name of columns like ['2021-2022', '2022-2023']
    threshold: float
        should be superior to -1
    """
    df_abs_em = df_absolute_emissions_rate.set_index(["isin", "scope"])[delta_years_str]
    df_int = df_intensities_rate.set_index(["isin", "scope"])[delta_years_str]

    df_abs_em_is_valid_value = create_boolean_table_is_valid_value(df_abs_em, threshold)
    df_int_is_valid_value = create_boolean_table_is_valid_value(df_int, threshold)

    # STEP 1
    df_with_3_valid_values_else_nan = df_abs_em.loc[
        are_3_valid_values(df_abs_em_is_valid_value)
    ].mask(~df_abs_em_is_valid_value)
    median_3_valid_values = df_with_3_valid_values_else_nan.median(axis=1)
    df_1st_imputation = df_with_3_valid_values_else_nan.fillna(
        pd.concat(
            [median_3_valid_values] * len(delta_years_str), axis=1, keys=delta_years_str
        )
    )
    df_1st_imputation["nbr_imput"] = (
        df_abs_em.loc[are_3_valid_values(df_abs_em_is_valid_value)]
        .mask(~df_abs_em_is_valid_value)
        .isna()
        .sum(axis=1)
    )
    df_1st_imputation["nbr_imput_1"] = df_1st_imputation["nbr_imput"].copy()

    # STEP 2
    df_without_3_valid_values = df_abs_em.loc[
        ~are_3_valid_values(df_abs_em_is_valid_value)
    ]
    df_without_3_valid_values_with_na = df_without_3_valid_values.mask(
        ~df_abs_em_is_valid_value
    )
    df_2nd_imputation = df_without_3_valid_values_with_na.fillna(
        (np.sign(df_int) * threshold).mask(~df_int_is_valid_value)
    )
    df_2nd_imputation["nbr_imput"] = (
        df_without_3_valid_values_with_na.isna() & df_int_is_valid_value
    ).sum(axis=1)

    df_2nd_imputation["nbr_imput_2"] = df_2nd_imputation["nbr_imput"].copy()

    # STEP 3
    df_3_positive_values = df_without_3_valid_values[
        are_3_positive_values(df_without_3_valid_values)
    ]
    df_3_negative_values = df_without_3_valid_values[
        are_3_negative_values(df_without_3_valid_values)
    ]
    df_3rd_imputation = df_2nd_imputation.fillna(
        pd.concat(
            [
                (
                    df_3_positive_values.notna().where(
                        df_3_positive_values.notna(), np.nan
                    )
                    * threshold
                ),
                (
                    df_3_negative_values.notna().where(
                        df_3_negative_values.notna(), np.nan
                    )
                    * -threshold
                ),
            ]
        )
    )

    nbr_positive_imput = (df_2nd_imputation.isna() & df_3_positive_values.notna()).sum(
        axis=1
    )
    nbr_negative_imput = (df_2nd_imputation.isna() & df_3_negative_values.notna()).sum(
        axis=1
    )

    df_3rd_imputation["nbr_imput"] = nbr_positive_imput[
        nbr_positive_imput > 0
    ].combine_first(nbr_negative_imput[nbr_negative_imput > 0])

    df_3rd_imputation["nbr_imput_3"] = df_3rd_imputation["nbr_imput"]

    df_3rd_imputation["nbr_imput"] = (
        df_3rd_imputation["nbr_imput"] + df_2nd_imputation["nbr_imput"]
    )

    df_all_imputation = pd.concat([df_1st_imputation, df_3rd_imputation]).loc[
        df_abs_em.index
    ]

    df_all_imputation["nbr_not_na"] = (
        df_all_imputation[delta_years_str].notna().sum(axis=1)
    )

    return df_all_imputation.reset_index()


# ----------------------------
# Parameters
# ----------------------------

first_year_available, last_year_available = 2019, 2024
delta_years_str = [
    f"{int(y)}-{y+1}" for y in range(first_year_available, last_year_available)
]
threshold = 0.3

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

# ----------------------------
# Create DataFrame with outliers treated
# ----------------------------

df_outliers_treated = treat_outliers_vVB(
    df_abs_em_rate, df_intensities_rate, delta_years_str, threshold
)

df_em_after_outliers_treatment = df_outliers_treated[
    (df_outliers_treated["nbr_imput"] < 3) & (df_outliers_treated["nbr_not_na"] >= 3)
]
# df_em_after_outliers_treatment.to_parquet(
#     "data/intermediate_data/df_em_after_outliers_treatment.parquet"
# )


# ----------------------------
# Visualise coverage
# ----------------------------

import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_sector_region_fraction_heatmap_with_text(
    numerator_df,
    denominator_df,
    output_file,
    title="",
    reverse_colors=False,
    region_col="region_0",
    sector_col="high_impact_sector",
    value_col="isin",
    show_percentage=False,
    figsize=(12, 8),
):
    # Comptages
    numerator = numerator_df.pivot_table(
        index=sector_col, columns=region_col, values=value_col, aggfunc="count"
    ).reindex(index=list_high_impact_sector, columns=regions)
    numerator = numerator.fillna(0)

    denominator = denominator_df.pivot_table(
        index=sector_col, columns=region_col, values=value_col, aggfunc="count"
    ).reindex(index=list_high_impact_sector, columns=regions)

    # Fraction pour la couleur
    df_color = numerator / denominator
    df_color = df_color.where(denominator != 0, np.nan)

    # Texte affiché
    df_text = pd.DataFrame("", index=df_color.index, columns=df_color.columns)

    for i in df_color.index:
        for j in df_color.columns:
            num = numerator.loc[i, j]
            den = denominator.loc[i, j]

            num = 0 if pd.isna(num) else int(num)
            den = 0 if pd.isna(den) else int(den)

            if den == 0:
                df_text.loc[i, j] = "0/0"
            elif show_percentage:
                frac = num / den
                df_text.loc[i, j] = f"{num}/{den}\n({frac:.0%})"
            else:
                df_text.loc[i, j] = f"{num}/{den}"

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    data = df_color.values.astype(float)

    cmap = mpl.colormaps["RdYlGn_r" if reverse_colors else "RdYlGn"].copy()
    cmap.set_bad(color="lightgrey")

    data_masked = np.where(np.isnan(data), np.nan, data)

    ax.imshow(data_masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(df_color.columns)))
    ax.set_xticklabels(df_color.columns, rotation=45, ha="right")

    ax.set_yticks(range(len(df_color.index)))
    ax.set_yticklabels(df_color.index)

    for i in range(df_color.shape[0]):
        for j in range(df_color.shape[1]):
            val = data[i, j]
            txt = df_text.iloc[i, j]

            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sector_region_count_heatmap(
    df,
    output_file,
    title="",
    reverse_colors=False,
    region_col="region_0",
    sector_col="high_impact_sector",
    value_col="isin",
    figsize=(12, 8),
):
    # Comptage
    table = df.pivot_table(
        index=sector_col, columns=region_col, values=value_col, aggfunc="count"
    )

    data = table.values.astype(float)

    cmap = mpl.colormaps["RdYlGn_r" if reverse_colors else "RdYlGn"].copy()
    cmap.set_bad(color="lightgrey")

    # échelle automatique
    vmin = np.nanmin(data)
    # vmax = np.nanmax(data)
    vmax = np.nanpercentile(data, 95)
    # sécurité si constant
    if vmin == vmax:
        vmax = vmin + 1

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # Axes
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=45, ha="right")

    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)

    # Option : afficher les valeurs
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = data[i, j]
            txt = "" if np.isnan(val) else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


df_merge = pd.merge(
    df[df["is_relevant_scopes"] == True],
    df_em_after_outliers_treatment,
    on=["isin", "scope"],
    how="inner",
)

plot_sector_region_fraction_heatmap_with_text(
    numerator_df=df_merge,
    denominator_df=df[df["is_relevant_scopes"] == True],
    output_file="output/fraction_remaining_eq_after_outlier_treatment_heatmap.png",
    title="Coverage after outliers treatment by region and sector",
    show_percentage=False,
)

# plot_sector_region_count_heatmap(
#     df=df_merge,
#     output_file="output/count_remaining_eq_after_outlier_treatment_heatmap.png",
#     title="Nbr eq after outliers treatment by region and sector",
# )
