import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# ----------------------------
# Load data
# ----------------------------
first_year_available, last_year_available = 2019, 2024

df = pd.read_parquet(
    f"data/intermediate_data/df_merged_all_infos_{first_year_available}_{last_year_available}.parquet"
)
df_hist_trends = pd.read_csv(
    "data/intermediate_data/region_sector_historical_trends.csv"
)
df_emissions_adjusted = pd.read_csv(
    f"data/intermediate_data/df_emissions_rate_adjusted_{first_year_available}_{last_year_available}.csv"
)

# ----------------------------
# Parameters
# ----------------------------
regions = list(df["region_0"].unique())
list_high_impact_sector = list(df["high_impact_sector"].dropna().sort_values().unique())
trend_years_interval_col = (
    "average_trend_2021_2024"  # "average_trend_2021_2024"  # average_trend_2014_2024
)
delta_years_str = [
    f"{int(y)}-{y+1}" for y in range(first_year_available, last_year_available)
]

# ----------------------------
# Find aligned equities
# ----------------------------
df_eq_emissions_adjusted = pd.merge(
    df[["isin", "region_0", "high_impact_sector"]].drop_duplicates(),
    df_emissions_adjusted,
    on="isin",
    how="right",
)
df_eq_emissions_adjusted = pd.merge(
    df_eq_emissions_adjusted,
    df_hist_trends.rename(columns={"region": "region_0"}),
    on=["region_0", "high_impact_sector"],
    how="left",
)
df_eq_emissions_adjusted["average_rate"] = df_eq_emissions_adjusted[
    delta_years_str
].mean(axis=1)
df_eq_emissions_adjusted["is_average_rate_below_trend"] = (
    df_eq_emissions_adjusted["average_rate"]
    <= df_eq_emissions_adjusted[trend_years_interval_col]
)

# ----------------------------
# Final table
# ----------------------------

nbr_eq_below_trend = (
    df_eq_emissions_adjusted.groupby(["region_0", "high_impact_sector"])[
        "is_average_rate_below_trend"
    ]
    .sum()
    .reset_index()
    .pivot(
        index="high_impact_sector",
        columns="region_0",
        values="is_average_rate_below_trend",
    )
    .fillna(0)
)

total_nbr = (
    df_eq_emissions_adjusted.groupby(["region_0", "high_impact_sector"])["isin"]
    .count()
    .reset_index()
    .pivot(index="high_impact_sector", columns="region_0", values="isin")
    .fillna(0)
)

prop_float_eq_below_trend = nbr_eq_below_trend.div(total_nbr)

prop_eq_below_trend = (
    nbr_eq_below_trend.astype(int).astype(str) + "/" + total_nbr.astype(int).astype(str)
)

# missing
df_eq_emissions_adjusted["is_average_rate_missing"] = df_eq_emissions_adjusted[
    "average_rate"
].isna()

nbr_eq_missing = (
    df_eq_emissions_adjusted.groupby(["region_0", "high_impact_sector"])[
        "is_average_rate_missing"
    ]
    .sum()
    .reset_index()
    .pivot(
        index="high_impact_sector", columns="region_0", values="is_average_rate_missing"
    )
    .fillna(0)
)

prop_eq_missing = (
    nbr_eq_missing.astype(int).astype(str) + "/" + total_nbr.astype(int).astype(str)
)

prop_float_eq_missing = nbr_eq_missing.div(total_nbr).fillna(0)


def plot_table_heatmap_with_text(
    df_color, df_text, output_file, title="", reverse_colors=False
):
    fig, ax = plt.subplots(figsize=(12, 8))

    data = df_color.values

    cmap = mpl.colormaps["RdYlGn_r" if reverse_colors else "RdYlGn"].copy()
    cmap.set_bad(color="lightgrey")  # couleur des NaN

    # masque les cases où denom = 0 → NaN dans df_color
    data_masked = np.where(np.isnan(data), np.nan, data)

    im = ax.imshow(data_masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(df_color.columns)))
    ax.set_xticklabels(df_color.columns, rotation=45, ha="right")

    ax.set_yticks(range(len(df_color.index)))
    ax.set_yticklabels(df_color.index)

    for i in range(df_color.shape[0]):
        for j in range(df_color.shape[1]):
            val = data[i, j]
            txt = df_text.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
            else:
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


# ----------------------------
# Download
# ----------------------------
# nbr_eq_below_trend.to_excel("output/nbr_eq_with_adj_em_below_trend.xlsx")
# prop_float_eq_below_trend.to_excel("output/prop_float_eq_with_adj_em_below_trend.xlsx")
# prop_float_eq_missing.to_excel("output/prop_float_eq_missing.xlsx")

plot_table_heatmap_with_text(
    prop_float_eq_below_trend,
    prop_eq_below_trend,
    f"output/table_region_sector_prop_eq_below_trend_{trend_years_interval_col[-9:-5]}_{trend_years_interval_col[-4:]}_frac.png",
    title="Proportion below trend",
)

# plot_table_heatmap_with_text(
#     prop_float_eq_missing,
#     prop_eq_missing,
#     "output/table_region_sector_prop_eq_missing_frac.png",
#     title="Proportion missing",
#     reverse_colors=True,
# )

# ----------------------------
# Plot historic trend
# ----------------------------
df_average_trends = (
    df_hist_trends.rename(
        columns={"region": "region_0", trend_years_interval_col: "average_rate"}
    )
    .groupby(["region_0", "high_impact_sector"])["average_rate"]
    .mean()
    .reset_index()
)

df_average_trends_regions = (
    df_hist_trends.rename(
        columns={"region": "region_0", trend_years_interval_col: "average_rate"}
    )
    .groupby("region_0")["average_rate"]
    .mean()
)

df_average_trends_sectors = (
    df_hist_trends.rename(
        columns={"region": "region_0", trend_years_interval_col: "average_rate"}
    )
    .groupby("high_impact_sector")["average_rate"]
    .mean()
)

average_trend_all = df_hist_trends[trend_years_interval_col].mean()


def plot_sector_region_trends_heatmap_with_totals(
    df,
    region_totals,
    sector_totals,
    totals,
    output_file,
    title="",
    reverse_colors=False,
    value_col="average_rate",
    region_col="region_0",
    sector_col="high_impact_sector",
    figsize=(12, 8),
):
    # Pivot principal
    table = df.pivot(index=sector_col, columns=region_col, values=value_col)

    # Alignement
    region_totals = region_totals.reindex(table.columns)
    sector_totals = sector_totals.reindex(table.index)

    # Ajout colonne Total
    table["Total"] = sector_totals

    # Ajout ligne Total
    total_row = pd.Series(index=table.columns, dtype=float)
    total_row.loc[region_totals.index] = region_totals.values
    total_row.loc["Total"] = totals  # ou df[value_col].mean()
    table.loc["Total"] = total_row

    data = table.values.astype(float)

    cmap = mpl.colormaps["RdYlGn_r" if reverse_colors else "RdYlGn"].copy()
    cmap.set_bad(color="lightgrey")

    vmax = np.nanmax(np.abs(data))
    vmin = -vmax if not np.isnan(vmax) else -1
    vmax = vmax if not np.isnan(vmax) and vmax != 0 else 1

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=45, ha="right")

    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = data[i, j]
            txt = "" if np.isnan(val) else f"{val*100:+.1f}%"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

    # séparation visuelle des totaux
    n_rows, n_cols = table.shape
    ax.axvline(x=n_cols - 1 - 0.5, color="black", linewidth=2)
    ax.axhline(y=n_rows - 1 - 0.5, color="black", linewidth=2)

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


plot_sector_region_trends_heatmap_with_totals(
    df=df_average_trends,
    region_totals=df_average_trends_regions,
    sector_totals=df_average_trends_sectors,
    totals=average_trend_all,
    output_file=f"output/regions_sectors_hist_{trend_years_interval_col}.png",
    title=f"Average hist trends btwn {trend_years_interval_col[-9:-5]} and {trend_years_interval_col[-4:]} by region and sector",
    reverse_colors=True,
)
