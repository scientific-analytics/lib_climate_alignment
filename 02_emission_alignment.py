import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
trend_years_interval_col = "average_trend_2014_2024"
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

prop_eq_below_trend = (
    nbr_eq_below_trend.astype(int).astype(str) + "/" + total_nbr.astype(int).astype(str)
)

prop_float_eq_below_trend = (nbr_eq_below_trend / total_nbr).fillna(0)
prop_float_eq_below_trend = nbr_eq_below_trend.div(total_nbr).fillna(0)

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

    cmap = "RdYlGn_r" if reverse_colors else "RdYlGn"
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

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
    "output/table_region_sector_prop_eq_below_trend_frac.png",
    title="Proportion below trend",
)

plot_table_heatmap_with_text(
    prop_float_eq_missing,
    prop_eq_missing,
    "output/table_region_sector_prop_eq_missing_frac.png",
    title="Proportion missing",
    reverse_colors=True,
)
