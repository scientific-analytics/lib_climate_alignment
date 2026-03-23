import pandas as pd

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
df_nbr_eq_below_trend = (
    df_eq_emissions_adjusted.groupby(["region_0", "high_impact_sector"])[
        "is_average_rate_below_trend"
    ]
    .sum()
    .reset_index()
)

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

df_prop_eq_below_trend = (
    "'"
    + nbr_eq_below_trend.astype(int).astype(str)
    + "/"
    + total_nbr.astype(int).astype(str)
)

# ----------------------------
# Download
# ----------------------------
nbr_eq_below_trend.to_csv("output/nbr_eq_with_adj_em_below_trend.csv")
df_prop_eq_below_trend.to_csv("output/prop_eq_with_adj_em_below_trend.csv")
