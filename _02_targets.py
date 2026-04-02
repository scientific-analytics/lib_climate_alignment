import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ----------------------------
# Parameters and Data
# ----------------------------

first_year_available, last_year_available = 2019, 2024

df = pd.read_parquet(
    f"data/intermediate_data/df_merged_all_infos_{first_year_available}_{last_year_available}.parquet"
)
df_em_after_outliers_treatment = pd.read_parquet(
    "data/intermediate_data/df_em_after_outliers_treatment.parquet"
)

df_targets = pd.read_excel("data/intermediate_data/df_eq_target_ambitions.xlsx")

regions = list(df["region_0"].unique())
list_high_impact_sector = list(df["high_impact_sector"].dropna().sort_values().unique())


# ----------------------------
# Function for heatmap
# ----------------------------


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

    # Ajout des Totaux
    numerator["Total"] = numerator.sum(axis=1)
    denominator["Total"] = denominator.sum(axis=1)

    numerator.loc["Total"] = numerator.sum(axis=0)
    denominator.loc["Total"] = denominator.sum(axis=0)

    # Alignement
    numerator, denominator = numerator.align(denominator, join="outer")

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

    # encadrement totaux
    n_rows, n_cols = df_color.shape

    ax.add_patch(
        plt.Rectangle(
            (n_cols - 1 - 0.5, -0.5),
            1,
            n_rows,
            fill=False,
            edgecolor="black",
            linewidth=2,
        )
    )

    ax.add_patch(
        plt.Rectangle(
            (-0.5, n_rows - 1 - 0.5),
            n_cols,
            1,
            fill=False,
            edgecolor="black",
            linewidth=2,
        )
    )

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


# ----------------------------
# Generate heatmaps
# ----------------------------

df_merge_targets_sbt_or_ambitious_or_commited = pd.merge(
    df[df["is_relevant_scopes"] == True][
        ["isin", "scope", "region_0", "high_impact_sector", "r_s_mc_weight"]
    ],
    df_targets[df_targets["is_target_sbt_or_ambitious_or_commited"] == True],
    on="isin",
    how="inner",
)

df_merge_targets_sbt_or_ambitious = pd.merge(
    df[df["is_relevant_scopes"] == True][
        ["isin", "scope", "region_0", "high_impact_sector", "r_s_mc_weight"]
    ],
    df_targets[df_targets["is_target_sbt_or_ambitious"] == True],
    on="isin",
    how="inner",
)

plot_sector_region_fraction_heatmap_with_text(
    numerator_df=df_merge_targets_sbt_or_ambitious_or_commited,
    denominator_df=df[df["is_relevant_scopes"] == True],
    output_file="output/fraction_target_heatmap.png",
    title="Number eq with target sbt, ambitious or commited by region and sector",
    show_percentage=False,
)

plot_sector_region_fraction_heatmap_with_text(
    numerator_df=df_merge_targets_sbt_or_ambitious,
    denominator_df=df[df["is_relevant_scopes"] == True],
    output_file="output/fraction_target_ambitious_heatmap.png",
    title="Number eq with target sbt, ambitious by region and sector",
    show_percentage=False,
)
