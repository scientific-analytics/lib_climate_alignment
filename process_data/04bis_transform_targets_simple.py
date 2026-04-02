import pandas as pd
import numpy as np

# ----------------------------
# Functions
# ----------------------------


def create_df_with_climate_target_ambition_columns(
    df_raw_targets, col_ambition_target="climate_target_ambition"
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

    return df_targets_ambition


# ----------------------------
# Create Dataframe
# ----------------------------

first_year_available = 2019
last_year_available = 2024

raw_targets = pd.read_parquet(
    f"data/raw_data/iss_raw/iss_targets_{last_year_available}.parquet"
)

df_targets_ambition = create_df_with_climate_target_ambition_columns(raw_targets)

df_targets_ambition.to_excel("data/intermediate_data/df_eq_target_ambitions.xlsx")
