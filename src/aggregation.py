import pandas as pd


# ==========================================================
# ORIGIN STATISTICS
# ==========================================================

def build_origin_statistics(train_df):

    stats = (

        train_df

        .groupby("origin_account")

        .agg(

            origin_amount_mean=("amount", "mean"),
            origin_amount_std=("amount", "std"),
            origin_amount_max=("amount", "max"),
            origin_amount_min=("amount", "min"),
            origin_amount_median=("amount", "median"),
            origin_transaction_count=("amount", "count")

        )

        .reset_index()

    )

    stats["origin_amount_std"] = (
        stats["origin_amount_std"].fillna(0)
    )

    return stats


def apply_origin_statistics(df, stats):

    return df.merge(

        stats,

        on="origin_account",

        how="left"

    )


# ==========================================================
# DESTINATION STATISTICS
# ==========================================================

def build_destination_statistics(train_df):

    stats = (

        train_df

        .groupby("destination_account")

        .agg(

            destination_amount_mean=("amount", "mean"),
            destination_amount_std=("amount", "std"),
            destination_amount_max=("amount", "max"),
            destination_amount_min=("amount", "min"),
            destination_amount_median=("amount", "median"),
            destination_transaction_count=("amount", "count")

        )

        .reset_index()

    )

    stats["destination_amount_std"] = (
        stats["destination_amount_std"].fillna(0)
    )

    return stats


def apply_destination_statistics(df, stats):

    return df.merge(

        stats,

        on="destination_account",

        how="left"

    )


