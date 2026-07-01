import numpy as np


# ==========================================================
# BALANCE FEATURES
# ==========================================================

def add_balance_features(df):

    df["origin_delta"] = (
        df["origin_balance_before"]
        - df["origin_balance_after"]
    )

    df["destination_delta"] = (
        df["destination_balance_after"]
        - df["destination_balance_before"]
    )

    df["delta_difference"] = (
        df["origin_delta"]
        - df["destination_delta"]
    )

    return df


# ==========================================================
# RATIO FEATURES
# ==========================================================

def add_ratio_features(df):

    eps = 1e-6

    df["amount_origin_ratio"] = (
        df["amount"]
        / (df["origin_balance_before"] + eps)
    )

    df["amount_destination_ratio"] = (
        df["amount"]
        / (df["destination_balance_before"] + eps)
    )

    df["origin_after_ratio"] = (
        df["origin_balance_after"]
        / (df["origin_balance_before"] + eps)
    )

    df["destination_after_ratio"] = (
        df["destination_balance_after"]
        / (df["destination_balance_before"] + eps)
    )


    return df


# ==========================================================
# ACCOUNT FEATURES
# ==========================================================

def add_account_features(df):

    df["origin_negative"] = (
        df["origin_balance_after"] < 0
    ).astype(int)

    df["destination_negative"] = (
        df["destination_balance_after"] < 0
    ).astype(int)

    df["origin_balance_zero"] = (
        df["origin_balance_before"] == 0
    ).astype(int)

    df["destination_balance_zero"] = (
        df["destination_balance_before"] == 0
    ).astype(int)

    df["amount_greater_origin"] = (
        df["amount"] > df["origin_balance_before"]
    ).astype(int)

    df["origin_balance_unchanged"] = (
        df["origin_balance_before"]
        == df["origin_balance_after"]
    ).astype(int)

    return df

def add_frequency_features(df):

    origin_freq = (
        df["origin_account"]
        .value_counts()
    )

    destination_freq = (
        df["destination_account"]
        .value_counts()
    )

    df["origin_frequency"] = (
        df["origin_account"]
        .map(origin_freq)
    )

    df["destination_frequency"] = (
        df["destination_account"]
        .map(destination_freq)
    )

    return df

# ==========================================================
# AGGREGATION FEATURES
# ==========================================================

def add_aggregation_features(df):

    eps = 1e-6

    df["amount_vs_origin_mean"] = (
        df["amount"] /
        (df["origin_amount_mean"] + eps)
    )

    df["amount_vs_destination_mean"] = (
        df["amount"] /
        (df["destination_amount_mean"] + eps)
    )

    df["amount_vs_origin_max"] = (
        df["amount"] /
        (df["origin_amount_max"] + eps)
    )

    df["amount_vs_destination_max"] = (
        df["amount"] /
        (df["destination_amount_max"] + eps)
    )

    df["origin_transaction_density"] = (
        df["origin_transaction_count"] /
        (df["period"] + 1)
    )

    df["destination_transaction_density"] = (
        df["destination_transaction_count"] /
        (df["period"] + 1)
    )


    return df

def add_behavior_features(df):

    eps = 1e-6



    # ==============================
    # 2. Z-score du montant
    # ==============================

    df["amount_zscore"] = (

        df["amount"]

        - df["origin_amount_mean"]

    ) / (

        df["origin_amount_std"] + eps

    )


    # ==============================
    # 4. Ratio de fréquence
    # ==============================

    df["frequency_ratio"] = (

        df["origin_frequency"]

        /

        (

            df["destination_frequency"]

            + eps

        )

    )

    # ==============================
    # 5. Ratio delta / montant
    # ==============================

    df["origin_delta_ratio"] = (

        df["origin_delta"]

        /

        (

            df["amount"]

            + eps

        )


    )
    df["amount_zscore"] = (
        df["amount_zscore"]
        .fillna(0)
    )


    return df
# ==========================================================
# BUILD FEATURES
# ==========================================================
def build_features(df):

    df = add_balance_features(df)

    df = add_ratio_features(df)

    df = add_account_features(df)

    df = add_frequency_features(df)

    df = add_aggregation_features(df)

    df = add_behavior_features(df)

    df = df.replace([np.inf, -np.inf], 0)

    df = df.fillna(0)

    return df

