"""
Feature engineering étendu — v2

Ce module NE REMPLACE PAS src/feature_engineering.py, il l'étend.
On garde toutes les features existantes (build_features) et on ajoute
des features à haute valeur pour la détection de fraude mobile money,
inspirées des patterns connus (PaySim / mobile money) :

  1. Features d'erreur de solde (errorBalanceOrig / errorBalanceDest)
     -> historiquement les features les PLUS prédictives sur ce type
        de données (transferts + cash-out simulés). Un compte source
        frauduleux vide souvent le compte, un compte destination
        frauduleux (mule) reste à 0 avant/après malgré le montant reçu.
  2. Flag de "vidage complet" du compte (full drain).
  3. Log-amount (la distribution est très asymétrique, cf EDA).
  4. Fréquence de la paire (origin_account, destination_account) —
     détecte les allers-retours répétés entre deux mêmes comptes.
  5. Flags "premier compte vu" (nouveaux comptes = souvent plus risqués).
  6. Z-score du montant côté destination (le fichier actuel ne le fait
     que côté origin).
  7. Auto-transaction (origin == destination).

Toutes les stats d'agrégation utilisées ici doivent être calculées sur
train UNIQUEMENT puis appliquées à train/valid/test (comme le fait déjà
aggregation.py) pour ne pas fuiter d'information.
"""

import numpy as np
import pandas as pd


# ==========================================================
# ERREUR DE SOLDE (feature la plus discriminante sur ce type de data)
# ==========================================================

def add_balance_error_features(df):

    eps = 1e-6

    # Si la comptabilité est cohérente : origin_balance_after
    # devrait valoir origin_balance_before - amount.
    # Un écart important est un signal fort d'anomalie/fraude.
    df["error_balance_origin"] = (
        df["origin_balance_after"]
        + df["amount"]
        - df["origin_balance_before"]
    )

    df["error_balance_destination"] = (
        df["destination_balance_before"]
        + df["amount"]
        - df["destination_balance_after"]
    )

    # Version normalisée (relative au montant) pour comparer les
    # erreurs entre petites et grosses transactions
    df["error_balance_origin_ratio"] = (
        df["error_balance_origin"] / (df["amount"] + eps)
    )

    df["error_balance_destination_ratio"] = (
        df["error_balance_destination"] / (df["amount"] + eps)
    )

    return df


# ==========================================================
# VIDAGE DE COMPTE / COMPTE "MULE"
# ==========================================================

def add_drain_features(df):

    # Compte source vidé complètement (pattern classique de fraude :
    # on transfère tout puis on abandonne le compte)
    df["origin_full_drain"] = (
        (df["origin_balance_before"] > 0)
        & (df["origin_balance_after"] <= 0)
    ).astype(int)

    # Compte destination qui reste à 0 avant ET après malgré la
    # réception d'argent -> comportement typique d'un compte "mule"
    # utilisé uniquement pour faire transiter l'argent
    df["destination_stays_zero"] = (
        (df["destination_balance_before"] == 0)
        & (df["destination_balance_after"] == 0)
        & (df["amount"] > 0)
    ).astype(int)

    # Auto-transaction (origin == destination) : souvent suspect
    df["is_self_transaction"] = (
        df["origin_account"] == df["destination_account"]
    ).astype(int)

    return df


# ==========================================================
# TRANSFORMATION LOG DU MONTANT
# ==========================================================

def add_log_features(df):

    df["amount_log"] = np.log1p(df["amount"].clip(lower=0))

    return df


# ==========================================================
# FRÉQUENCE DE LA PAIRE (origin, destination)
# ==========================================================

def build_pair_statistics(train_df):
    """A calculer sur train UNIQUEMENT, comme les stats origin/destination."""

    stats = (
        train_df
        .groupby(["origin_account", "destination_account"])
        .size()
        .reset_index(name="pair_transaction_count")
    )

    return stats


def apply_pair_statistics(df, stats):

    df = df.merge(
        stats,
        on=["origin_account", "destination_account"],
        how="left",
    )

    df["pair_transaction_count"] = (
        df["pair_transaction_count"].fillna(0)
    )

    return df


# ==========================================================
# NOUVEAUX COMPTES (jamais vus dans les stats train)
# ==========================================================

def add_new_account_flags(df):
    """
    A appeler APRES le merge des stats origin/destination.
    Si origin_transaction_count / destination_transaction_count sont
    NaN après le merge, c'est que le compte n'existe pas dans train
    -> compte "nouveau", souvent un profil plus risqué.
    """

    df["origin_is_new_account"] = (
        df["origin_transaction_count"].isna()
    ).astype(int)

    df["destination_is_new_account"] = (
        df["destination_transaction_count"].isna()
    ).astype(int)

    return df


# ==========================================================
# Z-SCORE DESTINATION (symétrique de celui déjà fait côté origin)
# ==========================================================

def add_destination_zscore(df):

    eps = 1e-6

    df["destination_amount_zscore"] = (
        (df["amount"] - df["destination_amount_mean"])
        / (df["destination_amount_std"] + eps)
    )

    df["destination_amount_zscore"] = (
        df["destination_amount_zscore"].fillna(0)
    )

    return df


# ==========================================================
# BUILD FEATURES V2 — combine tout
# ==========================================================

def build_features_v2(df):
    """
    IMPORTANT : df doit déjà avoir subi
        apply_origin_statistics / apply_destination_statistics
    (comme dans le pipeline existant), et build_pair_statistics /
    apply_pair_statistics doivent être appliqués AVANT cette fonction
    si on veut la feature pair_transaction_count.

    Cette fonction ajoute les nouvelles features PUIS appelle
    build_features() du module existant pour ne rien perdre.
    """
    from src.feature_engineering import build_features

    # Le flag "nouveau compte" doit être calculé AVANT le fillna(0)
    # de build_features() (sinon on ne peut plus distinguer un NaN
    # d'un vrai zéro), donc on le fait ici en premier.
    df = add_new_account_flags(df)

    df = add_balance_error_features(df)
    df = add_drain_features(df)
    df = add_log_features(df)
    df = add_destination_zscore(df) if "destination_amount_mean" in df.columns else df

    # Toutes les features "historiques" (delta, ratios, fréquences...)
    df = build_features(df)

    return df
