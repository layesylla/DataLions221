"""
Validation v2 — corrige un biais du split actuel.

PROBLÈME avec src/validation.py (split_data) :
    Le split train_test_split(..., stratify=fraud_flag) est un split
    ALÉATOIRE AU NIVEAU DE LA LIGNE. Comme un même compte
    (origin_account / destination_account) peut apparaître plusieurs
    fois dans le dataset, le MÊME compte se retrouve à la fois dans
    train_df et valid_df. Les features d'agrégation (moyenne, écart-
    type, fréquence par compte...) apprennent alors implicitement le
    "profil" du compte sur des lignes qui sont physiquement séparées
    en train/valid, ce qui optimise artificiellement le score de
    validation par rapport à la vraie généralisation sur des comptes
    peu/pas vus.

    Résultat concret : le score de CV local peut être supérieur au
    score obtenu sur le leaderboard, et les décisions prises en se
    fiant à la CV locale (choix de features, d'hyperparamètres) ne
    se traduisent pas forcément par un gain sur le LB.

SOLUTION : GroupKFold sur origin_account (ou sur la paire
origin+destination) pour garantir qu'un compte n'apparaît jamais à la
fois dans le fold d'entraînement et le fold de validation.

On fournit aussi une fonction de diagnostic qui mesure le taux de
recouvrement de comptes entre train et test : c'est LA chose à
vérifier en premier pour savoir quelle stratégie de validation est la
plus fidèle au leaderboard.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score


def evaluate_predictions(y_true, y_pred):
    return average_precision_score(y_true, y_pred)


def group_kfold_splits(df, n_splits=5, group_col="origin_account"):
    """
    Retourne une liste de (train_idx, valid_idx) façon GroupKFold,
    en garantissant qu'un même origin_account ne se retrouve jamais
    à cheval sur train et valid pour un même fold.
    """

    gkf = GroupKFold(n_splits=n_splits)

    groups = df[group_col]

    splits = list(
        gkf.split(df, df["fraud_flag"], groups=groups)
    )

    return splits


def diagnose_account_overlap(train_df, test_df):
    """
    A lancer UNE FOIS sur les vraies données de la compétition.

    Donne le % de comptes de test déjà vus dans train. C'est
    l'information clé pour choisir la stratégie de validation :

    - Si le recouvrement est FORT (>70-80%) : un split aléatoire par
      ligne (comme l'actuel) est presque acceptable, mais un
      GroupKFold reste plus prudent et donne une borne basse fiable.
    - Si le recouvrement est FAIBLE : le split actuel SURESTIME
      largement le score réel, et il faut absolument utiliser
      GroupKFold (ou un split temporel sur `period`) pour calibrer
      les décisions (choix de modèle, hyperparamètres, features) sur
      un score représentatif du leaderboard.
    """

    train_origin = set(train_df["origin_account"].unique())
    train_dest = set(train_df["destination_account"].unique())

    test_origin = set(test_df["origin_account"].unique())
    test_dest = set(test_df["destination_account"].unique())

    origin_overlap = len(test_origin & train_origin) / max(len(test_origin), 1)
    dest_overlap = len(test_dest & train_dest) / max(len(test_dest), 1)

    report = {
        "n_test_origin_accounts": len(test_origin),
        "pct_test_origin_seen_in_train": round(origin_overlap * 100, 2),
        "n_test_destination_accounts": len(test_dest),
        "pct_test_destination_seen_in_train": round(dest_overlap * 100, 2),
    }

    print("=" * 70)
    print("DIAGNOSTIC : recouvrement des comptes train / test")
    print("=" * 70)
    for k, v in report.items():
        print(f"{k:45s}: {v}")
    print("=" * 70)

    return report


def time_series_splits(df, period_col="period", n_splits=4, min_train_frac=0.4):
    """
    Validation "walk-forward" (backtesting), LA méthode à privilégier
    ici : le diagnostic (src/check_shift.py) a confirmé que train et
    test ont des périodes totalement disjointes (train = périodes
    anciennes, test = périodes futures jamais vues). GroupKFold
    mélange les périodes entre train/valid et surestime donc la
    performance réelle.

    Découpe la timeline en (n_splits + 1) blocs consécutifs. Pour
    chaque fold i, on entraîne sur tout ce qui précède le bloc i et on
    valide sur le bloc i lui-même (jamais vu). C'est exactement le
    scénario "prédire le futur" que la plateforme évalue.
    """

    periods_sorted = np.sort(df[period_col].unique())
    n_periods = len(periods_sorted)

    start_idx = int(n_periods * min_train_frac)
    fold_boundaries = np.linspace(start_idx, n_periods, n_splits + 1).astype(int)

    splits = []
    for i in range(n_splits):
        train_cutoff = periods_sorted[fold_boundaries[i] - 1]
        valid_start = periods_sorted[fold_boundaries[i]] if fold_boundaries[i] < n_periods else None
        valid_end_idx = fold_boundaries[i + 1] - 1
        valid_end = periods_sorted[min(valid_end_idx, n_periods - 1)]

        train_mask = df[period_col] <= train_cutoff
        valid_mask = (df[period_col] > train_cutoff) & (df[period_col] <= valid_end)

        train_idx = np.where(train_mask.values)[0]
        valid_idx = np.where(valid_mask.values)[0]

        if len(valid_idx) == 0:
            continue

        splits.append((train_idx, valid_idx))
        print(
            f"Fold {i}: train periods <= {train_cutoff} (n={len(train_idx)}), "
            f"valid periods in ({train_cutoff}, {valid_end}] (n={len(valid_idx)})"
        )

    return splits
    """
    Alternative au GroupKFold : split temporel simple.
    Entraîne sur les périodes les plus anciennes, valide sur les
    plus récentes (mime la logique public/private leaderboard qui
    évalue sur un test set "futur" par rapport au train).

    A utiliser en COMPLÉMENT du GroupKFold pour croiser les deux
    diagnostics (si les deux méthodes donnent un score similaire,
    on peut avoir confiance dans la CV).
    """

    periods_sorted = np.sort(df[period_col].unique())
    cutoff_idx = int(len(periods_sorted) * (1 - valid_frac))
    cutoff_period = periods_sorted[cutoff_idx]

    train_df = df[df[period_col] < cutoff_period].copy()
    valid_df = df[df[period_col] >= cutoff_period].copy()

    print(f"Split temporel : train periods < {cutoff_period}, "
          f"valid periods >= {cutoff_period}")
    print(f"Train : {train_df.shape}, Valid : {valid_df.shape}")

    return train_df, valid_df