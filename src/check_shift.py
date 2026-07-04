"""
src/check_shift.py

Diagnostic à lancer AVANT toute nouvelle soumission (aucun coût, tout
est local) :

    python -m src.check_shift

Objectif : comprendre pourquoi le score OOF (GroupKFold, 0.36+) est
très supérieur au score obtenu sur le leaderboard public (0.348-0.351).

Ce script :
  1. Compare la plage de `period` entre train et test (dérive
     temporelle : test est-il "dans le futur" par rapport à train ?)
  2. Compare le taux de fraude par tranche de période dans train (la
     fraude évolue-t-elle dans le temps ?)
  3. Entraîne un SEUL modèle CatBoost (rapide) avec deux stratégies
     de validation différentes :
       - GroupKFold (celle utilisée jusqu'ici)
       - Split temporel (train = périodes anciennes, valid = périodes
         récentes, comme le fait probablement la plateforme)
     et affiche les deux scores PR-AUC. Si le score "split temporel"
     est nettement plus bas que le score GroupKFold, la dérive
     temporelle est bien la cause principale de l'écart CV/LB.
"""

import numpy as np

from src.data_loader import load_train, load_test
from src.feature_engineering_v2 import (
    build_features_v2,
    build_pair_statistics,
    apply_pair_statistics,
)
from src.aggregation import (
    build_origin_statistics,
    build_destination_statistics,
    apply_origin_statistics,
    apply_destination_statistics,
)
from src.validation_v2 import (
    group_kfold_splits,
    time_based_split,
    evaluate_predictions,
)
from src.models import get_catboost

DROP_COLUMNS = ["id", "fraud_flag", "origin_account", "destination_account"]
CAT_COL = "operation"


def period_report(train, test):

    print("=" * 70)
    print("1. PLAGE DE PERIODES")
    print("=" * 70)
    print(f"Train period : min={train['period'].min()}  max={train['period'].max()}")
    print(f"Test  period : min={test['period'].min()}  max={test['period'].max()}")

    overlap = set(train["period"].unique()) & set(test["period"].unique())
    print(f"Nombre de périodes communes train/test : {len(overlap)}")
    print(f"Nombre de périodes uniques dans train   : {train['period'].nunique()}")
    print(f"Nombre de périodes uniques dans test    : {test['period'].nunique()}")

    print("\n" + "=" * 70)
    print("2. TAUX DE FRAUDE PAR TRANCHE DE PERIODE (train)")
    print("=" * 70)
    n_bins = 10
    train = train.copy()
    train["period_bin"] = pd_qcut_safe(train["period"], n_bins)
    print(
        train.groupby("period_bin")["fraud_flag"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "taux_fraude", "count": "nb_transactions"})
    )


def pd_qcut_safe(series, n_bins):
    import pandas as pd
    try:
        return pd.qcut(series, n_bins, duplicates="drop")
    except Exception:
        return pd.cut(series, n_bins)


def quick_catboost_score(train_fold, valid_fold, label):

    origin_stats = build_origin_statistics(train_fold)
    destination_stats = build_destination_statistics(train_fold)
    pair_stats = build_pair_statistics(train_fold)

    train_fold = apply_origin_statistics(train_fold, origin_stats)
    valid_fold = apply_origin_statistics(valid_fold, origin_stats)
    train_fold = apply_destination_statistics(train_fold, destination_stats)
    valid_fold = apply_destination_statistics(valid_fold, destination_stats)
    train_fold = apply_pair_statistics(train_fold, pair_stats)
    valid_fold = apply_pair_statistics(valid_fold, pair_stats)

    train_fold = build_features_v2(train_fold)
    valid_fold = build_features_v2(valid_fold)

    X_train = train_fold.drop(columns=DROP_COLUMNS)
    y_train = train_fold["fraud_flag"]
    X_valid = valid_fold.drop(columns=DROP_COLUMNS)
    y_valid = valid_fold["fraud_flag"]

    from catboost import Pool

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / max(pos, 1)

    train_pool = Pool(X_train, y_train, cat_features=[CAT_COL])
    valid_pool = Pool(X_valid, y_valid, cat_features=[CAT_COL])

    model = get_catboost()
    model.set_params(
        class_weights=[1.0, scale_pos_weight],
        iterations=800,
        early_stopping_rounds=80,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    pred = model.predict_proba(X_valid)[:, 1]
    score = evaluate_predictions(y_valid, pred)

    print(f"\n[{label}] CatBoost PR-AUC : {score:.6f} "
          f"(train={len(train_fold)}, valid={len(valid_fold)})")

    return score


def main():

    train = load_train()
    test = load_test()

    period_report(train, test)

    print("\n" + "=" * 70)
    print("3. COMPARAISON GroupKFold vs SPLIT TEMPOREL (CatBoost seul)")
    print("=" * 70)

    # -- GroupKFold : on prend juste le 1er fold pour aller vite
    splits = group_kfold_splits(train, n_splits=5, group_col="origin_account")
    train_idx, valid_idx = splits[0]
    train_fold_gk = train.iloc[train_idx].copy()
    valid_fold_gk = train.iloc[valid_idx].copy()
    score_groupkfold = quick_catboost_score(train_fold_gk, valid_fold_gk, "GroupKFold")

    # -- Split temporel : train = périodes anciennes, valid = récentes
    train_fold_time, valid_fold_time = time_based_split(train, valid_frac=0.2)
    score_time = quick_catboost_score(train_fold_time, valid_fold_time, "Split temporel")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"GroupKFold      : {score_groupkfold:.6f}")
    print(f"Split temporel  : {score_time:.6f}")
    ecart = score_groupkfold - score_time
    print(f"Ecart           : {ecart:+.6f}")
    if ecart > 0.02:
        print(
            "\n-> Ecart important : la dérive temporelle est probablement "
            "la cause principale du décalage CV/LB. Il faut privilégier "
            "un split (ou une pondération) temporel pour calibrer les "
            "décisions de modélisation, et éventuellement pondérer les "
            "transactions récentes plus fortement à l'entraînement."
        )
    else:
        print(
            "\n-> Ecart faible : la dérive temporelle n'explique "
            "probablement pas, à elle seule, le décalage observé sur le "
            "leaderboard. Le problème vient plus probablement du "
            "méta-modèle de stacking (coefficients trop extrêmes) -> "
            "privilégier un blend pondéré/rank-average plus robuste."
        )


if __name__ == "__main__":
    main()
