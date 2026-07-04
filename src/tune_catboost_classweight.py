"""
src/tune_catboost_classweight.py

CatBoost s'est montré instable d'un seed à l'autre dans
train_final_bagged.py (0.357 / 0.325 / 0.342 / 0.340). Suspect
principal : class_weights=[1.0, scale_pos_weight] avec un ratio brut
(souvent 10-20x) combiné à une profondeur de 6 -- ça peut pousser le
modèle vers des règles trop tranchées, sensibles à l'initialisation.

Ce script compare 3 stratégies de gestion du déséquilibre pour
CatBoost, sur les 4 folds walk-forward déjà validés (aucun coût de
soumission, tout est local) :

  A. class_weights=[1, scale_pos_weight]     (stratégie actuelle)
  B. auto_class_weights='Balanced'           (CatBoost natif)
  C. auto_class_weights='SqrtBalanced'       (CatBoost natif, plus doux)

Pour chaque stratégie, on regarde :
  - le PR-AUC moyen sur les 4 folds (qualité)
  - l'écart-type entre folds (stabilité -- règle n°1 : un modèle
    stable qui généralise vaut mieux qu'un modèle qui overfit un
    seed chanceux)

Usage :
    python -m src.tune_catboost_classweight
"""

import numpy as np

from src.data_loader import load_train
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
from src.validation_v2 import time_series_splits, evaluate_predictions
from src.models import get_catboost

N_SPLITS = 4
DROP_COLUMNS = ["id", "fraud_flag", "origin_account", "destination_account"]
CAT_COL = "operation"

STRATEGIES = {
    "A_scale_pos_weight": {"mode": "manual"},
    "B_balanced": {"mode": "auto", "auto_class_weights": "Balanced"},
    "C_sqrt_balanced": {"mode": "auto", "auto_class_weights": "SqrtBalanced"},
}


def build_fold_features(train_fold, valid_fold):

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

    return train_fold, valid_fold


def main():

    from catboost import Pool

    train = load_train()
    print("Train :", train.shape)

    splits = time_series_splits(train, n_splits=N_SPLITS, min_train_frac=0.4)

    results = {name: [] for name in STRATEGIES}

    for fold, (train_idx, valid_idx) in enumerate(splits):

        print(f"\n{'=' * 70}\nFOLD {fold}\n{'=' * 70}")

        train_fold = train.iloc[train_idx].copy()
        valid_fold = train.iloc[valid_idx].copy()
        train_fold, valid_fold = build_fold_features(train_fold, valid_fold)

        X_train = train_fold.drop(columns=DROP_COLUMNS)
        y_train = train_fold["fraud_flag"]
        X_valid = valid_fold.drop(columns=DROP_COLUMNS)
        y_valid = valid_fold["fraud_flag"]

        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale_pos_weight = neg / max(pos, 1)

        train_pool = Pool(X_train, y_train, cat_features=[CAT_COL])
        valid_pool = Pool(X_valid, y_valid, cat_features=[CAT_COL])

        for name, cfg in STRATEGIES.items():

            model = get_catboost()

            if cfg["mode"] == "manual":
                model.set_params(
                    class_weights=[1.0, scale_pos_weight],
                    iterations=1500,
                    early_stopping_rounds=100,
                )
            else:
                model.set_params(
                    auto_class_weights=cfg["auto_class_weights"],
                    iterations=1500,
                    early_stopping_rounds=100,
                )

            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            pred = model.predict_proba(X_valid)[:, 1]
            score = evaluate_predictions(y_valid, pred)
            results[name].append(score)
            print(f"{name:25s} : {score:.6f}")

    print(f"\n{'=' * 70}\nRÉCAPITULATIF (moyenne +/- écart-type sur {N_SPLITS} folds)\n{'=' * 70}")
    for name, scores in results.items():
        scores = np.array(scores)
        print(f"{name:25s} : {scores.mean():.6f} +/- {scores.std():.6f}  (folds: {np.round(scores, 4).tolist()})")

    best = max(results, key=lambda n: np.mean(results[n]))
    print(f"\nMeilleure stratégie (moyenne la plus haute) : {best}")

    most_stable = min(results, key=lambda n: np.std(results[n]))
    print(f"Stratégie la plus stable (écart-type le plus bas) : {most_stable}")


if __name__ == "__main__":
    main()
