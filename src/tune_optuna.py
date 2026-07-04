"""
src/tune_optuna.py

Les hyperparamètres actuels (src/models.py) ont été choisis à la main,
jamais optimisés. Ce script utilise Optuna (déjà dans requirements.txt
mais jusqu'ici inutilisé) pour chercher de meilleurs hyperparamètres,
en utilisant la validation walk-forward (time_series_splits) comme
objectif -- la même méthode qui a déjà permis de valider votre
meilleure soumission (submission_blend_weighted.csv, 0.353833).

Aucun coût de soumission : tout se passe en local.

Usage :
    python -m src.tune_optuna --model catboost --trials 20
    python -m src.tune_optuna --model xgboost --trials 20
    python -m src.tune_optuna --model lightgbm --trials 20

A la fin, les meilleurs hyperparamètres trouvés sont affichés ET
sauvegardés dans models/{model}_best_params.pkl, prêts à être
réinjectés dans train_stack_temporal.py (voir instructions affichées
en fin de script).
"""

import argparse
import numpy as np
import joblib
import optuna

from src.config import MODEL_DIR
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

N_SPLITS = 3  # réduit à 3 (au lieu de 4) pour accélérer le tuning ; on
              # revalidera avec 4 folds une fois les meilleurs params trouvés
DROP_COLUMNS = ["id", "fraud_flag", "origin_account", "destination_account"]
CAT_COL = "operation"

optuna.logging.set_verbosity(optuna.logging.WARNING)


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


def prepare_folds(train):
    """Pré-calcule les features pour chaque fold UNE SEULE FOIS,
    réutilisées à chaque essai Optuna (évite de refaire le feature
    engineering à chaque trial -- gros gain de temps)."""

    splits = time_series_splits(train, n_splits=N_SPLITS, min_train_frac=0.5)

    prepared = []
    for train_idx, valid_idx in splits:
        train_fold = train.iloc[train_idx].copy()
        valid_fold = train.iloc[valid_idx].copy()
        train_fold, valid_fold = build_fold_features(train_fold, valid_fold)

        X_train = train_fold.drop(columns=DROP_COLUMNS)
        y_train = train_fold["fraud_flag"]
        X_valid = valid_fold.drop(columns=DROP_COLUMNS)
        y_valid = valid_fold["fraud_flag"]

        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale_pos_weight = neg / max(pos, 1)

        prepared.append((X_train, y_train, X_valid, y_valid, scale_pos_weight))

    return prepared


def objective_catboost(trial, folds):

    from catboost import Pool, CatBoostClassifier

    params = {
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
    }

    scores = []
    for X_train, y_train, X_valid, y_valid, scale_pos_weight in folds:

        train_pool = Pool(X_train, y_train, cat_features=[CAT_COL])
        valid_pool = Pool(X_valid, y_valid, cat_features=[CAT_COL])

        model = CatBoostClassifier(
            iterations=1000,
            loss_function="Logloss",
            eval_metric="PRAUC",
            class_weights=[1.0, scale_pos_weight],
            early_stopping_rounds=80,
            random_seed=42,
            verbose=False,
            **params,
        )
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        pred = model.predict_proba(X_valid)[:, 1]
        scores.append(evaluate_predictions(y_valid, pred))

    return float(np.mean(scores))


def objective_xgboost(trial, folds):

    from xgboost import XGBClassifier

    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    scores = []
    for X_train, y_train, X_valid, y_valid, scale_pos_weight in folds:

        X_train_c = X_train.copy()
        X_valid_c = X_valid.copy()
        X_train_c[CAT_COL] = X_train_c[CAT_COL].astype("category")
        X_valid_c[CAT_COL] = X_valid_c[CAT_COL].astype("category")

        model = XGBClassifier(
            n_estimators=1000,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            enable_categorical=True,
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=80,
            random_state=42,
            n_jobs=-1,
            **params,
        )
        model.fit(
            X_train_c, y_train,
            eval_set=[(X_valid_c, y_valid)],
            verbose=False,
        )
        pred = model.predict_proba(X_valid_c)[:, 1]
        scores.append(evaluate_predictions(y_valid, pred))

    return float(np.mean(scores))


def objective_lightgbm(trial, folds, operation_mapping, unknown_code):

    from lightgbm import LGBMClassifier, early_stopping

    params = {
        "num_leaves": trial.suggest_int("num_leaves", 31, 511),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    scores = []
    for X_train, y_train, X_valid, y_valid, scale_pos_weight in folds:

        X_train_c = X_train.copy()
        X_valid_c = X_valid.copy()
        X_train_c[CAT_COL] = X_train_c[CAT_COL].map(operation_mapping).fillna(unknown_code).astype("int32")
        X_valid_c[CAT_COL] = X_valid_c[CAT_COL].map(operation_mapping).fillna(unknown_code).astype("int32")

        model = LGBMClassifier(
            n_estimators=1500,
            objective="binary",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
            bagging_freq=5,
            **params,
        )
        model.fit(
            X_train_c, y_train,
            eval_set=[(X_valid_c, y_valid)],
            eval_metric="average_precision",
            callbacks=[early_stopping(80, verbose=False)],
        )
        pred = model.predict_proba(X_valid_c)[:, 1]
        scores.append(evaluate_predictions(y_valid, pred))

    return float(np.mean(scores))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["catboost", "xgboost", "lightgbm"], required=True)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    print("=" * 70)
    print(f"Tuning Optuna : {args.model} ({args.trials} essais, {N_SPLITS} folds walk-forward)")
    print("=" * 70)

    train = load_train()
    folds = prepare_folds(train)

    if args.model == "catboost":
        objective = lambda trial: objective_catboost(trial, folds)
    elif args.model == "xgboost":
        objective = lambda trial: objective_xgboost(trial, folds)
    else:
        categories = sorted(train[CAT_COL].unique())
        mapping = {v: i for i, v in enumerate(categories)}
        unknown_code = len(categories)
        objective = lambda trial: objective_lightgbm(trial, folds, mapping, unknown_code)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print("\n" + "=" * 70)
    print(f"MEILLEUR SCORE ({args.model}) : {study.best_value:.6f}")
    print("MEILLEURS HYPERPARAMÈTRES :")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 70)

    joblib.dump(study.best_params, MODEL_DIR / f"{args.model}_best_params.pkl")
    print(f"\nSauvegardé dans {MODEL_DIR / f'{args.model}_best_params.pkl'}")
    print(
        "\nPour comparer : le score walk-forward actuel (hyperparamètres "
        "par défaut de src/models.py) était de :\n"
        "  catboost ~0.371 (moyenne sur 4 folds, cf tune_catboost_classweight.py)\n"
        "  xgboost  ~0.372 (OOF global, cf train_stack_temporal.py)\n"
        "  lightgbm ~0.338 (OOF global, cf train_stack_temporal.py)\n"
        "Si le score ci-dessus est notablement meilleur, il faut injecter "
        "ces hyperparamètres dans src/models.py avant de relancer "
        "train_stack_temporal.py + predict_stack_v2.py."
    )


if __name__ == "__main__":
    main()
