"""
src/train_stack_temporal.py

Version corrigée de train_stack.py suite au diagnostic de
src/check_shift.py : train et test ont des périodes totalement
disjointes (0-105 vs 106-143), donc GroupKFold surestimait la
performance réelle. Ce script utilise une validation walk-forward
(entraîner sur le passé, valider sur un futur jamais vu) qui reflète
fidèlement le scénario évalué par la plateforme.

Il remplace aussi le stacking par régression logistique (qui avait
produit des coefficients extrêmes, cat=50 -> surapprentissage du
méta-modèle) par un blend pondéré CONTRAINT (poids >= 0, somme = 1),
optimisé pour maximiser le PR-AUC sur les folds temporels. C'est une
combinaison beaucoup plus stable et moins sujette à l'overfitting
qu'une régression logistique libre sur seulement 3 colonnes.

Usage :
    python -m src.train_stack_temporal
    python -m src.predict_stack   # (inchangé, réutilise les modèles sauvegardés)
"""

import numpy as np
import joblib
from scipy.optimize import minimize

from src.config import MODEL_DIR
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
    time_series_splits,
    evaluate_predictions,
    diagnose_account_overlap,
)
from src.models import get_lightgbm, get_catboost, get_xgboost_best


N_SPLITS = 4
DROP_COLUMNS = ["id", "fraud_flag", "origin_account", "destination_account"]
CAT_COL = "operation"


def load_tuned_params(model_name):
    """Charge models/{model_name}_best_params.pkl s'il existe (produit
    par src/tune_optuna.py), sinon retourne un dict vide (comportement
    par défaut de src/models.py inchangé)."""

    path = MODEL_DIR / f"{model_name}_best_params.pkl"
    if path.exists():
        params = joblib.load(path)
        print(f"[{model_name}] Hyperparamètres tunés chargés depuis {path} : {params}")
        return params
    print(f"[{model_name}] Pas de fichier de tuning trouvé, hyperparamètres par défaut utilisés.")
    return {}


def build_operation_mapping(train):
    categories = sorted(train[CAT_COL].unique())
    mapping = {v: i for i, v in enumerate(categories)}
    unknown_code = len(categories)
    return mapping, unknown_code


def apply_operation_mapping(X, mapping, unknown_code):
    X = X.copy()
    X[CAT_COL] = X[CAT_COL].map(mapping).fillna(unknown_code).astype("int32")
    return X


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


def optimize_blend_weights(oof_preds_dict, y_true):
    """
    Cherche les poids w >= 0, somme(w) = 1 qui maximisent le PR-AUC
    d'un blend pondéré, sur les prédictions OOF temporelles.
    Beaucoup plus robuste qu'une régression logistique libre.
    """

    names = list(oof_preds_dict.keys())
    preds_matrix = np.column_stack([oof_preds_dict[n] for n in names])

    def negative_ap(w):
        w = np.clip(w, 0, None)
        if w.sum() == 0:
            return 0.0
        w = w / w.sum()
        blend = preds_matrix @ w
        return -evaluate_predictions(y_true, blend)

    n_models = len(names)
    init = np.ones(n_models) / n_models
    bounds = [(0, 1)] * n_models
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    result = minimize(
        negative_ap, init, method="SLSQP", bounds=bounds, constraints=constraints
    )

    weights = np.clip(result.x, 0, None)
    weights = weights / weights.sum()

    return dict(zip(names, weights)), -result.fun


def main():

    print("=" * 70)
    print("Chargement des données")
    print("=" * 70)

    train = load_train()
    print("Train :", train.shape)

    try:
        test = load_test()
        diagnose_account_overlap(train, test)
    except FileNotFoundError:
        test = None
        print("test.csv introuvable, diagnostic de recouvrement ignoré.")

    operation_mapping, unknown_code = build_operation_mapping(train)
    joblib.dump(
        {"mapping": operation_mapping, "unknown_code": unknown_code},
        MODEL_DIR / "operation_mapping_v2.pkl",
    )

    lgb_tuned = load_tuned_params("lightgbm")
    xgb_tuned = load_tuned_params("xgboost")
    cat_tuned = load_tuned_params("catboost")

    splits = time_series_splits(train, n_splits=N_SPLITS, min_train_frac=0.4)

    oof_lgb = np.full(len(train), np.nan)
    oof_xgb = np.full(len(train), np.nan)
    oof_cat = np.full(len(train), np.nan)

    lgb_models, xgb_models, cat_models = [], [], []

    for fold, (train_idx, valid_idx) in enumerate(splits):

        print(f"\n{'=' * 70}\nFOLD {fold} (walk-forward)\n{'=' * 70}")

        train_fold = train.iloc[train_idx].copy()
        valid_fold = train.iloc[valid_idx].copy()

        train_fold, valid_fold = build_fold_features(train_fold, valid_fold)

        X_train = train_fold.drop(columns=DROP_COLUMNS)
        y_train = train_fold["fraud_flag"]
        X_valid = valid_fold.drop(columns=DROP_COLUMNS)
        y_valid = valid_fold["fraud_flag"]

        # -------------------- LightGBM --------------------
        X_train_lgb = apply_operation_mapping(X_train, operation_mapping, unknown_code)
        X_valid_lgb = apply_operation_mapping(X_valid, operation_mapping, unknown_code)

        from lightgbm import early_stopping

        lgb_model = get_lightgbm()
        if lgb_tuned:
            lgb_model.set_params(**lgb_tuned)
        lgb_model.fit(
            X_train_lgb, y_train,
            eval_set=[(X_valid_lgb, y_valid)],
            eval_metric="average_precision",
            callbacks=[early_stopping(100, verbose=False)],
        )
        pred_lgb = lgb_model.predict_proba(X_valid_lgb)[:, 1]
        oof_lgb[valid_idx] = pred_lgb
        lgb_models.append(lgb_model)
        print(f"LightGBM  PR-AUC : {evaluate_predictions(y_valid, pred_lgb):.6f}")

        # -------------------- XGBoost --------------------
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale_pos_weight = neg / max(pos, 1)

        X_train_xgb = X_train.copy()
        X_valid_xgb = X_valid.copy()
        X_train_xgb[CAT_COL] = X_train_xgb[CAT_COL].astype("category")
        X_valid_xgb[CAT_COL] = X_valid_xgb[CAT_COL].astype("category")

        xgb_model = get_xgboost_best(scale_pos_weight=scale_pos_weight)
        xgb_model.set_params(early_stopping_rounds=100)
        if xgb_tuned:
            xgb_model.set_params(**xgb_tuned)
        xgb_model.fit(
            X_train_xgb, y_train,
            eval_set=[(X_valid_xgb, y_valid)],
            verbose=False,
        )
        pred_xgb = xgb_model.predict_proba(X_valid_xgb)[:, 1]
        oof_xgb[valid_idx] = pred_xgb
        xgb_models.append(xgb_model)
        print(f"XGBoost   PR-AUC : {evaluate_predictions(y_valid, pred_xgb):.6f}")

        # -------------------- CatBoost --------------------
        from catboost import Pool

        train_pool = Pool(X_train, y_train, cat_features=[CAT_COL])
        valid_pool = Pool(X_valid, y_valid, cat_features=[CAT_COL])

        cat_model = get_catboost()
        cat_model.set_params(
            class_weights=[1.0, scale_pos_weight],
            iterations=1500,
            early_stopping_rounds=100,
        )
        if cat_tuned:
            cat_model.set_params(**cat_tuned)
        cat_model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        pred_cat = cat_model.predict_proba(X_valid)[:, 1]
        oof_cat[valid_idx] = pred_cat
        cat_models.append(cat_model)
        print(f"CatBoost  PR-AUC : {evaluate_predictions(y_valid, pred_cat):.6f}")

    # -------------------- Résultats globaux (uniquement les lignes évaluées) --------------------
    evaluated_mask = ~np.isnan(oof_lgb)
    y_eval = train["fraud_flag"].values[evaluated_mask]

    print(f"\n{'=' * 70}\nRÉSULTATS GLOBAUX (walk-forward, {evaluated_mask.sum()} lignes évaluées)\n{'=' * 70}")
    print(f"LightGBM OOF PR-AUC : {evaluate_predictions(y_eval, oof_lgb[evaluated_mask]):.6f}")
    print(f"XGBoost  OOF PR-AUC : {evaluate_predictions(y_eval, oof_xgb[evaluated_mask]):.6f}")
    print(f"CatBoost OOF PR-AUC : {evaluate_predictions(y_eval, oof_cat[evaluated_mask]):.6f}")

    simple_blend = (oof_lgb[evaluated_mask] + oof_xgb[evaluated_mask] + oof_cat[evaluated_mask]) / 3
    print(f"Blend simple (moyenne) PR-AUC : {evaluate_predictions(y_eval, simple_blend):.6f}")

    # -------------------- Blend pondéré optimisé (remplace le stacking logreg) --------------------
    oof_dict = {
        "lgb": oof_lgb[evaluated_mask],
        "xgb": oof_xgb[evaluated_mask],
        "cat": oof_cat[evaluated_mask],
    }
    weights, best_ap = optimize_blend_weights(oof_dict, y_eval)
    print(f"Blend pondéré optimisé PR-AUC : {best_ap:.6f}")
    print(f"Poids optimaux : {weights}")

    # -------------------- Sauvegarde --------------------
    for i, m in enumerate(lgb_models):
        joblib.dump(m, MODEL_DIR / f"lgb_fold_{i}_v2.pkl")
    for i, m in enumerate(xgb_models):
        joblib.dump(m, MODEL_DIR / f"xgb_fold_{i}_v2.pkl")
    for i, m in enumerate(cat_models):
        joblib.dump(m, MODEL_DIR / f"cat_fold_{i}_v2.pkl")
    joblib.dump(weights, MODEL_DIR / "blend_weights_v2.pkl")

    print(f"\n{len(lgb_models)} modèles par famille sauvegardés dans", MODEL_DIR)
    print("NOTE : predict_stack.py doit être mis à jour pour utiliser "
          "blend_weights_v2.pkl au lieu de stack_model_v2.pkl "
          "(voir predict_stack_v2.py fourni séparément).")


if __name__ == "__main__":
    main()