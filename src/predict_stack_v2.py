"""
src/predict_stack_v2.py

A lancer APRES src/train_stack_temporal.py :

    python -m src.predict_stack_v2

Identique à predict_stack.py, sauf que la combinaison finale des 3
modèles utilise le blend pondéré contraint (blend_weights_v2.pkl)
plutôt que la régression logistique (stack_model_v2.pkl), qui avait
des coefficients trop extrêmes et généralisait mal sur le
leaderboard.
"""

import numpy as np
import joblib

from src.config import MODEL_DIR, SUBMISSION_DIR
from src.data_loader import load_train, load_test, load_submission
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

N_SPLITS = 4
DROP_COLUMNS_TEST = ["id", "origin_account", "destination_account"]
CAT_COL = "operation"


def main():

    print("=" * 70)
    print("Chargement des données")
    print("=" * 70)

    train = load_train()
    test = load_test()

    origin_stats = build_origin_statistics(train)
    destination_stats = build_destination_statistics(train)
    pair_stats = build_pair_statistics(train)

    train = apply_origin_statistics(train, origin_stats)
    test = apply_origin_statistics(test, origin_stats)
    train = apply_destination_statistics(train, destination_stats)
    test = apply_destination_statistics(test, destination_stats)
    train = apply_pair_statistics(train, pair_stats)
    test = apply_pair_statistics(test, pair_stats)

    train = build_features_v2(train)
    test = build_features_v2(test)

    X_test = test.drop(columns=DROP_COLUMNS_TEST)

    mapping_payload = joblib.load(MODEL_DIR / "operation_mapping_v2.pkl")
    mapping = mapping_payload["mapping"]
    unknown_code = mapping_payload["unknown_code"]

    # -------------------- LightGBM (moyenne des folds) --------------------
    X_test_lgb = X_test.copy()
    X_test_lgb[CAT_COL] = (
        X_test_lgb[CAT_COL].map(mapping).fillna(unknown_code).astype("int32")
    )
    lgb_preds = []
    for i in range(N_SPLITS):
        model = joblib.load(MODEL_DIR / f"lgb_fold_{i}_v2.pkl")
        lgb_preds.append(model.predict_proba(X_test_lgb)[:, 1])
    pred_lgb = np.mean(lgb_preds, axis=0)

    # -------------------- XGBoost (moyenne des folds) --------------------
    X_test_xgb = X_test.copy()
    X_test_xgb[CAT_COL] = X_test_xgb[CAT_COL].astype("category")
    xgb_preds = []
    for i in range(N_SPLITS):
        model = joblib.load(MODEL_DIR / f"xgb_fold_{i}_v2.pkl")
        xgb_preds.append(model.predict_proba(X_test_xgb)[:, 1])
    pred_xgb = np.mean(xgb_preds, axis=0)

    # -------------------- CatBoost (moyenne des folds) --------------------
    cat_preds = []
    for i in range(N_SPLITS):
        model = joblib.load(MODEL_DIR / f"cat_fold_{i}_v2.pkl")
        cat_preds.append(model.predict_proba(X_test)[:, 1])
    pred_cat = np.mean(cat_preds, axis=0)

    # -------------------- Blend pondéré (poids optimisés en walk-forward) --------------------
    weights = joblib.load(MODEL_DIR / "blend_weights_v2.pkl")
    print("Poids utilisés :", weights)

    final_pred = (
        weights["lgb"] * pred_lgb
        + weights["xgb"] * pred_xgb
        + weights["cat"] * pred_cat
    )

    # -------------------- Soumission --------------------
    try:
        submission = load_submission()
        submission["target"] = final_pred
    except FileNotFoundError:
        print("sample_submission.csv introuvable, construction de la "
              "soumission directement depuis test.csv (colonnes id, target).")
        submission = test[["id"]].copy()
        submission["target"] = final_pred

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSION_DIR / "submission_blend_weighted.csv"
    submission.to_csv(out_path, index=False)

    print("=" * 70)
    print(f"Soumission (blend pondéré) créée : {out_path}")
    print("=" * 70)
    print(submission.head())


if __name__ == "__main__":
    main()
