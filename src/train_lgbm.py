import joblib

from sklearn.metrics import average_precision_score
from lightgbm import early_stopping

from src.config import MODEL_DIR
from src.data_loader import load_train
from src.validation import split_data

from src.aggregation import (
    build_origin_statistics,
    build_destination_statistics,
    apply_origin_statistics,
    apply_destination_statistics
)

from src.feature_engineering import build_features

from src.models import get_lightgbm


# ==========================================================
# DATA
# ==========================================================

train = load_train()

train_df, valid_df = split_data(train)

origin_stats = build_origin_statistics(train_df)
destination_stats = build_destination_statistics(train_df)

train_df = apply_origin_statistics(train_df, origin_stats)
valid_df = apply_origin_statistics(valid_df, origin_stats)

train_df = apply_destination_statistics(train_df, destination_stats)
valid_df = apply_destination_statistics(valid_df, destination_stats)

train_df = build_features(train_df)
valid_df = build_features(valid_df)

drop_columns = [
    "id",
    "fraud_flag",
    "origin_account",
    "destination_account"
]

X_train = train_df.drop(columns=drop_columns)
y_train = train_df["fraud_flag"]

X_valid = valid_df.drop(columns=drop_columns)
y_valid = valid_df["fraud_flag"]


# ==========================================================
# OPERATION ENCODING
# ==========================================================

categories = sorted(
    set(X_train["operation"]).union(
        set(X_valid["operation"])
    )
)

mapping = {
    value: idx
    for idx, value in enumerate(categories)
}

X_train["operation"] = X_train["operation"].map(mapping).astype("int32")
X_valid["operation"] = X_valid["operation"].map(mapping).astype("int32")


# ==========================================================
# MODEL
# ==========================================================

model = get_lightgbm()

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="average_precision",
    callbacks=[
        early_stopping(
            100,
            verbose=False
        )
    ]
)

pred = model.predict_proba(X_valid)[:, 1]

ap = average_precision_score(
    y_valid,
    pred
)

print("=" * 70)
print(f"Average Precision : {ap:.6f}")
print("=" * 70)

joblib.dump(
    model,
    MODEL_DIR / "lgb_best.pkl"
)