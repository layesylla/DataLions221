from src.data_loader import (
    load_train,
    load_test,
    load_submission
)

from src.aggregation import (
    build_origin_statistics,
    build_destination_statistics,
    apply_origin_statistics,
    apply_destination_statistics
)

from src.feature_engineering import build_features

from src.models import get_lightgbm

from src.config import SUBMISSION_DIR


# ==========================================================
# DATA
# ==========================================================

train = load_train()
test = load_test()


# ==========================================================
# AGGREGATIONS
# ==========================================================

origin_stats = build_origin_statistics(train)
destination_stats = build_destination_statistics(train)

train = apply_origin_statistics(train, origin_stats)
test = apply_origin_statistics(test, origin_stats)

train = apply_destination_statistics(train, destination_stats)
test = apply_destination_statistics(test, destination_stats)


# ==========================================================
# FEATURES
# ==========================================================

train = build_features(train)
test = build_features(test)


# ==========================================================
# DATASET
# ==========================================================

drop_columns = [
    "id",
    "fraud_flag",
    "origin_account",
    "destination_account"
]

X_train = train.drop(columns=drop_columns)
y_train = train["fraud_flag"]

X_test = test.drop(
    columns=[
        "id",
        "origin_account",
        "destination_account"
    ]
)


# ==========================================================
# ENCODAGE
# ==========================================================

categories = sorted(
    set(X_train["operation"]).union(
        set(X_test["operation"])
    )
)

mapping = {
    value: idx
    for idx, value in enumerate(categories)
}

X_train["operation"] = (
    X_train["operation"]
    .map(mapping)
    .astype("int32")
)

X_test["operation"] = (
    X_test["operation"]
    .map(mapping)
    .astype("int32")
)


# ==========================================================
# TRAIN FULL DATA
# ==========================================================

model = get_lightgbm()

model.fit(
    X_train,
    y_train
)


# ==========================================================
# PREDICTION
# ==========================================================

pred = model.predict_proba(X_test)[:, 1]


# ==========================================================
# SUBMISSION
# ==========================================================

submission = load_submission()

submission["target"] = pred

SUBMISSION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

submission.to_csv(
    SUBMISSION_DIR / "submission.csv",
    index=False
)

print("=" * 70)
print("Submission créée avec succès.")
print("=" * 70)
print(submission.head())