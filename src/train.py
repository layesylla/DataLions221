import pandas as pd

from catboost import Pool

from src.data_loader import load_train
from src.feature_engineering import build_features
from src.validation import split_data
from src.validation import evaluate_predictions
from src.models import get_catboost

from src.aggregation import (
    build_origin_statistics,
    build_destination_statistics,
    apply_origin_statistics,
    apply_destination_statistics
)

print("=" * 60)
print("Chargement des données")
print("=" * 60)

train = load_train()

print(train.shape)


print("\nCréation des features...")


train_df, valid_df = split_data(train)

origin_stats = build_origin_statistics(train_df)

destination_stats = build_destination_statistics(train_df)

train_df = apply_origin_statistics(
    train_df,
    origin_stats
)

valid_df = apply_origin_statistics(
    valid_df,
    origin_stats
)

train_df = apply_destination_statistics(
    train_df,
    destination_stats
)

valid_df = apply_destination_statistics(
    valid_df,
    destination_stats
)

train_df = build_features(train_df)

valid_df = build_features(valid_df)

print(f"\nTrain : {train_df.shape}")
print(f"Validation : {valid_df.shape}")


DROP_COLUMNS = [

    "id",

    "fraud_flag",

    "origin_account",

    "destination_account"

]


X_train = train_df.drop(columns=DROP_COLUMNS)
y_train = train_df["fraud_flag"]

X_valid = valid_df.drop(columns=DROP_COLUMNS)
y_valid = valid_df["fraud_flag"]


cat_features = [

    "operation"

]


train_pool = Pool(

    X_train,

    y_train,

    cat_features=cat_features

)


valid_pool = Pool(

    X_valid,

    y_valid,

    cat_features=cat_features

)


model = get_catboost()

model.fit(

    train_pool,

    eval_set=valid_pool,

    use_best_model=True

)


pred = model.predict_proba(X_valid)[:, 1]

score = evaluate_predictions(

    y_valid,

    pred

)

print("\n")
print("=" * 60)
print(f"Average Precision : {score:.6f}")
print("=" * 60)


importance = (

    pd.DataFrame({

        "feature": X_train.columns,

        "importance": model.get_feature_importance()

    })

    .sort_values(

        by="importance",

        ascending=False

    )

)

print("\nTop Features\n")

print(importance.head(20))