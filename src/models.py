from catboost import CatBoostClassifier

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier
)


# ==========================================================
# CatBoost
# ==========================================================

def get_catboost():

    return CatBoostClassifier(

        iterations=500,

        learning_rate=0.03,

        depth=6,

        loss_function="Logloss",

        eval_metric="PRAUC",

        random_seed=42,

        verbose=False
    )


# ==========================================================
# LightGBM
# ==========================================================

from lightgbm import LGBMClassifier

# ==========================================================
# LightGBM
# ==========================================================

def get_lightgbm():

    return LGBMClassifier(

        n_estimators=2000,

        objective="binary",

        random_state=42,

        n_jobs=-1,

        verbosity=-1,

        num_leaves=255,

        max_depth=-1,

        learning_rate=0.01,

        feature_fraction=0.8,

        bagging_fraction=0.9,

        bagging_freq=5,

        min_child_samples=40

    )




# ==========================================================
# XGBoost
# ==========================================================

from xgboost import XGBClassifier

def get_xgboost_best(scale_pos_weight=1.0):

    return XGBClassifier(

        n_estimators=1500,

        max_depth=10,

        learning_rate=0.02,

        subsample=0.9,

        colsample_bytree=0.9,

        min_child_weight=5,

        gamma=0,

        reg_alpha=0,

        reg_lambda=1,

        scale_pos_weight=scale_pos_weight,

        objective="binary:logistic",

        eval_metric="logloss",

        tree_method="hist",

        random_state=42,

        n_jobs=-1

    )


# ==========================================================
# Extra Trees
# ==========================================================

def get_extra_trees():

    return ExtraTreesClassifier(

        n_estimators=500,

        random_state=42,

        n_jobs=-1

    )


# ==========================================================
# Random Forest
# ==========================================================

def get_random_forest():

    return RandomForestClassifier(

        n_estimators=500,

        random_state=42,

        n_jobs=-1

    )


# ==========================================================
# Hist Gradient Boosting
# ==========================================================

def get_hist_gradient():

    return HistGradientBoostingClassifier(

        learning_rate=0.03,

        max_depth=6,

        random_state=42

    )


