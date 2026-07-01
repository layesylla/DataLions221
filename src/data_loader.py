import pandas as pd

from src.config import RAW_DATA


def load_train():

    return pd.read_csv(
        RAW_DATA / "train.csv"
    )


def load_test():

    return pd.read_csv(
        RAW_DATA / "test.csv"
    )


def load_submission():

    return pd.read_csv(
        RAW_DATA / "sample_submission.csv"
    )