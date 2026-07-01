from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score


def split_data(df, test_size=0.2, random_state=42):

    train_df, valid_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["fraud_flag"]
    )

    return train_df, valid_df


def evaluate_predictions(y_true, y_pred):

    return average_precision_score(
        y_true,
        y_pred
    )