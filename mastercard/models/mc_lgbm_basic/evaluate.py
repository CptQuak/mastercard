from typing import Dict
import pandas as pd
import polars as pl
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from mastercard.experiment_template import Config


def evaluate_pipe(
    config: Config,
    test_dataset: pl.DataFrame,
    predict_dataset: pd.DataFrame,
) -> Dict[str, float]:
    y_test = test_dataset[config.target]
    y_hat = predict_dataset[f"{config.target}_pred"]
    y_hat_proba = predict_dataset[f"{config.target}_pred_proba"]

    return {
        "accuracy": accuracy_score(y_test, y_hat),
        "auc": roc_auc_score(y_test, y_hat),
        "f1_score": f1_score(y_test, y_hat),
        'average_precision': average_precision_score(y_test, y_hat),
    }
