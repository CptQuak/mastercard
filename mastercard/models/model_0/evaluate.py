import pandas as pd
from sklearn.metrics import accuracy_score

from mastercard.experiment_template import Config


def evaluate_pipe(
    config: Config,
    test_dataset: pd.DataFrame,
    predict_dataset: pd.DataFrame,
):
    return {
        "accuracy": accuracy_score(
            test_dataset[config.target],
            predict_dataset[f"{config.target}_pred"],
        ),
    }
