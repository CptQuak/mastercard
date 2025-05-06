import pandas as pd
from mastercard.experiment_template import Config
from mastercard.models.model_0.artifacts import Artifacts


def predict_pipe(
    config: Config,
    artifacts: Artifacts,
    predict_dataset: pd.DataFrame,
):
    return pd.DataFrame(
        artifacts.model.predict(predict_dataset[config.columns]),
        columns=[f"{config.target}_pred"],
    )
