import pandas as pd
from mastercard.experiment_template import Config
from mastercard.models.model_0.artifacts import Artifacts


def predict_pipe(
    config: Config,
    artifacts: Artifacts,
    test_dataset: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        artifacts.model.predict(test_dataset[artifacts.features]),
        columns=[f"{config.target}_pred"],
    )
