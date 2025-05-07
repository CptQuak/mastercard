import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from mastercard.experiment_template import Config
from mastercard.models.model_0.artifacts import Artifacts
from mastercard.models.model_0.hyperparameters import Hyperparameters


def train_pipe(
    config: Config,
    hyper_params: Hyperparameters,
    dataset: pd.DataFrame,
) -> Artifacts:
    features = hyper_params.numeric_features + hyper_params.categorical_features
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty=hyper_params.penalty,
            C=hyper_params.C,
        ),
    )
    model.fit(dataset[features], dataset[config.target])

    return Artifacts(model=model, features=features)
