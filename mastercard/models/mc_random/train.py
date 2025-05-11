import pandas as pd
import polars as pl
import sklearn
from sklearn.dummy import DummyClassifier
from mastercard.experiment_template import Config
from mastercard.models.mc_random.artifacts import Artifacts
from mastercard.models.mc_random.hyperparameters import Hyperparameters


def train_pipe(
    config: Config,
    hyper_params: Hyperparameters,
    train_dataset: pl.DataFrame,
) -> Artifacts:
    if isinstance(train_dataset, pd.DataFrame):
        train_dataset = pl.from_pandas(train_dataset)

    numeric_features, categorical_features = hyper_params.numeric_features, hyper_params.categorical_features

    features = numeric_features + categorical_features
    X_train, y_train = train_dataset[features], train_dataset[config.target]

    model = DummyClassifier(strategy="stratified").fit(X_train, y_train)

    return Artifacts(features=features, model=model)
