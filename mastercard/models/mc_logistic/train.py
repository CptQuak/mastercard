import sklearn
import lightgbm
import polars as pl
import pandas as pd
from sklearn.frozen import FrozenEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, TimeSeriesSplit, TunedThresholdClassifierCV, train_test_split
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from mastercard.experiment_template import Config
from mastercard.models.mc_logistic.artifacts import Artifacts
from mastercard.models.mc_logistic.hyperparameters import Hyperparameters
from mastercard.data_proc.src import compute_user_time_statistics, compute_time_features, interesting_features
from skrub import tabular_learner


def train_pipe(
    config: Config,
    hyper_params: Hyperparameters,
    train_dataset: pl.DataFrame,
) -> Artifacts:
    if isinstance(train_dataset, pd.DataFrame):
        train_dataset = pl.from_pandas(train_dataset)
    train_dataset_init = train_dataset.clone()
    numeric_features, categorical_features = hyper_params.numeric_features, hyper_params.categorical_features

    user_statistics = {}
    if hyper_params.user_statistics:
        user_statistics, time_features = compute_user_time_statistics(train_dataset)
        for stats in user_statistics.values():
            train_dataset = train_dataset.join_asof(stats, on="timestamp", by="user_id")
        numeric_features = numeric_features + time_features

    if hyper_params.time_features:
        train_dataset, time_features = compute_time_features(train_dataset)
        numeric_features = numeric_features + time_features

    train_dataset, feat = interesting_features(train_dataset)
    numeric_features = numeric_features + feat

    features = numeric_features + categorical_features
    X_train, y_train = train_dataset[features], train_dataset[config.target]

    model = tabular_learner(
        LogisticRegression(
            class_weight=hyper_params.class_weight,
            random_state=hyper_params.random_state,
            penalty=hyper_params.penalty,
            C=hyper_params.C,
            solver=hyper_params.solver,
            max_iter=hyper_params.max_iter,
            l1_ratio=hyper_params.l1_ratio,
        ),
        n_jobs=2,
    )

    if hyper_params.prob_calibration:
        cv = StratifiedKFold(2, shuffle=True, random_state=13)
        model = CalibratedClassifierCV(model, method="isotonic", cv=cv, ensemble=False, n_jobs=3)
        cv = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=13)
        model = TunedThresholdClassifierCV(model, scoring="balanced_accuracy", cv=cv, refit=True)
    model.fit(X_train, y_train)

    return Artifacts(
        features=features,
        model=model,
        user_statistics=user_statistics,
        hyperparams=dict(hyper_params),
        train_dataset=train_dataset_init,
    )
