import sklearn
import lightgbm
import polars as pl
import pandas as pd
from sklearn.frozen import FrozenEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, TimeSeriesSplit, TunedThresholdClassifierCV, train_test_split
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from mastercard.experiment_template import Config
from mastercard.models.mc_lgbm_basic.artifacts import Artifacts
from mastercard.models.mc_lgbm_basic.hyperparameters import Hyperparameters
from mastercard.models.mc_lgbm_basic.src import compute_user_time_statistics, compute_time_features, interesting_features


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

    transformer = make_column_transformer(
        # (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), categorical_features),
        ("passthrough", features),
        remainder="passthrough",
    ).set_output(transform="polars")

    X_train = transformer.fit_transform(X_train).to_pandas()
    categorical_features_trns = [i for i in X_train.columns if i.split("__")[1] in hyper_params.categorical_features]
    model = lightgbm.LGBMClassifier(
        random_state=13,
        # n_estimators=1,
        # max_depth=10,
        **dict(hyper_params),
    )
    # model.fit(
    #     X_train,
    #     y_train,
    #     eval_metric=hyper_params.eval_metric,
    #     categorical_feature=categorical_features_trns,
    # )

    # cv = TimeSeriesSplit(2)
    model = Pipeline([("lgbm", model)])
    if hyper_params.prob_calibration:
        cv = StratifiedKFold(2, shuffle=True, random_state=13)
        model = CalibratedClassifierCV(model, method="isotonic", cv=cv, ensemble=False, n_jobs=3)
        cv = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=13)
        model = TunedThresholdClassifierCV(model, scoring="balanced_accuracy", cv=cv, refit=True)
    model.fit(X_train, y_train)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=13, stratify=y_train)
    # .fit(X_train, y_train)

    return Artifacts(
        features=features,
        model=model,
        transformer=transformer,
        user_statistics=user_statistics,
        hyperparams=dict(hyper_params),
        train_dataset=train_dataset_init,
    )
