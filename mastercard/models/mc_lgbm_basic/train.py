import lightgbm
import polars as pl
import pandas as pd
from sklearn.compose import make_column_transformer
from mastercard.experiment_template import Config
from mastercard.models.mc_lgbm_basic.artifacts import Artifacts
from mastercard.models.mc_lgbm_basic.hyperparameters import Hyperparameters


def train_pipe(
    config: Config,
    hyper_params: Hyperparameters,
    train_dataset: pl.DataFrame,
) -> Artifacts:
    if isinstance(train_dataset, pd.DataFrame):
        train_dataset = pl.from_pandas(train_dataset)

    numeric_features, categorical_features = hyper_params.numeric_features, hyper_params.categorical_features

    quarterly_statistics = None
    if hyper_params.quarterly_statistics:
        time_features = ["amount_quarter_mean", "amount_quarter_max"]
        quarterly_statistics = (
            train_dataset.group_by_dynamic(index_column="timestamp", group_by=["user_id"], every="1q")
            .agg(
                pl.col("amount").mean().alias("amount_quarter_mean"),
                pl.col("amount").max().alias("amount_quarter_max"),
            )
            .sort("timestamp")
        )
        train_dataset = train_dataset.join_asof(quarterly_statistics, on="timestamp", by="user_id")
        numeric_features = numeric_features + time_features

    features = numeric_features + categorical_features
    X_train, y_train = train_dataset[features], train_dataset[config.target]

    transformer = make_column_transformer(
        # (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), categorical_features),
        ("passthrough", features),
        remainder="passthrough",
    ).set_output(transform="polars")
    print(hyper_params)
    print(dict(hyper_params))
    model = lightgbm.LGBMClassifier(random_state=13, **dict(hyper_params))

    X_train = transformer.fit_transform(X_train).to_pandas()
    categorical_features_trns = [i for i in X_train.columns if i.split("__")[1] in hyper_params.categorical_features]
    model.fit(
        X_train,
        y_train,
        eval_metric=hyper_params.eval_metric,
        categorical_feature=categorical_features_trns,
    )

    return Artifacts(
        features=features,
        model=model,
        transformer=transformer,
        quarterly_statistics=quarterly_statistics,
        hyperparams=dict(hyper_params),
    )
