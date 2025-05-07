import lightgbm
import pandas as pd
from sklearn.compose import make_column_transformer
from mastercard.experiment_template import Config
from mastercard.models.mc_lgbm_basic.artifacts import Artifacts
from mastercard.models.mc_lgbm_basic.hyperparameters import Hyperparameters


def train_pipe(
    config: Config,
    hyper_params: Hyperparameters,
    train_dataset: pd.DataFrame,
) -> Artifacts:
    features = hyper_params.numeric_features + hyper_params.categorical_features
    X_train, y_train = train_dataset[features], train_dataset[config.target]

    transformer = make_column_transformer(
        # (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), categorical_features),
        ("passthrough", features),
        remainder="passthrough",
    ).set_output(transform="polars")
    model = lightgbm.LGBMClassifier(
        class_weight=hyper_params.class_weight,
        random_state=13,
        max_depth=hyper_params.max_depth,
    )

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
    )
