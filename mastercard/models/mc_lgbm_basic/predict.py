import pandas as pd
import polars as pl
from mastercard.experiment_template import Config
from mastercard.models.mc_lgbm_basic.artifacts import Artifacts


def predict_pipe(
    config: Config,
    artifacts: Artifacts,
    test_dataset: pl.DataFrame,
) -> pd.DataFrame:
    X_test = test_dataset[artifacts.features]
    X_test = artifacts.transformer.transform(X_test).to_pandas()
    y_hat = artifacts.model.predict(X_test)

    out_df = pd.DataFrame()
    out_df["transaction_id"] = test_dataset["transaction_id"]
    out_df[f"{config.target}_pred"] = y_hat

    return out_df
