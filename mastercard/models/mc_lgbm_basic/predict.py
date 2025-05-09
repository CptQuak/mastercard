import pandas as pd
import polars as pl
from mastercard.experiment_template import Config
from mastercard.models.mc_lgbm_basic.artifacts import Artifacts
from mastercard.models.mc_lgbm_basic.src import compute_time_features


def predict_pipe(
    config: Config,
    artifacts: Artifacts,
    test_dataset: pl.DataFrame,
) -> pd.DataFrame:
    if isinstance(test_dataset, pd.DataFrame):
        test_dataset = pl.from_pandas(test_dataset)
        
    if artifacts.hyperparams["quarterly_statistics"]:
        test_dataset = test_dataset.join_asof(artifacts.quarterly_statistics, on="timestamp", by="user_id")
        
    if artifacts.hyperparams["time_features"]:
        test_dataset, _ = compute_time_features(test_dataset)
    
    X_test = test_dataset[artifacts.features]
    X_test = artifacts.transformer.transform(X_test).to_pandas()
    y_hat = artifacts.model.predict(X_test)

    out_df = pd.DataFrame()
    out_df["transaction_id"] = test_dataset["transaction_id"]
    out_df[f"{config.target}_pred"] = y_hat

    return out_df
