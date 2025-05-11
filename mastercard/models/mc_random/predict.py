import pandas as pd
import polars as pl
from mastercard.experiment_template import Config
from mastercard.models.mc_random.artifacts import Artifacts

def predict_pipe(
    config: Config,
    artifacts: Artifacts,
    test_dataset: pl.DataFrame,
) -> pd.DataFrame:
    
    if isinstance(test_dataset, pd.DataFrame):
        test_dataset = pl.from_pandas(test_dataset)
        
    X_test = test_dataset[artifacts.features]
    y_hat = artifacts.model.predict(X_test)
    y_hat_proba = artifacts.model.predict_proba(X_test)

    out_df = pd.DataFrame()
    out_df["transaction_id"] = test_dataset["transaction_id"]
    out_df[f"{config.target}_pred"] = y_hat
    out_df[f"{config.target}_pred_proba"] = y_hat_proba[:, 1]

    return out_df
