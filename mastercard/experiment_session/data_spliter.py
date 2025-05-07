from typing import Tuple
from flask.config import T
import pandas as pd
from sklearn.datasets import load_iris
import polars as pl


def create_session_iris(random_seed=13) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = load_iris(as_frame=True)["frame"]

    train_dataset, test_dataset = [], []
    for value in df["target"].unique():
        group = df[df["target"] == value].copy()

        test = group.sample(n=int(len(group) * 0.3), random_state=random_seed)
        train = group.drop(test.index)

        train_dataset.append(train), test_dataset.append(test)

    train_dataset, test_dataset = (
        pd.concat(train_dataset, ignore_index=True),
        pd.concat(test_dataset, ignore_index=True),
    )
    return train_dataset, test_dataset


def create_session_mastercard(data_path="../datasets", random_seed=13) -> Tuple[pl.DataFrame, pl.DataFrame]:
    df_merchants = pl.read_csv(f"{data_path}/merchants.csv").cast({i: pl.Categorical for i in ["merchant_id", "category", "country"]})
    df_merchants = df_merchants.rename({i: f"{i}_merchant" for i in df_merchants.columns[1:]})

    cat_columns = {i: pl.Categorical for i in ["user_id", "education", "sex", "primary_source_of_income", "country"]}
    df_users = pl.read_csv(f"{data_path}/users.csv", try_parse_dates=True).cast(cat_columns)
    df_users = df_users.rename({i: f"{i}_user" for i in df_users.columns[1:]})

    cat_columns = {i: pl.Categorical for i in ["user_id", "merchant_id", "channel", "currency", "device", "payment_method"]}
    date_columns = {i: pl.Datetime for i in ["timestamp"]}
    df_transactions = pl.read_ndjson(f"{data_path}/transactions.json").cast(cat_columns | date_columns).unnest("location")

    df_full = df_transactions.join(df_users, on="user_id", how="left").join(df_merchants, on="merchant_id", how="left").sort("timestamp")

    train_dataset = df_full.filter(pl.col("timestamp") < pl.datetime(2023, 9, 1))
    test_dataset = df_full.filter(pl.col("timestamp") >= pl.datetime(2023, 9, 1))

    return train_dataset, test_dataset
