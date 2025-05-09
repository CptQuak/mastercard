import numpy as np
import polars as pl


def compute_quarterly_statistics(df_train):
    time_features = ["amount_quarter_mean", "amount_quarter_max", 'pcnt_frauds']
    quarterly_statistics = (
        df_train.group_by_dynamic(
            index_column="timestamp",
            group_by=["user_id"],
            every="1q",
        )
        .agg(
            pl.col("amount").mean().alias("amount_quarter_mean"),
            pl.col("amount").max().alias("amount_quarter_max"),
            (pl.col("is_fraud").sum() / pl.col("is_fraud").len()).alias("pcnt_frauds"),
        )
        .sort("timestamp")
    )
    return quarterly_statistics, time_features


def compute_time_features(df):
    time_features = [
        "year",
        "month",
        "day_of_month",
        "hour_of_day",
        "day_of_week",
        "day_of_year",
        "week_of_year",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "day_of_week_cos",
        "month_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "is_business_hours",
    ]
    df_with_features = df.with_columns(
        [
            # standard date features
            pl.col("timestamp").dt.year().alias("year"),
            pl.col("timestamp").dt.month().alias("month"),
            pl.col("timestamp").dt.day().alias("day_of_month"),
            pl.col("timestamp").dt.hour().alias("hour_of_day"),
            pl.col("timestamp").dt.weekday().alias("day_of_week"),
            pl.col("timestamp").dt.ordinal_day().alias("day_of_year"),
            pl.col("timestamp").dt.week().alias("week_of_year"),
            # weekend
            pl.col("timestamp").dt.weekday().is_in([6, 7]).alias("is_weekend"),
            # cyclical features
            (pl.col("timestamp").dt.hour() * (2 * np.pi / 24)).sin().alias("hour_sin"),
            (pl.col("timestamp").dt.hour() * (2 * np.pi / 24)).cos().alias("hour_cos"),
            ((pl.col("timestamp").dt.weekday() - 1) * (2 * np.pi / 7)).cos().alias("day_of_week_cos"),
            ((pl.col("timestamp").dt.month() - 1) * (2 * np.pi / 12)).cos().alias("month_cos"),
            ((pl.col("timestamp").dt.ordinal_day() - 1) * (2 * np.pi / 365.25)).sin().alias("day_of_year_sin"),
            ((pl.col("timestamp").dt.ordinal_day() - 1) * (2 * np.pi / 365.25)).cos().alias("day_of_year_cos"),
            # business hours
            (
                (
                    (pl.col("timestamp").dt.weekday().is_in(list(range(1, 6))))
                    & (pl.col("timestamp").dt.hour() >= 9)
                    & (pl.col("timestamp").dt.hour() < 17)
                ).alias("is_business_hours")
            ),
        ]
    )
    return df_with_features, time_features
