import numpy as np
import polars as pl


def compute_user_time_statistics(df_train: pl.DataFrame):
    time_features = ["amount_mean", "amount_max", "number_of_transactions"]
    time_ranges = [
        ("1h", "hourly"),  # (1 hour)
        ("1d", "daily"),  # (1 calendar day)
        ("1w", "weekly"),  # (1 calendar week)
        ("1mo", "monthly"),  # (1 calendar month)
        ("1q", "quarterly"),  # (1 calendar quarter)
        ("1y", "yearly"),  # (1 calendar year)
    ]

    all_time_features = [f"{time}_{naming}" for time in time_features for _, naming in time_ranges]

    df_train = df_train.sort("timestamp")
    user_statistics = {}

    for time_range, name in time_ranges:
        user_statistics[name] = (
            df_train.group_by_dynamic(
                index_column="timestamp",
                group_by=["user_id"],
                every=time_range,
            )
            .agg(
                pl.col("amount").mean().alias(f"amount_mean_{name}"),
                pl.col("amount").max().alias(f"amount_max_{name}"),
                (pl.col("is_fraud").sum() / pl.col("is_fraud").count()).alias(f"pcnt_frauds_{name}"),
                pl.col("amount").count().alias(f"number_of_transactions_{name}"),
            )
            .sort("timestamp")
        )

    all_time_features += ["distinct_locations_daily", "transaction_count_past_24h"]
    user_statistics["locations"] = df_train.group_by_dynamic(
        "timestamp",
        every="1d",
        group_by="user_id",
    ).agg(
        pl.concat_str(
            [pl.col("lat"), pl.col("long")],
            separator=", ",
        )
        .count()
        .alias("distinct_locations_daily")
    )

    user_statistics["dynamic_counts"] = df_train.group_by_dynamic(
        index_column="timestamp",
        every="1d",
        period="24h",
        group_by="user_id",
    ).agg(pl.len().alias("transaction_count_past_24h"))
    return user_statistics, all_time_features


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

def interesting_features(df: pl.DataFrame):
    return df.with_columns(
        (pl.col('country_user') == pl.col('country_merchant')).alias('same_country'),
        (pl.col('amount') == pl.col('amount').cast(pl.Int64())).alias('integer_amount'),
    ), ['same_country', 'integer_amount']