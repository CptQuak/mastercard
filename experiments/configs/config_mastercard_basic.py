from mastercard.experiment_template import Config


experiments = [
    {
        "experiment_session_id": "master_card_basic",
        "data": "mastercard",
        "config": Config(
            model_name="mc_lgbm_basic",
            target="is_fraud",
            optuna_n_trials=10,
            optuna_main_metric='auc',
            kfold_strategy='timeseries',
        ),
        "constant_params": {
            "numeric_features": [
                "amount",
                "session_length_seconds",
                "age_user",
                "sum_of_monthly_installments_user",
                "sum_of_monthly_expenses_user",
                "risk_score_user",
                "trust_score_merchant",
                "number_of_alerts_last_6_months_merchant",
                "avg_transaction_amount_merchant",
                "account_age_months_merchant",
                "has_fraud_history_merchant",
            ],
            "categorical_features": [
                "user_id",
                "merchant_id",
                "channel",
                "currency",
                "device",
                "payment_method",
                "is_international",
                "sex_user",
                "education_user",
                "primary_source_of_income_user",
                "country_user",
                "category_merchant",
                "country_merchant",
            ],
            "class_weight": "balanced",
            "eval_metric": "auc",
            "n_jobs": 2,
        },
        "optuna_params": lambda trial: {
            # feature based
            "user_statistics": trial.suggest_categorical("user_statistics", [True]),
            "time_features": trial.suggest_categorical("time_features", [True]),
            # model based
            "max_depth": trial.suggest_int("max_depth", 5, 100),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1e-1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1e-1, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.05, 0.5),
        },
    }
]
