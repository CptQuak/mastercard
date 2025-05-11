from mastercard.experiment_template import Config


experiments = [
    {
        "experiment_session_id": "master_card_logistic",
        "data": "mastercard",
        "config": Config(
            model_name="mc_logistic",
            target="is_fraud",
            optuna_n_trials=150,
            optuna_n_jobs=6,
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
                # "user_id",
                # "merchant_id",
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
        },
        "optuna_params": lambda trial: {
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"]),
            "C": trial.suggest_float("C", 1e-6, 1, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 1000),
            "l1_ratio": trial.suggest_float("l1_ratio", 0, 1),
        },
    }
]
