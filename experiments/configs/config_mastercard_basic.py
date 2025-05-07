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
        },
        "optuna_params": lambda trial: {
            "max_depth": trial.suggest_int("max_depth", -1, 100),
        },
    }
]
