from mastercard.experiment_template import Config


experiments = [
    {
        "experiment_session_id": "mastercard_random",
        "data": "mastercard",
        "config": Config(
            model_name="mc_random",
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
            ],
            "categorical_features": [
                "channel",
            ],
        },
        "optuna_params": lambda trial: {
        },
    }
]
