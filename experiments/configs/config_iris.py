from mastercard.experiment_template import Config


experiments = [
    {
        "experiment_session_id": "iris_test",
        "data": "iris",
        "config": Config(
            model_name="model_0",
            target="target",
        ),
        "constant_params": {
            "numeric_features":["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
            "categorical_features": [],
        },
        "optuna_params": lambda trial: {
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2", None]),
            "C": trial.suggest_float("C", 1e-5, 1e3, log=True),
        },
    }
]
