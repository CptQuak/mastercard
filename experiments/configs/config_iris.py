from mastercard.experiment_template import Config


experiments = [
    {
        "experiment_session_id": "iris_test",
        "config": Config(
            model_name="model_0",
            target="target",
            columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
        ),
        "optuna_params": lambda trial: {
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2", None]),
            "C": trial.suggest_float("C", 1e-5, 1e3, log=True),
        },
    }
]
