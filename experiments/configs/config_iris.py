from mastercard.experiment_template import Config

experiment_session_id = "iris_test"

params = [
    Config(
        model_name='model_0',
        target="target",
        columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
    )
]
