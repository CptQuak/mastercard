import argparse
import importlib
import multiprocessing as mp
import os
from functools import partial
from typing import Any, Callable, Dict, Tuple

import joblib
import mlflow
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from mastercard.experiment_session.data_spliter import create_session
from mastercard.experiment_template import Config


def initialize_experiment_session_data(
    experiment_session_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Initializes and loads experiment session data.

    This function handles the creation or loading of experiment session data.
    If a session for the given experiment name doesn't exist, it creates a new
    session, generates train and test datasets using `create_session()`, and
    saves them as parquet files. If a session already exists, it loads the
    train and test datasets from the corresponding parquet files.

    Args:
        experiment_session_id (str): The name of the experiment session.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the train and test datasets as pandas DataFrames.
    """
    exp_data_path = os.path.join("datasets", experiment_session_id)
    if not os.path.exists(exp_data_path):
        print("Creating new experiment session")
        os.makedirs(exp_data_path)
        train_dataset, test_dataset = create_session()
        train_dataset.to_parquet(f"{exp_data_path}/train.parquet")
        test_dataset.to_parquet(f"{exp_data_path}/test.parquet")

    else:
        print("Loading data from existing experiment session")
        train_dataset, test_dataset = pd.read_parquet(f"{exp_data_path}/train.parquet"), pd.read_parquet(f"{exp_data_path}/test.parquet")
    return train_dataset, test_dataset


def objective(
    trial: optuna.Trial,
    config: Config,
    fun_params: Callable[[optuna.Trial], Dict[str, Any]],
    train_dataset: pd.DataFrame,
):
    """
    Optuna optimization trial used for model evaluation with cross validation on training set
    to find optimal model parameters.
    """
    with mlflow.start_run(nested=True):
        cv_metrics = []
        # extracting parameters form the experiment
        params = fun_params(trial)
        model_module = importlib.import_module(f"mastercard.models.{config.model_name}")
        
        # spliting 
        X, y = train_dataset[config.columns], train_dataset[config.target]

        challenger_hyperparameters = model_module.hyperparameters.Hyperparameters(**params)
        mlflow.log_params(dict(config))
        mlflow.log_params(dict(challenger_hyperparameters))

        try:
            for _, (train_index, val_index) in enumerate(
                StratifiedKFold(n_splits=5, shuffle=True, random_state=config.optuna_random_state).split(X, y)
            ):
                train, val = train_dataset.iloc[train_index], train_dataset.iloc[val_index]

                artifacts = model_module.train_pipe(config, challenger_hyperparameters, train)
                predict_dataset = model_module.predict_pipe(config, artifacts, val)
                metrics = model_module.evaluate.evaluate_pipe(config, val, predict_dataset)
                cv_metrics.append(metrics)

            cv_metrics = pd.DataFrame(cv_metrics).mean().to_dict()
            mlflow.log_metrics(cv_metrics)
            return cv_metrics[config.optuna_main_metric]
        except Exception:
            error = float("-inf") if config.optuna_direction == "maximize" else float("inf")
            mlflow.log_metric(config.optuna_main_metric, error)
            return error


def evaluation_loop(
    experiment_session_id: str,
    config: Config,
    fun_params: Callable[[optuna.Trial], Dict[str, Any]],
):
    """
    Main experiment model evaluation loop. 
    1. Initalizes experiment session
    2. Creates or loads dataset
    3. Runs optuna experiments to find optimal parameters
    4. Creates and saves model with optimal parameters
    """
    mlflow.set_experiment(experiment_session_id)

    train_dataset, test_dataset = initialize_experiment_session_data(experiment_session_id)
    with mlflow.start_run(nested=True) as run:
        run = mlflow.active_run()
        artifacts_path = f"artifacts/{run.info.run_id}"
        os.makedirs(artifacts_path)

        study = optuna.create_study(direction=config.optuna_direction)
        study.optimize(
            partial(
                objective,
                config=config,
                fun_params=fun_params,
                train_dataset=train_dataset,
            ),
            n_trials=config.optuna_n_trials,
            n_jobs=config.optuna_n_jobs,
        )

        mlflow.log_params(study.best_params)
        mlflow.log_metric("validation_objective", study.best_value)

        model_module = importlib.import_module(f"mastercard.models.{config.model_name}")

        best_hyperparameters = model_module.hyperparameters.Hyperparameters(**study.best_params)

        artifacts = model_module.train_pipe(config, best_hyperparameters, train_dataset)
        predict_dataset = model_module.predict_pipe(config, artifacts, test_dataset)
        metrics = model_module.evaluate.evaluate_pipe(config, test_dataset, predict_dataset)

        for name, artifact in dict(artifacts).items():
            try:
                joblib.dump(artifact, f"{artifacts_path}/{name}.joblib")
            except Exception:
                print(f"Failed to save model artifact: {name}")

        mlflow.log_params(dict(config))
        mlflow.log_params(dict(best_hyperparameters))
        mlflow.log_metrics(metrics)
        print("-" * 20)
        print("Final result:")
        print(best_hyperparameters)
        print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Experiment runner")
    parser.add_argument("--exp", type=str, help="Name of experiment configuration file")
    parser.add_argument("--workers", type=int, default=4, help="Name of experiment configuration file")
    args = parser.parse_args()

    configs_module = importlib.import_module(f"configs.{args.exp}")

    mlflow.set_tracking_uri(uri="http://127.0.0.1:5002")

    if args.workers == 1:
        for experiment in configs_module.experiments:
            experiment_session_id, config, fun_params = experiment["experiment_session_id"], experiment["config"], experiment["optuna_params"]
            evaluation_loop(experiment_session_id, config, fun_params)
    else:
        with mp.Pool(processes=args.workers) as pool:
            pool.starmap(
                evaluation_loop,
                [
                    (
                        experiment["experiment_session_id"],
                        experiment["config"],
                        experiment["optuna_params"],
                    )
                    for experiment in configs_module.experiments
                ],
            )
