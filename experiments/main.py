import argparse
import importlib
import multiprocessing as mp
import os
from functools import partial
import traceback
from typing import Any, Callable, Dict, Tuple

import joblib
import mlflow
import optuna
import pandas as pd
import polars
import polars as pl
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from mastercard.experiment_session.data_spliter import create_session_iris, create_session_mastercard
from mastercard.experiment_template import Config


def initialize_experiment_session_data(
    experiment_session_id: str,
    data: str = "iris",
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
        if data == "iris":
            train_dataset, test_dataset = create_session_iris()
            train_dataset.to_parquet(f"{exp_data_path}/train.parquet")
            test_dataset.to_parquet(f"{exp_data_path}/test.parquet")
        elif data == "mastercard":
            train_dataset, test_dataset = create_session_mastercard()
            train_dataset.write_parquet(f"{exp_data_path}/train.parquet")
            test_dataset.write_parquet(f"{exp_data_path}/test.parquet")
        else:
            raise Exception("Undefined data for experimental session")

    else:
        print("Loading data from existing experiment session")
        if data == "iris":
            train_dataset, test_dataset = pd.read_parquet(f"{exp_data_path}/train.parquet"), pd.read_parquet(f"{exp_data_path}/test.parquet")
        elif data == "mastercard":
            train_dataset, test_dataset = pl.read_parquet(f"{exp_data_path}/train.parquet"), pl.read_parquet(f"{exp_data_path}/test.parquet")
    return train_dataset, test_dataset


def objective(
    trial: optuna.Trial,
    config: Config,
    constant_params: Dict[str, Any],
    fun_params: Callable[[optuna.Trial], Dict[str, Any]],
    train_dataset: pd.DataFrame | polars.DataFrame,
    parent_run_id: str,
):
    """
    Optuna optimization trial used for model evaluation with cross validation on training set
    to find optimal model parameters.
    """
    with mlflow.start_run(nested=True, parent_run_id=parent_run_id) as nested_run:
        cv_metrics = []
        # extracting parameters form the experiment
        params = fun_params(trial)
        model_module = importlib.import_module(f"mastercard.models.{config.model_name}")

        # spliting
        features = constant_params["numeric_features"] + constant_params["categorical_features"]
        X, y = train_dataset[features], train_dataset[config.target]

        challenger_hyperparameters = model_module.hyperparameters.Hyperparameters(**(constant_params | params))
        mlflow.log_params(dict(config))
        mlflow.log_params(dict(challenger_hyperparameters))

        n_splits = 5
        if config.kfold_strategy == "stratified":
            stratifier = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.optuna_random_state).split(X, y)
        elif config.kfold_strategy == "timeseries":
            stratifier = TimeSeriesSplit(n_splits=n_splits).split(X, y)
        else:
            raise Exception("Invalid kfold strategy")

        try:
            for _, (train_index, val_index) in enumerate(stratifier):
                if isinstance(train_dataset, pd.DataFrame):
                    train, val = train_dataset.iloc[train_index], train_dataset.iloc[val_index]
                elif isinstance(train_dataset, pl.DataFrame):
                    train, val = train_dataset[train_index, :], train_dataset[val_index, :]
                else:
                    raise Exception("Unsupported dataset type")
                    

                artifacts = model_module.train_pipe(config, challenger_hyperparameters, train)
                predict_dataset = model_module.predict_pipe(config, artifacts, val)
                metrics = model_module.evaluate.evaluate_pipe(config, val, predict_dataset)
                cv_metrics.append(metrics)

            cv_metrics = pd.DataFrame(cv_metrics).mean().to_dict()
            mlflow.log_metrics(cv_metrics)
            artifacts_path = f"artifacts/{parent_run_id}/{nested_run.info.run_id}/"
            os.makedirs(artifacts_path)
            
            for name, artifact in dict(artifacts).items():
                try:
                    joblib.dump(artifact, f"{artifacts_path}/{name}.joblib")
                except Exception:
                    print(f"Failed to save model artifact: {name}")
            
            return cv_metrics[config.optuna_main_metric]
        except Exception:
            print(traceback.format_exc())
            error = float("-inf") if config.optuna_direction == "maximize" else float("inf")
            mlflow.log_metric(config.optuna_main_metric, error)
            return error


def evaluation_loop(
    experiment_session_id: str,
    data: str,
    config: Config,
    constant_params: Dict[str, Any],
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

    train_dataset, test_dataset = initialize_experiment_session_data(experiment_session_id, data)
    with mlflow.start_run() as run:
        run = mlflow.active_run()
        mlflow.set_tag("model_name", config.model_name)
        mlflow.set_tag("experiment_session_id", experiment_session_id)

        artifacts_path = f"artifacts/{run.info.run_id}"
        os.makedirs(artifacts_path)

        study = optuna.create_study(direction=config.optuna_direction)
        study.optimize(
            partial(
                objective,
                config=config,
                constant_params=constant_params,
                fun_params=fun_params,
                train_dataset=train_dataset,
                parent_run_id=run.info.run_id,
            ),
            n_trials=config.optuna_n_trials,
            n_jobs=config.optuna_n_jobs,
        )

        mlflow.log_params(study.best_params)
        mlflow.log_metric("validation_objective", study.best_value)

        model_module = importlib.import_module(f"mastercard.models.{config.model_name}")

        print(study.best_params)
        best_hyperparameters = model_module.hyperparameters.Hyperparameters(**(constant_params | study.best_params))

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
        
        predict_dataset = model_module.predict_pipe(config, artifacts, train_dataset)
        metrics = model_module.evaluate.evaluate_pipe(config, train_dataset, predict_dataset)
        train_metrics = {f'train_{k}': v for k, v in metrics.items()}
        mlflow.log_metrics(train_metrics)
        
        print("-" * 20)
        print("Final result:")
        print(best_hyperparameters)
        print(metrics)
        print(train_metrics)


if __name__ == "__main__":
    pl.enable_string_cache()
    parser = argparse.ArgumentParser(prog="Experiment runner")
    parser.add_argument("--exp", type=str, help="Name of experiment configuration file")
    parser.add_argument("--workers", type=int, default=1, help="Name of experiment configuration file")
    args = parser.parse_args()

    configs_module = importlib.import_module(f"configs.{args.exp}")

    mlflow.set_tracking_uri(uri="http://127.0.0.1:5001")

    if args.workers == 1:
        for experiment in configs_module.experiments:
            experiment_session_id, data, config, constant_params, fun_params = (
                experiment["experiment_session_id"],
                experiment["data"],
                experiment["config"],
                experiment["constant_params"],
                experiment["optuna_params"],
            )
            evaluation_loop(experiment_session_id, data, config, constant_params, fun_params)
    else:
        with mp.Pool(processes=args.workers) as pool:
            pool.starmap(
                evaluation_loop,
                [
                    (
                        experiment["experiment_session_id"],
                        experiment["data"],
                        experiment["config"],
                        experiment["constant_params"],
                        experiment["optuna_params"],
                    )
                    for experiment in configs_module.experiments
                ],
            )
