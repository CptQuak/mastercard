import argparse
import importlib
import os
import joblib
import mlflow
import mlflow.artifacts
import mlflow.sklearn
import pandas as pd

from mastercard.experiment_session.data_spliter import create_session
from mastercard.experiment_template import Config
import multiprocessing as mp


def initialize_experiment_session_data(experiment_name):
    exp_data_path = os.path.join("datasets", experiment_name)
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


def evaluation_loop(config: Config, train_dataset: pd.DataFrame, test_dataset: pd.DataFrame):
    with mlflow.start_run() as run:
        run = mlflow.active_run()
        artifacts_path = f'artifacts/{run.info.run_id}'
        os.makedirs(artifacts_path)
        
        model_module = importlib.import_module(f"mastercard.models.{config.model_name}")

        artifacts = model_module.train_pipe(config, train_dataset)
        predict_dataset = model_module.predict_pipe(config, artifacts, test_dataset)
        metrics = model_module.evaluate.evaluate_pipe(config, test_dataset, predict_dataset)
        
        for name, artifact in dict(artifacts).items():
            try:
                joblib.dump(artifact, f"{artifacts_path}/{name}.joblib")
            except Exception:
                print(F'Failed to save model artifact: {name}')
        
        mlflow.log_params(dict(config))
        mlflow.log_metrics(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Experiment runner")
    parser.add_argument("--exp", type=str, help="Name of experiment configuration file")
    parser.add_argument("--workers", type=int, default=4, help="Name of experiment configuration file")
    args = parser.parse_args()

    configs_module = importlib.import_module(f"configs.{args.exp}")
    configs = [config for config in configs_module.params]

    mlflow.set_tracking_uri(uri="http://127.0.0.1:5002")
    experiment_name = configs_module.experiment_session_id
    mlflow.set_experiment(experiment_name)

    train_dataset, test_dataset = initialize_experiment_session_data(experiment_name)

    if args.workers == 1:
        for config in configs:
            evaluation_loop(config, train_dataset, test_dataset)
    else:
        with mp.Pool(processes=args.workers) as pool:
            pool.starmap(evaluation_loop, [(config, train_dataset, test_dataset) for config in configs])