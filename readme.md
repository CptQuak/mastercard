# Env setup

## Install dependencies

pip install -r requirements.txt


## Install local package in editable mode

pip install -e .

---
# Running experiments

## Start mlflow server
mlflow server --host 127.0.0.1 --port 5001

## Creating experiment session

TODO: Update mastercard.experiment_session.create_session to use appropriate data

## Creating experiment file

In experiments/configs model create new file experiment file:
1. Define name of experiment_session_id, it will use the data from the experiment session definition after each run so that results are reproducible. (fixes train and test sets)
2. Define params list
- stores a configuration of model experiments to run with optuna


## Running experiments with main.py file

cd experiments

python main.py --exp config_mastercard_logistic