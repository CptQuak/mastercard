from typing import Any, Callable, Dict, List, Literal

from git import Optional
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator


class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_name: str
    comment: str = ""
    # target column
    target: str = "is_outlier"
    # features to consider in experiment session
    columns: List[str] = []
    # Optuna specific config
    optuna_random_state: int = 13
    optuna_n_trials: int = 20
    optuna_n_jobs: int = 4
    optuna_direction: Literal["minimize", "maximize"] = "maximize"
    optuna_main_metric: str = "accuracy"

