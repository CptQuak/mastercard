from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class Hyperparameters(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    numeric_features: List[str]
    categorical_features: List[str] = Field(default_factory=list)
    #### model params
    eval_metric: Literal["auc"] = "auc"
    class_weight: Literal["balanced"] = "balanced"
    n_jobs: int = 2
    user_statistics: bool = True
    time_features: bool = True
    max_depth: int = -1
    num_leaves: int = 10 
    min_data_in_leaf: int = 10
    learning_rate: float = 1e-1
    feature_fraction: float = 0.8
    bagging_fraction: float= 0.8
    boosting_type: Literal["gbdt", "dart"]  = 'dart'
    n_estimators: int = 100
    min_split_gain: float = 0.1
    reg_alpha: float = 1e-1
    reg_lambda: float= 1e-1
    min_gain_to_split: float= 0.1
