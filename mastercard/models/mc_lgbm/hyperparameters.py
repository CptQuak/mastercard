from typing import List, Literal, Optional
from lightgbm import LGBMClassifier
from pydantic import BaseModel, ConfigDict, Field

LGBMClassifier


class Hyperparameters(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    numeric_features: List[str]
    categorical_features: List[str] = Field(default_factory=list)
    #### model params
    eval_metric: Literal["auc"] = "auc"
    class_weight: Literal["balanced"] = "balanced"
    n_jobs: int = 2
    prob_calibration: bool = True
    user_statistics: bool = True
    time_features: bool = True
    boosting_type: Literal["gbdt", "dart"] = "gbdt"
    num_leaves: int = 31
    max_depth: int = -1
    learning_rate: float = 1e-1
    n_estimators: int = 100
    min_split_gain: float = 0.1
    reg_alpha: float = 1e-1
    reg_lambda: float = 1e-1
