from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class Hyperparameters(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    numeric_features: List[str]
    categorical_features: List[str] = Field(default_factory=list)
    #### model params
    prob_calibration: bool = False
    user_statistics: bool = False
    time_features: bool = False
    class_weight: Literal["balanced"] = "balanced"
    random_state: int = 13
    penalty: Literal["l1", "l2", "elasticnet"] = "l2"
    C: float = 1.0
    solver: str = "saga"
    max_iter: int = 100
    l1_ratio: float = 0.5
