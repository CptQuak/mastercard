from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class Hyperparameters(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    numeric_features: List[str]
    categorical_features: List[str] = Field(default_factory=list)
    #### model params