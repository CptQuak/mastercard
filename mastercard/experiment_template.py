from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator

class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_name: str 
    comment: str = ""
    target: str = "is_outlier"
    columns: List[str] = []
    