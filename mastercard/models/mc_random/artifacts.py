import polars as pl
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class Artifacts(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    features: List[str]
    model: BaseEstimator|Pipeline