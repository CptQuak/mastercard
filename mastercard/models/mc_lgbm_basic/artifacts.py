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
    transformer: ColumnTransformer
    hyperparams: Dict[str, Any]
    user_statistics: Dict[str, pl.DataFrame] = Field(default_factory=dict)
    train_dataset: Optional[pl.DataFrame] = None