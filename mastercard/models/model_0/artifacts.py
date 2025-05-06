from typing import List
from pydantic import BaseModel, ConfigDict
from sklearn.pipeline import Pipeline


class Artifacts(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: Pipeline
    features: List[str]