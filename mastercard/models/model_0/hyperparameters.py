from typing import Optional
from pydantic import BaseModel, ConfigDict


class Hyperparameters(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    penalty: Optional[str] = "l2"
    C: Optional[float] = 1.0