from pydantic import BaseModel
import pandas as pd


class TrainResult(BaseModel):
    x: pd.DataFrame
    y: pd.Series
    model_config = {"arbitrary_types_allowed": True}
