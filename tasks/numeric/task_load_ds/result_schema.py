from pydantic import BaseModel
import pandas as pd


class LoadDsResult(BaseModel):
    x: pd.DataFrame
    y: pd.Series
    model_config = {"arbitrary_types_allowed": True}
