from pydantic import BaseModel
import pandas as pd

from tasks.numeric.task_load_ds.result_schema import LoadDsResult


class PreprocessResult(BaseModel):
    train: LoadDsResult
    test: LoadDsResult
    model_config = {"arbitrary_types_allowed": True}
