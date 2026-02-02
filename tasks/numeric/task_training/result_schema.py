from pydantic import BaseModel
from typing import Any, Dict

from tasks.numeric.task_preprocess_ds.result_schema import PreprocessResult


class TrainResult(BaseModel):
    model: Any
    metrics: dict
    data: PreprocessResult
    model_params: Dict[str, Any]
    model_name: str
    model_type: str
