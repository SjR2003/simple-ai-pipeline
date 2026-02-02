from pydantic import BaseModel


class TuningConfig(BaseModel):
    task_name: str

    search_method: str
    n_trials: int
    metric: str
    direction: str
    param_space: dict
