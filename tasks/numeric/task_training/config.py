from pydantic import BaseModel


class TrainConfig(BaseModel):
    task_name: str

    model_file_name: str
    sklearn_model_path: str | None

    model_params: dict
