from pydantic import BaseModel


class TrainConfig(BaseModel):
    task_name: str

    model_file_name: str
    model_type: str
    model_params: dict
    train_params: dict = {}
