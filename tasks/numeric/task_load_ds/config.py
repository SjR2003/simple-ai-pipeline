from pydantic import BaseModel


class LoadDsConfig(BaseModel):
    task_name: str

    project: str
    experiment: str
    seed: int
    dataset_src: str
    dataset_path: str
    target_name: str
    show: bool
