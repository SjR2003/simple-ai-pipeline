from pydantic import BaseModel


class EdaDsConfig(BaseModel):
    task_name: str

    show: bool
