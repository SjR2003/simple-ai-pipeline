from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
import os

from core.base_task import BaseTask
from core.task_registry import register_task
from tasks.numeric.task_load_ds.config import LoadDsConfig
from tasks.numeric.task_load_ds.result_schema import LoadDsResult

logging.getLogger("faker").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


@register_task("load_ds")
class LoadDs(BaseTask):
    def __init__(self, config: dict, input_data: any):
        super().__init__(config, input_data)

    def _validate_config(self) -> None:
        self._config = LoadDsConfig.model_validate(self._config_dict)

    def _load_data(self):
        self._inside_data = pd.read_csv(self._config.dataset_path)

    @property
    def result(self) -> LoadDsResult:
        return self._result

    def run(self) -> None:
        target = self._inside_data[self._config.target_name]
        features = self._inside_data.drop(self._config.target_name, axis=1)
        self._result = LoadDsResult(x=features, y=target)

        if self._config.show:
            print(f"*" * 50)
            print(f"Dataset {self._config.project} info: ")
            print(f"Dataset src link: {self._config.dataset_src} ")
            self._inside_data.info()
            print(f"*" * 50)
