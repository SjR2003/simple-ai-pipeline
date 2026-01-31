from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
import os
import importlib

from core.base_task import BaseTask
from core.task_registry import register_task
from tasks.numeric.task_training.config import TrainConfig
from tasks.numeric.task_training.result_schema import TrainResult


logging.getLogger("faker").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


@register_task("training")
class Training(BaseTask):
    def __init__(self, config: dict, input_data: any):
        super().__init__(config, input_data)

    def _validate_config(self) -> None:
        self._config = TrainConfig.model_validate(self._config_dict)

    def _load_data(self):
        self._train_data = self._injected_data.train
        self._test_data = self._injected_data.test

        module_path = f"models.{self._config.model_file_name}"
        module = importlib.import_module(module_path)
        model_class = getattr(
            module, self._snake_to_pascal(self._config.model_file_name)
        )

        if self._config.sklearn_model_path:
            full_path = self._config.sklearn_model_path
            module_path, class_name = full_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            rf_model_class = getattr(module, class_name)
            params = dict(self._config.model_params)
            params["random_state"] = self._seed
            rf_model = rf_model_class(**params)
            self._model = model_class(rf_model)
        else:
            params = dict(self._config.model_params)
            print(params["hidden_sizes"])
            params["input_size"] = self._train_data.x.shape[1]
            y = self._train_data.y.values
            unique_values = np.unique(y)
            n_unique = len(unique_values)
            params["output_size"] = n_unique
            self._model = model_class(**params)

    @property
    def result(self) -> TrainResult:
        return self._result

    def _snake_to_pascal(self, name: str) -> str:
        return "".join(part.capitalize() for part in name.split("_"))

    def run(self) -> None:
        history = self._model.train(
            self._train_data.x.to_numpy(),
            self._train_data.y.to_numpy(),
            self._test_data.x.to_numpy(),
            self._test_data.y.to_numpy(),
        )
        print(history)
