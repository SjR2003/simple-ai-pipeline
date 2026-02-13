import time
import shutil
import logging
import datetime
import importlib
import numpy as np
from pathlib import Path
from mlflow.models.signature import infer_signature

from core.base_task import BaseTask
from core.base_model import BaseMLModel
from core.task_registry import register_task
from tasks.numeric.task_training.config import TrainConfig
from tasks.numeric.task_training.modules.plotter import *
from tasks.numeric.task_training.result_schema import TrainResult


logging.getLogger("faker").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

current_state = datetime.datetime.now()
saved_state = None


@register_task("training")
class Training(BaseTask):
    def __init__(self, config: dict, input_data: any):
        super().__init__(config, input_data)
        np.random.seed(self._seed)
        global current_state, saved_state
        if saved_state == None:
            saved_state = current_state
            self._new_run = True
        elif current_state == saved_state:
            self._new_run = False

        self._output_path = self._output_path / Path(__file__).resolve().parent.name
        if self._output_path.exists() and self._new_run:
            shutil.rmtree(self._output_path)

        self._output_path.mkdir(exist_ok=True, parents=True)

        self._run_num = 0
        result_dir = f"result_{self._run_num}"

        if self._output_path.exists() and not self._new_run:
            self._run_num = self._get_last_folder()
            result_dir = f"result_{self._run_num}"

        self._output_path = self._output_path / result_dir
        self._output_path.mkdir(exist_ok=True, parents=True)

    def _validate_config(self) -> None:
        self._config = TrainConfig.model_validate(self._config_dict)

    def _load_data(self) -> None:
        self._train_data = self._injected_data.train
        self._test_data = self._injected_data.test

        module_path = f"models.{self._config.model_file_name}"
        module = importlib.import_module(module_path)
        model_class = getattr(
            module, self._snake_to_pascal(self._config.model_file_name)
        )

        y = self._train_data.y.values
        unique_values = np.unique(y)
        classes = len(unique_values)

        params = dict(self._config.model_params)
        params["random_state"] = self._seed
        params["input_size"] = self._train_data.x.shape[1]
        params["output_size"] = classes
        for param, value in params.items():
            self._log_param(f"train_{param}", value)

        self._model = model_class(**params)

    @property
    def result(self) -> TrainResult:
        return self._result

    def _snake_to_pascal(self, name: str) -> str:
        return "".join(part.capitalize() for part in name.split("_"))

    def _log_inference_time(
        self,
        model: BaseMLModel,
        X: np.ndarray,
        n_runs: int = 50,
        prefix: str = "inference",
    ) -> None:
        times = []

        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model.predict(X)
            times.append(time.perf_counter() - start)

        times = np.array(times)

        self._log_metric(f"{prefix}_mean_ms", times.mean() * 100)
        self._log_metric(f"{prefix}_p95_ms", np.percentile(times, 95) * 100)
        self._log_metric(f"{prefix}_max_ms", times.max() * 100)

    def run(self) -> None:
        X_train = self._train_data.x.to_numpy()
        y_train = self._train_data.y.to_numpy()
        X_test = self._test_data.x.to_numpy()
        y_test = self._test_data.y.to_numpy()

        train_params = dict(self._config.train_params)
        prefix = "train_"
        prefixed_dict = {f"{prefix}{key}": value for key, value in train_params.items()}
        self._log_params(prefixed_dict)

        train_params["X_train"] = X_train
        train_params["y_train"] = y_train
        train_params["X_val"] = X_test
        train_params["y_val"] = y_test
        history = self._model.train(**train_params)
        loss_plot_path, acc_plot_path = plot_history(history, self._output_path)
        self._log_artifact(loss_plot_path, "train - loss curve")
        self._log_artifact(acc_plot_path, "train - accuracy curve")

        metrics = self._model.eval(X_test, y_test)
        confusion_plot_path = plot_confusion_matrix(
            metrics["confusion"], self._output_path, random_state=self._seed
        )
        self._log_artifact(confusion_plot_path, "train - confusion matrix")

        per_class_path = plot_per_class_metrics(
            metrics["per_class_metrics"], self._output_path
        )
        for key, value in per_class_path.items():
            self._log_artifact(value, f"train - {key}")

        self._log_metric("train - accuracy", metrics["accuracy"])
        self._log_metric("train - f1_score", metrics["f1_score"])
        self._log_metric("train - recall", metrics["recall"])
        self._log_metric("train - precision", metrics["precision"])
        self._log_inference_time(self._model, X_test)

        self._model.save_model(self._output_path / f"{self._config.model_file_name}")
        signature = infer_signature(X_train, y_train)
        self._log_model(
            f"train_{self._config.model_file_name}",
            model_type=self._config.model_type,
            signature=signature,
        )

        self._result = TrainResult(
            model=self._model,
            metrics=metrics,
            data=self._injected_data,
            model_params=self._config.model_params,
            model_name=self._config.model_file_name,
            model_type=self._config.model_type,
        )
