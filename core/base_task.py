from abc import ABC, abstractmethod
from typing import Optional, Any
from pathlib import Path
import warnings
import logging
from altair import value
import mlflow
import os

project = None
seed = None
exp = None
output_path = Path(__file__).resolve().parent.parent / "out"


class BaseTask(ABC):
    def __init__(self, config: dict, input_data: any):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)

        self._config_dict = config
        self._config = None
        self._validate_config()

        self._injected_data = input_data
        self._inside_data = None
        self._load_data()

        self._result = None

        self._init_global_variables()

    def _init_global_variables(self):
        global project
        if (
            "project" in self._config_dict
            and len(self._config_dict["project"]) > 0
            and not project
        ):
            project = self._config_dict["project"]
        self._project = project

        global exp
        if (
            "experiment" in self._config_dict
            and len(self._config_dict["experiment"]) > 0
            and not exp
        ):
            exp = self._config_dict["experiment"]
        self._exp = exp

        global seed
        if "seed" in self._config_dict and not seed:
            seed = self._config_dict.get("seed", 42)
        self._seed = seed

        global output_path
        output_path = (
            Path(output_path) if not isinstance(output_path, Path) else output_path
        )

        output_path.mkdir(parents=True, exist_ok=True)

        if self._project and self._exp:
            self._output_path = output_path / self._project / self._exp
            self._output_path.mkdir(parents=True, exist_ok=True)
        else:
            self._output_path = output_path

    @abstractmethod
    def _validate_config(self) -> None:
        """Each task should implement its own config validation logic."""
        pass

    @abstractmethod
    def _load_data(self) -> None:
        """Each task should implement its own data loading logic."""
        pass

    @abstractmethod
    def run(self) -> None:
        """Each task must define its execution logic here."""
        pass

    @property
    @abstractmethod
    def result(self) -> Any:
        """Each task should implement its own data loading logic."""
        pass

    def postprocess(self) -> None:
        """Optional postprocessing steps after run()."""
        pass

    def _log_artifact(
        self, artifact_path: str, artifact_name: Optional[str] = None
    ) -> None:
        """Logs an artifact to MLflow.

        Args:
            artifact_path (str): The local path to the artifact.
            artifact_name (Optional[str]): The name to use for the artifact in MLflow.
                                           If None, uses the basename of artifact_path.
        """
        if not os.path.exists(artifact_path):
            self._logger.error(f"Artifact path {artifact_path} does not exist.")
            return

        if artifact_name is None:
            artifact_name = os.path.basename(artifact_path)

        mlflow.log_artifact(artifact_path, artifact_name)
        self._logger.debug(
            f"Logged artifact {artifact_name} from {artifact_path} to MLflow."
        )

    def _log_result(self) -> None:
        """Logs the result of the task to MLflow."""
        if self._result is not None:
            mlflow.log_param("result", str(self._result))
            self._logger.debug(f"Logged result: {self._result} to MLflow.")
        else:
            self._logger.warning("No result to log.")

    def _log_params(self, key, value) -> None:
        """Logs parameters to MLflow.

        Args:
            params (dict): A dictionary of parameters to log.
        """

        mlflow.log_param(key, value)
        self._logger.debug(f"Logged parameter {key}: {value} to MLflow.")

    def _log_metrics(self, key, value) -> None:
        """Logs metrics to MLflow.

        Args:
            metrics (dict): A dictionary of metrics to log.
        """
        mlflow.log_metric(key, value)
        self._logger.debug(f"Logged metric {key}: {value} to MLflow.")

    def _log_model(self, model_key: str, model_type: str) -> None:
        """Logs the model to MLflow.

        Args:
            model_key: A string identifier for the model (e.g., "pca_model", "random_forest")
            model_type: Type of model ("sklearn", etc.)
        """
        if model_type == "sklearn":
            mlflow.sklearn.log_model(
                sk_model=self._result,
                artifact_path=model_key,
                registered_model_name=model_key,
            )

            if hasattr(self._result, "get_params"):
                params = self._result.get_params()
                for param_name, param_value in params.items():
                    if isinstance(param_value, (int, float, str, bool)):
                        mlflow.log_param(f"{model_key}_{param_name}", param_value)

            self._logger.debug(
                f"Logged model {model_key}: {type(self._result).__name__} to MLflow."
            )
