import os
import os
import re
import json
import mlflow
import logging
from pathlib import Path
from typing import Optional, Any, Dict
from abc import ABC, abstractmethod

from utils.git_info import get_git_info_gitpython

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

        self._init_global_variables()
        self._injected_data = input_data
        self._inside_data = None
        self._load_data()

        self._result = None

    def _init_global_variables(self):
        global output_path
        output_path = (
            Path(output_path) if not isinstance(output_path, Path) else output_path
        )

        output_path.mkdir(parents=True, exist_ok=True)

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
            base_exp = self._config_dict["experiment"]

            exp_root = os.path.join(output_path, self._project)

            max_suffix = -1
            pattern = re.compile(rf"^{re.escape(base_exp)}(?:_(\d+))?$")

            if os.path.isdir(exp_root):
                for name in os.listdir(exp_root):
                    match = pattern.match(name)
                    if match:
                        suffix = match.group(1)
                        suffix = int(suffix) if suffix is not None else 0
                        max_suffix = max(max_suffix, suffix)

            if max_suffix < 0:
                max_suffix = 0
            else:
                max_suffix = max_suffix + 1

            mlflow.set_tag("version", f"v{max_suffix}.0.0")
            exp = f"{base_exp}_{max_suffix}"

            git_info = get_git_info_gitpython()
            if git_info:
                mlflow.set_tag("commit_hash", git_info["short_hash"])

        self._exp = exp

        global seed
        if "seed" in self._config_dict and not seed:
            seed = self._config_dict.get("seed", 42)
        self._seed = seed

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

    def _log_params(self, params: dict) -> None:
        """Logs parameters to MLflow.

        Args:
            params (dict): A dictionary of parameters to log.
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
            self._logger.debug(f"Logged parameter {key}: {value} to MLflow.")

    def _log_param(self, key, value) -> None:
        """Logs parameter to MLflow."""

        mlflow.log_param(key, value)
        self._logger.debug(f"Logged parameter {key}: {value} to MLflow.")

    def _log_metric(self, key, value, step: int = None) -> None:
        """Logs metrics to MLflow.

        Args:
            metrics (dict): A dictionary of metrics to log.
        """
        if step is not None:
            mlflow.log_metric(key, value, step=step)
        else:
            mlflow.log_metric(key, value)
        self._logger.debug(f"Logged metric {key}: {value} to MLflow.")

    def _log_model(self, model_key: str, model_type: str, signature) -> None:
        if model_type == "sklearn":
            mlflow.sklearn.log_model(
                sk_model=self._model._model,
                artifact_path=f"models_{model_key}",
                registered_model_name=model_key,
                signature=signature,
            )

        elif model_type == "pytorch":
            mlflow.pytorch.log_model(
                pytorch_model=self._model._model,
                artifact_path=f"models_{model_key}",
                registered_model_name=model_key,
                signature=signature,
            )

        elif model_type == "xgboost":
            mlflow.xgboost.log_model(
                xgb_model=self._model._model,
                artifact_path=f"models_{model_key}",
                registered_model_name=model_key,
                signature=signature,
            )

        elif model_type == "catboost":
            mlflow.catboost.log_model(
                cb_model=self._model._model,
                artifact_path=f"models_{model_key}",
                registered_model_name=model_key,
                signature=signature,
            )

        elif model_type == "lightgbm":
            mlflow.lightgbm.log_model(
                lgb_model=self._model._model,
                artifact_path=f"models_{model_key}",
                registered_model_name=model_key,
                signature=signature,
            )

    def _log_dataset(
        self,
        fingerprint: Dict[str, Any],
        out_path: str,
    ) -> None:
        """
        Log dataset fingerprint to MLflow in a standardized way.

        Args:
            fingerprint: Dataset fingerprint dictionary
            artifact_path: Path within MLflow artifacts
        """
        if not fingerprint:
            return

        mlflow.log_params(
            {
                "dataset.rows": fingerprint.get("row_count", 0),
                "dataset.columns": fingerprint.get("column_count", 0),
                "dataset.target": fingerprint.get("target_column", "unknown"),
                "dataset.name": fingerprint.get("dataset_name", "unknown"),
            }
        )

        mlflow.set_tag("dataset.hash", fingerprint.get("partial_sha256", "unknown"))

        with open(file=f"{out_path}", mode="w") as f:
            json.dump(fingerprint, f, indent=2)
            mlflow.log_artifact(f"{out_path}", f"dataset_fingerprint.json")

        dataset = mlflow.data.from_pandas(
            self._inside_data,
            source=self._config.dataset_src,
            name=self._config.project,
            targets=self._config.target_name,
        )
        mlflow.log_input(dataset)
