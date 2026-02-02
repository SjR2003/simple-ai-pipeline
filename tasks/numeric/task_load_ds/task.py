import os
import shutil
import logging
import hashlib
import datetime
import mlflow
import pandas as pd
from pathlib import Path

from core.base_task import BaseTask
from core.task_registry import register_task
from tasks.numeric.task_load_ds.config import LoadDsConfig
from tasks.numeric.task_load_ds.result_schema import LoadDsResult

logging.getLogger("faker").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

current_state = datetime.datetime.now()
saved_state = None


@register_task("load_ds")
class LoadDs(BaseTask):
    def __init__(self, config: dict, input_data: any):
        super().__init__(config, input_data)
        self._dataset_fingerprint = {}

        global current_state, saved_state
        if saved_state == None:
            saved_state = current_state
            self._new_id = True
        elif current_state == saved_state:
            self._new_id = False

        self._output_path = self._output_path / Path(__file__).resolve().parent.name
        if self._output_path.exists() and self._new_id:
            shutil.rmtree(self._output_path)

        self._output_path.mkdir(exist_ok=True, parents=True)

        self._run_id = 0
        result_dir = f"result_{self._run_id}"

        if self._output_path.exists() and not self._new_id:
            self._run_id = self._get_last_folder()
            result_dir = f"result_{self._run_id}"

        self._output_path = self._output_path / result_dir
        self._output_path.mkdir(exist_ok=True, parents=True)

    def _validate_config(self) -> None:
        self._config = LoadDsConfig.model_validate(self._config_dict)

    def _load_data(self) -> None:
        self._inside_data = pd.read_csv(self._config.dataset_path)

    @property
    def result(self) -> LoadDsResult:
        return self._result

    def _compute_dataset_fingerprint(self) -> None:
        try:
            file_stats = os.stat(self._config.dataset_path)

            file_hash = hashlib.sha256()
            file_size = file_stats.st_size

            if file_size <= 3 * 1024 * 1024:
                with open(self._config.dataset_path, "rb") as f:
                    file_hash.update(f.read())
            else:
                sample_positions = [0, file_size // 2, file_size - 1024 * 1024]
                with open(self._config.dataset_path, "rb") as f:
                    for pos in sample_positions:
                        f.seek(pos)
                        file_hash.update(f.read(1024 * 1024))

            row_count, col_count = self._inside_data.shape

            fingerprint = {
                "file_path": self._config.dataset_path,
                "file_size_bytes": file_size,
                "file_modified_time": datetime.datetime.fromtimestamp(
                    file_stats.st_mtime
                ).isoformat(),
                "row_count": row_count,
                "column_count": col_count,
                "target_column": self._config.target_name,
                "feature_columns_count": len(self._inside_data.columns) - 1,
                "partial_sha256": file_hash.hexdigest(),
                "computed_at": datetime.datetime.now().isoformat(),
                "dataset_name": (
                    self._config.project if hasattr(self._config, "project") else None
                ),
                "dataset_source": (
                    self._config.dataset_src
                    if hasattr(self._config, "dataset_src")
                    else None
                ),
            }

            self._dataset_fingerprint = fingerprint

        except Exception as e:
            self._loger.warning(f"Could not compute dataset fingerprint: {e}")

    def run(self) -> None:
        target = self._inside_data[self._config.target_name]
        features = self._inside_data.drop(self._config.target_name, axis=1)
        self._compute_dataset_fingerprint()
        if len(self._dataset_fingerprint) > 0:
            self._log_dataset(
                self._dataset_fingerprint,
                self._output_path / "dataset_fingerprint.json",
            )

        self._result = LoadDsResult(x=features, y=target)

        if self._config.show:
            self._loger.info(f"*" * 50)
            self._loger.info(f"Dataset {self._config.project} info:")
            self._loger.info(f"Dataset src link: {self._config.dataset_src}")
            self._inside_data.info()
            self._loger.info(f"*" * 50)
