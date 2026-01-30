import os
import shutil
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif

from core.base_task import BaseTask
from core.task_registry import register_task
from tasks.numeric.task_eda_ds.config import EdaDsConfig
from tasks.numeric.task_load_ds.result_schema import LoadDsResult
from tasks.numeric.task_preprocess_ds.result_schema import PreprocessResult

current_state = datetime.datetime.now()
saved_state = None


@register_task("eda_ds")
class EdaDs(BaseTask):
    def __init__(self, config: dict, input_data: any):
        super().__init__(config, input_data)
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

    def _get_last_folder(self):
        numbers = []
        for folder in os.listdir(self._output_path):
            numbers.append(folder)

        if not numbers:
            return 0

        return len(numbers)

    def _validate_config(self) -> None:
        self._config = EdaDsConfig.model_validate(self._config_dict)

    def _load_data(self):
        if type(self._injected_data) == LoadDsResult:
            self._df = self._injected_data
        elif type(self._injected_data) == PreprocessResult:
            self._df = self._injected_data.train

    @property
    def result(self) -> LoadDsResult | PreprocessResult:
        return self._result

    def run(self) -> None:
        X = self._df.x
        y = self._df.y

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.ravel()

        n_samples, n_features = X.shape

        target_stats = self._analyze_target(y)
        missing_stats = self._analyze_missing(X)
        variance_stats = self._analyze_variance(X)
        corr_stats = self._analyze_correlation(X)
        info_stats = self._analyze_informativeness(X, y)

        self._log_params(f"eda/n_samples_{self._run_num}", n_samples)
        self._log_params(f"eda/n_features_{self._run_num}", n_features)
        self._log_params(f"eda/n_classes_{self._run_num}", target_stats["n_classes"])
        self._log_params(
            f"eda/imbalance_ratio_{self._run_num}", target_stats["imbalance_ratio"]
        )
        self._log_params(f"eda/class_entropy_{self._run_num}", target_stats["entropy"])
        self._log_params(f"eda/class_gini_{self._run_num}", target_stats["gini"])
        self._log_params(
            f"eda/missing_ratio_{self._run_num}", missing_stats["missing_ratio"]
        )
        self._log_params(
            f"eda/low_variance_ratio_{self._run_num}",
            variance_stats["low_variance_ratio"],
        )
        self._log_params(
            f"eda/mean_abs_corr_{self._run_num}", corr_stats["mean_abs_corr"]
        )
        self._log_params(
            f"eda/high_corr_ratio_{self._run_num}", corr_stats["high_corr_ratio"]
        )
        self._log_params(f"eda/mean_mi_{self._run_num}", info_stats["mean_mi"])
        self._log_params(
            f"eda/zero_mi_ratio_{self._run_num}", info_stats["zero_mi_ratio"]
        )

        df_y = pd.DataFrame(
            {
                "class": list(target_stats["counts"].keys()),
                "count": list(target_stats["counts"].values()),
            }
        )

        fig_cls = px.bar(
            df_y,
            x="class",
            y="count",
            title=f"Class Distribution (EDA) {self._run_num}",
        )

        cls_html = self._output_path / f"class_distribution_{self._run_num}.html"
        fig_cls.write_html(cls_html)

        self._log_artifact(cls_html, f"eda - class_distribution_{self._run_num}")

        fig_var = px.histogram(
            np.nanvar(X, axis=0),
            nbins=50,
            title=f"Feature Variance Distribution {self._run_num}",
        )

        var_html = self._output_path / f"feature_variance_{self._run_num}.html"
        fig_var.write_html(var_html)

        self._log_artifact(var_html, f"eda - feature_variance_{self._run_num}")

        fig_nan = px.histogram(
            missing_stats["per_feature_ratio"],
            nbins=50,
            title=f"Missing Ratio per Feature {self._run_num}",
        )

        nan_html = self._output_path / f"missing_ratio_{self._run_num}.html"
        fig_nan.write_html(nan_html)

        self._log_artifact(nan_html, f"eda - missing_ratio_{self._run_num}")

        self._result = self._injected_data

    def _analyze_target(self, y: np.ndarray) -> dict:
        values, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()

        return {
            "n_classes": int(len(values)),
            "counts": dict(zip(values.tolist(), counts.tolist())),
            "percentages": dict(zip(values.tolist(), (probs * 100).tolist())),
            "imbalance_ratio": float(counts.max() / counts.min()),
            "entropy": float(entropy(probs)),
            "gini": float(1.0 - np.sum(probs**2)),
        }

    def _analyze_missing(self, X: np.ndarray) -> dict:
        nan_mask = np.isnan(X)

        return {
            "total_missing": int(nan_mask.sum()),
            "missing_ratio": float(nan_mask.mean()),
            "per_feature_ratio": nan_mask.mean(axis=0).tolist(),
        }

    def _analyze_variance(self, X: np.ndarray) -> dict:
        var = np.nanvar(X, axis=0)

        return {
            "mean_variance": float(var.mean()),
            "low_variance_ratio": float((var < 1e-6).mean()),
            "min_variance": float(var.min()),
            "max_variance": float(var.max()),
        }

    def _analyze_correlation(self, X: np.ndarray) -> dict:
        corr = np.corrcoef(X, rowvar=False)
        upper = np.abs(corr[np.triu_indices_from(corr, k=1)])

        return {
            "mean_abs_corr": float(np.nanmean(upper)),
            "high_corr_ratio": float((upper > 0.9).mean()),
            "max_corr": float(np.nanmax(upper)),
        }

    def _analyze_informativeness(self, X: np.ndarray, y: np.ndarray) -> dict:
        mi = mutual_info_classif(X, y, discrete_features=False)

        return {
            "mean_mi": float(mi.mean()),
            "zero_mi_ratio": float((mi < 1e-4).mean()),
            "max_mi": float(mi.max()),
        }
