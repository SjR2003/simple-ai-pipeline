import os
import shutil
import datetime
import pandas as pd
from pathlib import Path

from core.base_task import BaseTask
from core.task_registry import register_task
from tasks.numeric.task_eda_ds.config import EdaDsConfig
from tasks.numeric.task_eda_ds.modules.analyzers import *
from tasks.numeric.task_eda_ds.modules.plotters import *
from tasks.numeric.task_load_ds.result_schema import LoadDsResult
from tasks.numeric.task_preprocess_ds.result_schema import PreprocessResult

current_state = datetime.datetime.now()
saved_state = None


@register_task("eda_ds")
class EdaDs(BaseTask):
    def __init__(self, config: dict, input_data: any) -> None:
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

        self._run_id = 0
        result_dir = f"result_{self._run_id}"

        if self._output_path.exists() and not self._new_run:
            self._run_id = self._get_last_folder()
            result_dir = f"result_{self._run_id}"

        self._output_path = self._output_path / result_dir
        self._output_path.mkdir(exist_ok=True, parents=True)

    def _get_last_folder(self) -> int:
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
        X_df = self._df.x
        y_df = self._df.y

        if isinstance(X_df, pd.DataFrame):
            X = X_df.values
        if isinstance(y_df, (pd.Series, pd.DataFrame)):
            y = y_df.values.ravel()

        n_samples, n_features = X.shape

        target_stats = analyze_target(y, random_state=self._seed)
        missing_stats = analyze_missing(X, random_state=self._seed)
        variance_stats = analyze_variance(X, random_state=self._seed)
        corr_stats = analyze_correlation(X, random_state=self._seed)
        info_stats = analyze_informativeness(X, y, random_state=self._seed)
        outlier_stats = analyze_outliers(X, random_state=self._seed)
        feature_type_stats = analyze_feature_types(X, random_state=self._seed)
        importance_stats = analyze_feature_target_relationship(
            X, y, random_state=self._seed
        )

        self._log_metric(f"eda/n_samples_{self._run_id}", n_samples)
        self._log_metric(f"eda/n_features_{self._run_id}", n_features)
        self._log_metric(f"eda/n_classes_{self._run_id}", target_stats["n_classes"])
        self._log_metric(
            f"eda/imbalance_ratio_{self._run_id}", target_stats["imbalance_ratio"]
        )
        self._log_metric(f"eda/class_entropy_{self._run_id}", target_stats["entropy"])
        self._log_metric(f"eda/class_gini_{self._run_id}", target_stats["gini"])
        self._log_metric(
            f"eda/missing_ratio_{self._run_id}", missing_stats["missing_ratio"]
        )
        self._log_metric(
            f"eda/low_variance_ratio_{self._run_id}",
            variance_stats["low_variance_ratio"],
        )
        self._log_metric(
            f"eda/mean_abs_corr_{self._run_id}", corr_stats["mean_abs_corr"]
        )
        self._log_metric(
            f"eda/high_corr_ratio_{self._run_id}", corr_stats["high_corr_ratio"]
        )
        self._log_metric(f"eda/mean_mi_{self._run_id}", info_stats["mean_mi"])
        self._log_metric(
            f"eda/zero_mi_ratio_{self._run_id}", info_stats["zero_mi_ratio"]
        )
        self._log_metric(
            f"eda/mean_outlier_ratio_{self._run_id}",
            outlier_stats["mean_outlier_ratio"],
        )
        self._log_metric(
            f"eda/features_with_high_outliers_{self._run_id}",
            outlier_stats["features_with_high_outliers"],
        )
        self._log_metric(
            f"eda/binary_features_ratio_{self._run_id}",
            feature_type_stats["binary_ratio"],
        )
        self._log_metric(
            f"eda/n_significant_features_{self._run_id}",
            importance_stats["n_significant_features"],
        )
        df_y = pd.DataFrame(
            {
                "class": list(target_stats["counts"].keys()),
                "count": list(target_stats["counts"].values()),
            }
        )

        cls_html = plot_class_dist(df_y, self._output_path, self._run_id)
        self._log_artifact(cls_html, f"eda - class_distribution_{self._run_id}")

        corr_html = plot_correlation_matrix(
            X_df, y_df, self._output_path, self._run_id, random_state=self._seed
        )
        self._log_artifact(corr_html, f"eda - correlation_matrix_{self._run_id}")

        importance_html = plot_feature_importance(
            importance_stats, self._output_path, self._run_id
        )
        self._log_artifact(importance_html, f"eda - feature_importance_{self._run_id}")

        outlier_html = plot_outliers_distribution(
            outlier_stats, X, self._output_path, self._run_id
        )
        self._log_artifact(outlier_html, f"eda - outlier_analysis_{self._run_id}")

        dist_html = plot_feature_distributions_by_class(
            X, y, self._output_path, self._run_id, random_state=self._seed
        )
        self._log_artifact(dist_html, f"eda - feature_distributions_{self._run_id}")

        var_html = plot_variance(X, self._output_path, self._run_id)
        self._log_artifact(var_html, f"eda - feature_variance_{self._run_id}")

        nan_html = plot_missing(missing_stats, self._output_path, self._run_id)
        self._log_artifact(nan_html, f"eda - missing_ratio_{self._run_id}")

        self._result = self._injected_data
