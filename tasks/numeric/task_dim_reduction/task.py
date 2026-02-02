import shutil
import joblib
import logging
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import umap.umap_ as umap
from typing import Optional
from sklearn.decomposition import PCA
from mlflow.models.signature import infer_signature
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from core.base_task import BaseTask
from core.task_registry import register_task
from tasks.numeric.task_dim_reduction.config import DimReductionConfig, ReductionMethod
from tasks.numeric.task_dim_reduction.modules.plotter import *
from tasks.numeric.task_load_ds.result_schema import LoadDsResult
from tasks.numeric.task_preprocess_ds.result_schema import PreprocessResult


logging.getLogger("faker").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

current_state = datetime.datetime.now()
saved_state = None


@register_task("dim_reduction")
class DimReduction(BaseTask):
    def __init__(self, config: dict, input_data: any):
        super().__init__(config, input_data)
        np.random.seed(self._seed)
        self._reduced_data = None
        self._model = None
        self._explained_variances = {}

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
        self._config = DimReductionConfig.model_validate(self._config_dict)

    def _load_data(self) -> None:
        pass

    @property
    def result(self) -> PreprocessResult:
        return self._result

    def _apply_pca(self, data: pd.DataFrame, n_components: int) -> pd.DataFrame:
        n_components = min(len(data.columns), n_components)
        pca = PCA(n_components=n_components, random_state=self._seed)
        reduced = pca.fit_transform(data)
        self._model = pca
        self._explained_variances["pca"] = {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "singular_values": pca.singular_values_.tolist(),
        }
        return pd.DataFrame(reduced, columns=[f"PC_{i+1}" for i in range(n_components)])

    def _apply_umap(
        self, data: pd.DataFrame, n_components: int, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=self._config.umap_n_neighbors,
            min_dist=self._config.umap_min_dist,
            metric=self._config.umap_metric,
            random_state=self._seed,
        )

        if y is not None and self._config.umap_supervised:
            reduced = umap_model.fit_transform(data, y=y)
        else:
            reduced = umap_model.fit_transform(data)

        self._model = umap_model
        return pd.DataFrame(
            reduced, columns=[f"UMAP_{i+1}" for i in range(n_components)]
        )

    def _apply_lda(
        self, data: pd.DataFrame, n_components: int, labels: pd.Series
    ) -> pd.DataFrame:
        n_classes = len(np.unique(labels))
        max_components = n_classes - 1

        if n_components > max_components:
            self._logger.warning(
                f"LDA can only produce {max_components} components (n_classes - 1). "
                f"Reducing from {n_components} to {max_components}"
            )
            n_components = max_components

        lda = LinearDiscriminantAnalysis(n_components=n_components)
        reduced_data = lda.fit_transform(data, labels)
        self._model = lda

        columns = [f"LDA_{i+1}" for i in range(n_components)]
        return pd.DataFrame(reduced_data, columns=columns)

    def run(self) -> None:
        train_data = self._injected_data.train.x
        train_labels = self._injected_data.train.y
        test_data = self._injected_data.test.x
        test_labels = self._injected_data.test.y

        reduced_train = None
        if self._config.method == ReductionMethod.PCA:
            reduced_train = self._apply_pca(train_data, self._config.n_components)

        elif self._config.method == ReductionMethod.UMAP:
            reduced_train = self._apply_umap(
                train_data, self._config.n_components, train_labels
            )

        elif self._config.method == ReductionMethod.LDA:
            reduced_train = self._apply_lda(
                train_data, self._config.n_components, train_labels
            )

        reduced_test = None
        if self._config.method == ReductionMethod.PCA and self._model:
            reduced_test = pd.DataFrame(
                self._model.transform(test_data),
                columns=[f"PC_{i+1}" for i in range(self._config.n_components)],
            )

        elif self._config.method == ReductionMethod.LDA and self._model:
            reduced_test = pd.DataFrame(
                self._model.transform(test_data),
                columns=[
                    f"LDA_{i+1}"
                    for i in range((len(np.unique(self._injected_data.train.y)) - 1))
                ],
            )

            self._config.n_components = len(np.unique(self._injected_data.train.y)) - 1

        elif self._config.method == ReductionMethod.UMAP and self._model:
            reduced_test = pd.DataFrame(
                self._model.transform(test_data),
                columns=[f"UMAP_{i+1}" for i in range(self._config.n_components)],
            )

        vis_result_path = visualize_results(
            reduced_train,
            self._config.method,
            self._output_path,
            train_labels,
            self._config.show,
            f"Training Data - {self._config.method.upper()}",
        )
        self._log_artifact(
            vis_result_path,
            "dim-reduction - dimension_reduction",
        )

        if self._config.method == ReductionMethod.PCA:
            vis_result_path = plot_explained_variance(
                self._explained_variances, self._output_path, self._config.show
            )
            if vis_result_path is not None:
                self._log_artifact(
                    vis_result_path,
                    "dim-reduction - explained_variance_per_principal_component",
                )

        self._logger.info(f"Dimension reduction completed successfully!")
        self._logger.info(f"Original shape: {train_data.shape}")
        self._logger.info(f"Reduced shape: {reduced_train.shape}")

        if (
            self._config.method == ReductionMethod.PCA
            and "pca" in self._explained_variances
        ):
            cumulative = self._explained_variances["pca"]["cumulative_variance"]
            self._log_param("dim-red PCA variance", f"{cumulative[-1]:.4f}")
            self._logger.info(f"Cumulative variance: {cumulative[-1]:.4f}")

        model_type_name = type(self._model).__name__.lower()
        filepath = self._output_path / f"{model_type_name}.joblib"
        joblib.dump(self._model, filepath)
        signature = infer_signature(train_data, reduced_train)

        self._log_param("dim-red method", self._config.method)
        self._log_param(f"dim-red component", self._config.n_components)
        self._model._model = self._model
        self._log_model(
            f"{model_type_name}_model", model_type="sklearn", signature=signature
        )

        self._result = PreprocessResult(
            train=LoadDsResult(x=reduced_train, y=train_labels),
            test=LoadDsResult(x=reduced_test, y=test_labels),
        )
