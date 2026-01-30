from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import umap.umap_ as umap
import pandas as pd
import numpy as np
import logging
import shutil
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.base_task import BaseTask
from core.task_registry import register_task
from tasks.numeric.task_dim_reduction.config import DimReductionConfig, ReductionMethod
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

    def _load_data(self):
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

    def _apply_tsne(self, data: pd.DataFrame, n_components: int) -> pd.DataFrame:
        if self._config.tsne_use_pca_init and data.shape[1] > 50:
            pca_init = PCA(
                n_components=min(50, data.shape[0]),
                random_state=self._seed,
            )
            data_for_tsne = pca_init.fit_transform(data)
        else:
            data_for_tsne = data.values

        tsne = TSNE(
            n_components=n_components,
            perplexity=self._config.tsne_perplexity,
            n_iter=self._config.tsne_n_iter,
            random_state=self._seed,
            init=self._config.tsne_init,
        )
        reduced = tsne.fit_transform(data_for_tsne)
        self._model = tsne
        return pd.DataFrame(
            reduced, columns=[f"TSNE_{i+1}" for i in range(n_components)]
        )

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
        self, data: pd.DataFrame, n_components: int, y: pd.Series
    ) -> pd.DataFrame:
        n_classes = len(y.unique())
        n_components = min(n_components, n_classes - 1)

        lda = LDA(n_components=n_components)
        reduced = lda.fit_transform(data, y)
        self._model = lda
        self._explained_variances["lda"] = {
            "explained_variance_ratio": (
                lda.explained_variance_ratio_.tolist()
                if hasattr(lda, "explained_variance_ratio_")
                else []
            )
        }
        return pd.DataFrame(
            reduced, columns=[f"LDA_{i+1}" for i in range(n_components)]
        )

    def _visualize_results(
        self,
        reduced_data: pd.DataFrame,
        y: Optional[pd.Series] = None,
        title: str = "Dimension Reduction",
    ) -> None:
        if reduced_data.shape[1] < 2:
            return

        n_cols = min(2, reduced_data.shape[1])

        if n_cols == 2 and reduced_data.shape[1] >= 3:
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=(
                    f"{title} - First Two Components",
                    f"{title} - First Three Components",
                ),
                specs=[[{"type": "scatter"}, {"type": "scatter3d"}]],
            )

            if y is not None:
                fig.add_trace(
                    go.Scatter(
                        x=reduced_data.iloc[:, 0],
                        y=reduced_data.iloc[:, 1],
                        mode="markers",
                        marker=dict(
                            color=y,
                            colorscale="Viridis",
                            size=6,
                            opacity=0.6,
                            showscale=True,
                            colorbar=dict(x=0.45, thickness=15),
                        ),
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=reduced_data.iloc[:, 0],
                        y=reduced_data.iloc[:, 1],
                        mode="markers",
                        marker=dict(size=6, opacity=0.6),
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )

            fig.update_xaxes(title_text=f"{reduced_data.columns[0]}", row=1, col=1)
            fig.update_yaxes(title_text=f"{reduced_data.columns[1]}", row=1, col=1)

            if y is not None:
                fig.add_trace(
                    go.Scatter3d(
                        x=reduced_data.iloc[:, 0],
                        y=reduced_data.iloc[:, 1],
                        z=reduced_data.iloc[:, 2],
                        mode="markers",
                        marker=dict(
                            color=y,
                            colorscale="Viridis",
                            size=4,
                            opacity=0.6,
                            showscale=True,
                            colorbar=dict(x=1.0, thickness=15),
                        ),
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )
            else:
                fig.add_trace(
                    go.Scatter3d(
                        x=reduced_data.iloc[:, 0],
                        y=reduced_data.iloc[:, 1],
                        z=reduced_data.iloc[:, 2],
                        mode="markers",
                        marker=dict(size=4, opacity=0.6),
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )

            fig.update_scenes(
                xaxis_title=f"{reduced_data.columns[0]}",
                yaxis_title=f"{reduced_data.columns[1]}",
                zaxis_title=f"{reduced_data.columns[2]}",
                row=1,
                col=2,
            )

        else:
            fig = make_subplots(
                rows=1, cols=1, subplot_titles=(f"{title} - First Two Components",)
            )

            if y is not None:
                fig.add_trace(
                    go.Scatter(
                        x=reduced_data.iloc[:, 0],
                        y=reduced_data.iloc[:, 1],
                        mode="markers",
                        marker=dict(
                            color=y,
                            colorscale="Viridis",
                            size=6,
                            opacity=0.6,
                            showscale=True,
                        ),
                        showlegend=False,
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=reduced_data.iloc[:, 0],
                        y=reduced_data.iloc[:, 1],
                        mode="markers",
                        marker=dict(size=6, opacity=0.6),
                        showlegend=False,
                    )
                )

            fig.update_xaxes(title_text=f"{reduced_data.columns[0]}")
            fig.update_yaxes(title_text=f"{reduced_data.columns[1]}")

        fig.update_layout(
            title=f"Dimension Reduction using {self._config.method.upper()}",
            height=500,
            showlegend=False,
            template="plotly_white",
        )

        if self._config.show:
            fig.show()
        fig.write_html(self._output_path / "dimension_reduction.html")
        self._log_artifact(
            self._output_path / "dimension_reduction.html",
            "dim-reduction - dimension_reduction",
        )

    def _plot_explained_variance(self) -> None:
        if "pca" not in self._explained_variances:
            return

        explained_variance = self._explained_variances["pca"][
            "explained_variance_ratio"
        ]
        cumulative_variance = self._explained_variances["pca"]["cumulative_variance"]

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Variance per Principal Component",
                "Cumulative Explained Variance",
            ),
        )

        fig.add_trace(
            go.Bar(
                x=list(range(1, len(explained_variance) + 1)),
                y=explained_variance,
                opacity=0.7,
                marker_color="blue",
                name="Explained Variance",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(cumulative_variance) + 1)),
                y=cumulative_variance,
                mode="lines+markers",
                line=dict(dash="dash", color="red", width=2),
                marker=dict(size=8),
                name="Cumulative Variance",
                showlegend=True,
            ),
            row=1,
            col=2,
        )

        fig.add_hline(
            y=0.95,
            line_dash="dash",
            line_color="green",
            opacity=0.7,
            annotation_text="95% Variance",
            annotation_position="top left",
            row=1,
            col=2,
        )

        fig.add_hline(
            y=0.90,
            line_dash="dash",
            line_color="yellow",
            opacity=0.7,
            annotation_text="90% Variance",
            annotation_position="top left",
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="Principal Component", row=1, col=1)
        fig.update_yaxes(title_text="Explained Variance Ratio", row=1, col=1)

        fig.update_xaxes(title_text="Number of Components", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative Explained Variance", row=1, col=2)

        fig.update_layout(
            title="PCA Explained Variance Analysis",
            height=500,
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        if self._config.show:
            fig.show()

        fig.write_html(
            self._output_path / "explained_variance_per_principal_component.html"
        )
        self._log_artifact(
            self._output_path / "explained_variance_per_principal_component.html",
            "dim-reduction - explained_variance_per_principal_component",
        )

    def run(self) -> None:
        print(f"Starting dimension reduction using {self._config.method}...")
        train_data = self._injected_data.train.x
        train_labels = self._injected_data.train.y
        test_data = self._injected_data.test.x
        test_labels = self._injected_data.test.y

        reduced_train = None
        if self._config.method == ReductionMethod.PCA:
            reduced_train = self._apply_pca(train_data, self._config.n_components)

        elif self._config.method == ReductionMethod.TSNE:
            reduced_train = self._apply_tsne(train_data, self._config.n_components)

        elif self._config.method == ReductionMethod.UMAP:
            reduced_train = self._apply_umap(
                train_data, self._config.n_components, train_labels
            )

        elif self._config.method == ReductionMethod.LDA:
            if train_labels is None:
                raise ValueError("LDA requires labels for training data")
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
                columns=[f"LDA_{i+1}" for i in range(self._config.n_components)],
            )

        self._visualize_results(
            reduced_train,
            train_labels,
            f"Training Data - {self._config.method.upper()}",
        )

        if self._config.method == ReductionMethod.PCA:
            self._plot_explained_variance()

        print(f"Dimension reduction completed successfully!")
        print(f"Original shape: {train_data.shape}")
        print(f"Reduced shape: {reduced_train.shape}")

        if (
            self._config.method == ReductionMethod.PCA
            and "pca" in self._explained_variances
        ):
            explained = self._explained_variances["pca"]["explained_variance_ratio"]
            cumulative = self._explained_variances["pca"]["cumulative_variance"]
            print(f"Explained variance by components: {explained}")
            print(f"Cumulative variance: {cumulative[-1]:.4f}")

        self._result = PreprocessResult(
            train=LoadDsResult(x=reduced_train, y=train_labels),
            test=LoadDsResult(x=reduced_test, y=test_labels),
        )

        model_type_name = type(self._model).__name__.lower()
        self._log_model(f"{model_type_name}_model", model_type="sklearn")
