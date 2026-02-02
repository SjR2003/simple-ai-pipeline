import pandas as pd
from pathlib import Path
from typing import Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_results(
    reduced_data: pd.DataFrame,
    method: str,
    output_path: Path,
    y: Optional[pd.Series] = None,
    show: bool = False,
    title: str = "Dimension Reduction",
) -> Path:
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
        title=f"Dimension Reduction using {method}",
        height=500,
        showlegend=False,
        template="plotly_white",
    )

    if show:
        fig.show()

    output_path = output_path / "dimension_reduction.html"
    fig.write_html(output_path)
    return output_path


def plot_explained_variance(
    explained_variances: dict, output_path: Path, show: bool = False
) -> Path:
    if "pca" not in explained_variances:
        return None

    explained_variance = explained_variances["pca"]["explained_variance_ratio"]
    cumulative_variance = explained_variances["pca"]["cumulative_variance"]

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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if show:
        fig.show()

    output_path = output_path / "explained_variance_per_principal_component.html"
    fig.write_html(output_path)
    return output_path
