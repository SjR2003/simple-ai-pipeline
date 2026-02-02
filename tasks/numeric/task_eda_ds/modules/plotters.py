import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_missing(missing_stats: dict, output_path: Path, run_id: int) -> Path:
    fig_nan = px.histogram(
        missing_stats["per_feature_ratio"],
        nbins=50,
        title=f"Missing Ratio per Feature {run_id}",
    )

    nan_html = output_path / f"missing_ratio_{run_id}.html"
    fig_nan.write_html(nan_html)
    return nan_html


def plot_class_dist(df_y: pd.DataFrame, output_path: Path, run_id: int) -> Path:
    fig_cls = px.bar(
        df_y,
        x="class",
        y="count",
        title=f"Class Distribution (EDA) {run_id}",
    )

    cls_html = output_path / f"class_distribution_{run_id}.html"
    fig_cls.write_html(cls_html)
    return cls_html


def plot_variance(X: np.ndarray, output_path: Path, run_id: int) -> Path:
    fig_var = px.histogram(
        np.nanvar(X, axis=0),
        nbins=50,
        title=f"Feature Variance Distribution {run_id}",
    )

    var_html = output_path / f"feature_variance_{run_id}.html"
    fig_var.write_html(var_html)
    return var_html


def plot_correlation_matrix(
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Path,
    run_id: int,
    max_features: int = 50,
    random_state: int = 42,
) -> Path:
    np.random.seed(random_state)

    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_values = X.values
    else:
        X_values = X
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_values = y.values.ravel()
        target_name = y.name if hasattr(y, "name") and y.name is not None else "Target"
    else:
        y_values = y.ravel() if hasattr(y, "reshape") else y
        target_name = "Target"

    if X_values.shape[0] != len(y_values):
        raise ValueError(
            f"Number of samples in X ({X_values.shape[0]}) "
            f"does not match number of samples in y ({len(y_values)})"
        )

    if X_values.shape[1] > max_features:
        variances = np.var(X_values, axis=0)
        top_indices = np.argsort(variances)[-max_features:]
        X_subset = X_values[:, top_indices]
        selected_names = [feature_names[i] for i in top_indices]
    else:
        X_subset = X_values
        selected_names = feature_names

    data_with_target = pd.DataFrame(X_subset, columns=selected_names)
    data_with_target[target_name] = y_values

    corr_matrix = data_with_target.corr().values
    all_names = selected_names + [target_name]

    n_features = len(all_names)
    base_size = 40
    min_size = 800
    max_size = 2000

    plot_size = max(min_size, min(max_size, n_features * base_size))

    corr_text = np.round(corr_matrix, 2)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix,
            x=all_names,
            y=all_names,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Correlation"),
            text=corr_text,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate="<b>X</b>: %{x}<br><b>Y</b>: %{y}<br><b>Correlation</b>: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Correlation Matrix (Run {run_id}) - {n_features} variables",
        xaxis_title="Variables",
        yaxis_title="Variables",
        width=plot_size,
        height=plot_size,
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(all_names))),
            ticktext=all_names,
            tickangle=45,
            tickfont=dict(size=10),
            side="top",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(all_names))),
            ticktext=all_names,
            tickfont=dict(size=10),
            autorange="reversed",
        ),
        margin=dict(l=100, r=50, t=100, b=100),
    )

    html_path = output_path / f"correlation_matrix_{run_id}.html"
    fig.write_html(str(html_path))

    return html_path


def plot_feature_importance(
    importance_stats: dict, output_path: Path, run_id: int
) -> Path:
    fig = make_subplots(
        rows=2, cols=1, subplot_titles=("F-Scores (ANOVA)", "Mutual Information Scores")
    )

    fig.add_trace(
        go.Bar(y=importance_stats["f_scores"], name="F-Score", marker_color="skyblue"),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(y=importance_stats["mi_scores"], name="MI Score", marker_color="salmon"),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"Feature Importance Scores (Run {run_id})", height=800, showlegend=False
    )

    fig.update_xaxes(title_text="Feature Index", row=2, col=1)

    html_path = output_path / f"feature_importance_{run_id}.html"
    fig.write_html(str(html_path))

    return html_path


def plot_outliers_distribution(
    outlier_stats: dict, X: np.ndarray, output_path: Path, run_id: int
) -> Path:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Outlier Ratio per Feature",
            "Box Plot of Top 4 Features",
            "Outlier Distribution",
            "Feature vs Outlier Count",
        ),
    )

    outlier_ratios = outlier_stats["outlier_ratio_per_feature"]
    fig.add_trace(
        go.Bar(
            x=list(range(len(outlier_ratios))), y=outlier_ratios, name="Outlier Ratio"
        ),
        row=1,
        col=1,
    )

    outlier_indices = np.argsort(outlier_ratios)[-4:]
    for i, idx in enumerate(outlier_indices):
        fig.add_trace(
            go.Box(y=X[:, idx], name=f"Feature {idx}", boxpoints="outliers"),
            row=1,
            col=2,
        )

    fig.add_trace(
        go.Histogram(x=outlier_ratios, nbinsx=20, name="Outlier Distribution"),
        row=2,
        col=1,
    )

    variances = np.var(X, axis=0)
    fig.add_trace(
        go.Scatter(
            x=variances, y=outlier_ratios, mode="markers", name="Variance vs Outliers"
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=f"Outlier Analysis (Run {run_id})", height=900, showlegend=False
    )

    html_path = output_path / f"outlier_analysis_{run_id}.html"
    fig.write_html(str(html_path))

    return html_path


def plot_feature_distributions_by_class(
    X: np.ndarray,
    y: np.ndarray,
    output_path: Path,
    run_id: int,
    n_features: int = 6,
    random_state: int = 42,
) -> Path:
    np.random.seed(random_state)
    unique_classes = np.unique(y)

    variances = np.var(X, axis=0)
    top_indices = np.argsort(variances)[-n_features:]

    fig = make_subplots(
        rows=n_features,
        cols=1,
        subplot_titles=[f"Feature {idx}" for idx in top_indices],
    )

    for i, idx in enumerate(top_indices):
        for cls in unique_classes:
            mask = y == cls
            fig.add_trace(
                go.Histogram(
                    x=X[mask, idx],
                    name=f"Class {cls}",
                    opacity=0.7,
                    legendgroup=f"Class {cls}",
                    showlegend=(i == 0),
                ),
                row=i + 1,
                col=1,
            )

    fig.update_layout(
        title=f"Feature Distributions by Class (Run {run_id})",
        height=300 * n_features,
        barmode="overlay",
    )

    html_path = output_path / f"feature_distributions_{run_id}.html"
    fig.write_html(str(html_path))

    return html_path
