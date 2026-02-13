import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import numpy as np
from typing import Dict, Any, List


def plot_history(history, output_path: Path):
    """
    Plot training history and save as HTML files.

    Args:
        history: Dictionary with keys:
            - "train_loss": list of training losses per epoch
            - "train_accuracy": list of training accuracies per epoch
            - "val_loss": list of validation losses per epoch (optional)
            - "val_accuracy": list of validation accuracies per epoch (optional)
        output_path: Path object or string where plots should be saved
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    train_len = len(history["train_loss"])
    epochs = list(range(1, train_len + 1))

    has_val = "val_loss" in history

    fig_loss = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=("Training and Validation Loss" if has_val else "Training Loss"),
    )

    fig_loss.add_trace(
        go.Scatter(
            x=epochs,
            y=history["train_loss"],
            mode="lines+markers",
            name="Train Loss",
            line=dict(color="blue", width=2),
            marker=dict(size=6),
        ),
        row=1,
        col=1,
    )

    if has_val and history["val_loss"]:
        fig_loss.add_trace(
            go.Scatter(
                x=epochs[: len(history["val_loss"])],
                y=history["val_loss"],
                mode="lines+markers",
                name="Val Loss",
                line=dict(color="red", width=2, dash="dash"),
                marker=dict(size=6),
            ),
            row=1,
            col=1,
        )

        if "best_epoch" in history and history["best_epoch"] is not None:
            fig_loss.add_vline(
                x=history["best_epoch"] + 1,
                line_dash="dot",
                line_color="green",
                opacity=0.7,
                annotation_text=f"Best Epoch: {history['best_epoch'] + 1}",
                annotation_position="top right",
            )

    fig_loss.update_layout(
        title_text="Loss Curves",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend_title="Legend",
        hovermode="x unified",
    )

    loss_plot_path = output_path / "loss_plot.html"
    fig_loss.write_html(loss_plot_path)

    if "train_accuracy" in history and history["train_accuracy"]:
        fig_acc = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=(
                "Training and Validation Accuracy" if has_val else "Training Accuracy"
            ),
        )

        fig_acc.add_trace(
            go.Scatter(
                x=epochs,
                y=history["train_accuracy"],
                mode="lines+markers",
                name="Train Accuracy",
                line=dict(color="green", width=2),
                marker=dict(size=6),
            ),
            row=1,
            col=1,
        )

        if has_val and "val_accuracy" in history and history["val_accuracy"]:
            fig_acc.add_trace(
                go.Scatter(
                    x=epochs[: len(history["val_accuracy"])],
                    y=history["val_accuracy"],
                    mode="lines+markers",
                    name="Val Accuracy",
                    line=dict(color="orange", width=2, dash="dash"),
                    marker=dict(size=6),
                ),
                row=1,
                col=1,
            )

            if "best_epoch" in history and history["best_epoch"] is not None:
                fig_acc.add_vline(
                    x=history["best_epoch"] + 1,
                    line_dash="dot",
                    line_color="green",
                    opacity=0.7,
                    annotation_text=f"Best Epoch: {history['best_epoch'] + 1}",
                    annotation_position="top right",
                )

        fig_acc.update_layout(
            title_text="Accuracy Curves",
            xaxis_title="Epoch",
            yaxis_title="Accuracy",
            legend_title="Legend",
            hovermode="x unified",
        )

        acc_plot_path = output_path / "accuracy_plot.html"
        fig_acc.write_html(acc_plot_path)
    else:
        acc_plot_path = None

    return loss_plot_path, acc_plot_path


def plot_confusion_matrix(
    confusion: Dict[str, Any],
    output_path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    random_state: int = 42,
):
    np.random.seed(random_state)
    labels = confusion["labels"]
    cm = np.array(confusion["matrix"], dtype=float)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Predicted label",
        yaxis_title="True label",
    )

    fig_path = output_path / "confusion_matrix.html"
    fig.write_html(fig_path)
    return fig_path


def plot_per_class_metrics(
    per_class_metrics: Dict[str, Any],
    output_path: Path,
    metrics: List[str] = ("precision", "recall", "f1_score", "accuracy_ovr"),
    title_prefix: str = "Per-class",
) -> Dict[str, Path]:
    """
    Plot one Plotly bar chart per metric, with bars for each class.

    per_class_metrics: results["per_class_metrics"]
    """

    labels = per_class_metrics["labels"]
    output_path_dict = {}
    for metric in metrics:
        values = [per_class_metrics[metric][str(label)] for label in labels]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=values,
                )
            ]
        )

        fig.update_layout(
            title=f"{title_prefix} {metric.replace('_', ' ').title()}",
            xaxis_title="Class",
            yaxis_title=metric.replace("_", " ").title(),
            yaxis=dict(range=[0, 1]),
            bargap=0.25,
        )

        current_output_path = output_path / f"per_class_{metric}.html"
        fig.write_html(current_output_path)
        output_path_dict[metric] = current_output_path

    return output_path_dict
