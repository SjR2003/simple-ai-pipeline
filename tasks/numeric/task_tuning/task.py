import time
import shutil
import mlflow
import optuna
import datetime
import numpy as np
from pathlib import Path
from mlflow.models.signature import infer_signature
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler

from core.base_task import BaseTask
from core.base_model import BaseMLModel
from core.task_registry import register_task
from tasks.numeric.task_training.modules.plotter import *
from tasks.numeric.task_tuning.config import TuningConfig
from tasks.numeric.task_training.result_schema import TrainResult

current_state = datetime.datetime.now()
saved_state = None


@register_task("tuning")
class Tuning(BaseTask):
    def __init__(self, config: dict, input_data: any):
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

        self._run_num = 0
        result_dir = f"result_{self._run_num}"

        if self._output_path.exists() and not self._new_run:
            self._run_num = self._get_last_folder()
            result_dir = f"result_{self._run_num}"

        self._output_path = self._output_path / result_dir
        self._output_path.mkdir(exist_ok=True, parents=True)

        self._model = None

    def _validate_config(self) -> None:
        self._config = TuningConfig.model_validate(self._config_dict)

    def _load_data(self) -> None:
        self._train_result = self._injected_data

    @property
    def result(self) -> TrainResult:
        return self._result

    def _log_inference_time(
        self,
        model: BaseMLModel,
        X: np.ndarray,
        n_runs: int = 50,
        prefix: str = "inference",
    ):
        times = []

        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model.predict(X)
            times.append(time.perf_counter() - start)

        times = np.array(times)

        self._log_metric(f"{prefix}_mean_ms", times.mean() * 100)
        self._log_metric(f"{prefix}_p95_ms", np.percentile(times, 95) * 100)
        self._log_metric(f"{prefix}_max_ms", times.max() * 100)

    def _sanitize_best_params(self, best_params: dict) -> dict:
        params = {}

        if "num_layers" in best_params:
            n_layers = best_params["num_layers"]
            hidden_sizes = [
                best_params[f"layer_{i}_size"]
                for i in range(n_layers)
                if f"layer_{i}_size" in best_params
            ]
            if hidden_sizes:
                params["hidden_sizes"] = hidden_sizes

        model_params = {}
        train_params = {}

        for k, v in best_params.items():
            if k in ["num_layers"] or k.startswith("layer_"):
                continue
            elif k in ["epochs", "batch_size"]:
                train_params[k] = v
            else:
                model_params[k] = v

        return {"model_params": model_params, "train_params": train_params, **params}

    def run(self) -> None:
        base_model = self._train_result.model
        train_data = self._train_result.data.train
        test_data = self._train_result.data.test
        X_train = train_data.x.to_numpy()
        y_train = train_data.y.to_numpy()
        X_test = test_data.x.to_numpy()
        y_test = test_data.y.to_numpy()

        metric_name = self._config.metric
        direction = self._config.direction

        def _space_intersection(param_space: dict, allowed: set) -> dict:
            return {k: v for k, v in param_space.items() if k in allowed}

        def objective(trial):
            model_params = {}
            train_params = {}

            tunable = set(base_model.tunable_params)
            param_space = _space_intersection(self._config.param_space, tunable)

            if (
                "num_layers" in param_space.keys()
                and "layer_size" in param_space.keys()
            ):
                num_layers_config = param_space.get("num_layers")
                layer_size_config = param_space.get("layer_size")
                n_layers = trial.suggest_int(
                    "num_layers",
                    num_layers_config["low"],
                    num_layers_config["high"],
                )
                hidden_sizes = [
                    trial.suggest_int(
                        f"layer_{i}_size",
                        layer_size_config["low"],
                        layer_size_config["high"],
                        step=layer_size_config["step"],
                    )
                    for i in range(n_layers)
                ]
                model_params["hidden_sizes"] = hidden_sizes

            if "dropout_rate" in param_space.keys():
                dropout_config = param_space.get("dropout_rate")
                model_params["dropout_rate"] = trial.suggest_float(
                    "dropout_rate",
                    dropout_config["low"],
                    dropout_config["high"],
                )

            if "learning_rate" in param_space.keys():
                learning_rate_config = param_space.get("learning_rate")
                model_params["learning_rate"] = trial.suggest_float(
                    "learning_rate",
                    learning_rate_config["low"],
                    learning_rate_config["high"],
                    log=True,
                )

            if "n_estimators" in param_space.keys():
                n_estimators_config = param_space.get("n_estimators")
                model_params["n_estimators"] = trial.suggest_int(
                    "n_estimators",
                    n_estimators_config["low"],
                    n_estimators_config["high"],
                    step=n_estimators_config["step"],
                )

            if "min_samples_leaf" in param_space.keys():
                min_samples_leaf_config = param_space.get("min_samples_leaf")
                model_params["min_samples_leaf"] = trial.suggest_int(
                    "min_samples_leaf",
                    min_samples_leaf_config["low"],
                    min_samples_leaf_config["high"],
                    step=min_samples_leaf_config["step"],
                )

            if "max_depth" in param_space.keys():
                max_depth_config = param_space.get("max_depth")
                model_params["max_depth"] = trial.suggest_int(
                    "max_depth",
                    max_depth_config["low"],
                    max_depth_config["high"],
                )

                model_params["depth"] = model_params["max_depth"]

            if "l2_regularization" in param_space.keys():
                l2_regularization_config = param_space.get("l2_regularization")
                model_params["l2_regularization"] = trial.suggest_float(
                    "l2_regularization",
                    l2_regularization_config["low"],
                    l2_regularization_config["high"],
                )

            if "epochs" in param_space.keys():
                epochs_config = param_space.get("epochs")
                if epochs_config:
                    train_params["epochs"] = trial.suggest_int(
                        "epochs",
                        epochs_config["low"],
                        epochs_config["high"],
                        step=epochs_config.get("step", 20),
                    )
                    model_params["iterations"] = train_params["epochs"]

            if "batch_size" in param_space.keys():
                batch_size_config = param_space.get("batch_size")
                if batch_size_config:
                    train_params["batch_size"] = trial.suggest_int(
                        "batch_size",
                        batch_size_config["low"],
                        batch_size_config["high"],
                        step=batch_size_config.get("step", 4),
                    )

            with mlflow.start_run(nested=True):
                all_params = {**model_params, **train_params}
                mlflow.log_params(all_params)
                model = base_model.clone_with_params(model_params)

                history = model.train(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    epochs=train_params.get("epochs"),
                    batch_size=train_params.get("batch_size"),
                )
                metrics = model.eval(X_test, y_test)

            return metrics[metric_name]

        sampler = TPESampler(seed=self._seed)
        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(objective, n_trials=self._config.n_trials)

        best_params = study.best_params
        cleaned_params = self._sanitize_best_params(best_params)

        model_params = cleaned_params.get("model_params", {})
        train_params = cleaned_params.get("train_params", {})

        if "hidden_sizes" in cleaned_params:
            model_params["hidden_sizes"] = cleaned_params["hidden_sizes"]

        all_params = {**model_params, **train_params}
        prefix = "tune_"
        prefixed_dict = {f"{prefix}{key}": value for key, value in all_params.items()}
        self._log_params(prefixed_dict)
        self._model = base_model.clone_with_params(model_params)

        history = self._model.train(X_train, y_train, X_test, y_test, **train_params)
        loss_plot_path, acc_plot_path = plot_history(history, self._output_path)
        self._log_artifact(loss_plot_path, "tune - loss curve")
        self._log_artifact(acc_plot_path, "tune - accuracy curve")

        best_metrics = self._model.eval(X_test, y_test)

        confusion_plot_path = plot_confusion_matrix(
            best_metrics["confusion"], self._output_path, random_state=self._seed
        )
        self._log_artifact(confusion_plot_path, "tune - confusion matrix")
        per_class_path = plot_per_class_metrics(
            best_metrics["per_class_metrics"], self._output_path
        )
        for key, value in per_class_path.items():
            self._log_artifact(value, f"tune - {key}")

        self._log_metric("tune - accuracy", best_metrics["accuracy"])
        self._log_metric("tune - f1_score", best_metrics["f1_score"])
        self._log_metric("tune - recall", best_metrics["recall"])
        self._log_metric("tune - precision", best_metrics["precision"])

        self._log_inference_time(self._model, X_test)

        self._model.save_model(self._output_path / f"{self._train_result.model_name}")
        signature = infer_signature(X_train, y_train)
        self._log_model(
            f"tune_{self._train_result.model_name}",
            model_type=self._train_result.model_type,
            signature=signature,
        )

        self._result = TrainResult(
            model=self._model,
            metrics=best_metrics,
            model_type=self._train_result.model_type,
            data=self._train_result.data,
            model_params=best_params,
            model_name="tuned_" + self._train_result.model_name,
        )
