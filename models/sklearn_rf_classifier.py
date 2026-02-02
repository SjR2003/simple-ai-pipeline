from pathlib import Path
from copy import deepcopy
from typing import Optional, Dict, Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from core.base_model import BaseMLModel


class SklearnRfClassifier(BaseMLModel):
    def __init__(
        self,
        n_estimators: Optional[int] = None,
        criterion: Optional[str] = None,
        max_depth: Optional[int] = None,
        min_samples_split: Optional[Any] = None,
        min_samples_leaf: Optional[Any] = None,
        min_weight_fraction_leaf: Optional[float] = None,
        max_features: Optional[Any] = None,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: Optional[float] = None,
        bootstrap: Optional[bool] = None,
        oob_score: Optional[bool] = None,
        max_samples: Optional[Any] = None,
        ccp_alpha: Optional[float] = None,
        class_weight: Optional[Any] = None,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        verbose: Optional[int] = None,
        warm_start: Optional[bool] = None,
        **kwargs,
    ):
        base_allowed = {"model_name", "random_state"}
        base_kwargs = {k: v for k, v in kwargs.items() if k in base_allowed}

        super().__init__(model_name="RandomForestClassifier", **base_kwargs)

        rf_params = {
            "n_estimators": n_estimators,
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "min_weight_fraction_leaf": min_weight_fraction_leaf,
            "max_features": max_features,
            "max_leaf_nodes": max_leaf_nodes,
            "min_impurity_decrease": min_impurity_decrease,
            "bootstrap": bootstrap,
            "oob_score": oob_score,
            "max_samples": max_samples,
            "ccp_alpha": ccp_alpha,
            "class_weight": class_weight,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbose": verbose,
            "warm_start": warm_start,
        }

        self._model_params: Dict[str, Any] = {
            k: v for k, v in rf_params.items() if v is not None
        }

        valid_rf = set(RandomForestClassifier().get_params().keys())
        rf_from_kwargs = {k: v for k, v in kwargs.items() if k in valid_rf}

        dropped = set(kwargs) - set(rf_from_kwargs)
        if dropped:
            self._logger.warning(f"Ignored unknown kwargs for RF: {dropped}")

        self._model_params.update(rf_from_kwargs)

        self._model = self.build_model()
        self._is_trained = False
        self._is_loaded = False

        self._train_loss_history = []
        self._train_acc_history = []
        self._val_loss_history = []
        self._val_acc_history = []
        self._training_history = {}

    def build_model(self) -> BaseEstimator:
        return RandomForestClassifier(**self._model_params)

    def _get_predict_proba(self, X: np.ndarray):
        if hasattr(self._model, "predict_proba"):
            try:
                return self._model.predict_proba(X)
            except Exception:
                return None
        return None

    def _safe_log_loss(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        try:
            labels = (
                list(self._model.classes_) if hasattr(self._model, "classes_") else None
            )
            return log_loss(y_true, y_proba, labels=labels)
        except Exception:
            return float("nan")

    def _compute_metrics(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}

        try:
            y_pred = self._model.predict(X_train)
            metrics["train_accuracy"] = accuracy_score(y_train, y_pred)

            proba = self._get_predict_proba(X_train)
            metrics["train_loss"] = (
                self._safe_log_loss(y_train, proba) if proba is not None else np.nan
            )
        except Exception as e:
            self._logger.warning(f"Error computing train metrics: {e}")
            metrics["train_accuracy"] = 0.0
            metrics["train_loss"] = np.nan

        if X_val is not None and y_val is not None:
            try:
                y_pred = self._model.predict(X_val)
                metrics["val_accuracy"] = accuracy_score(y_val, y_pred)

                proba = self._get_predict_proba(X_val)
                metrics["val_loss"] = (
                    self._safe_log_loss(y_val, proba) if proba is not None else np.nan
                )
            except Exception as e:
                self._logger.warning(f"Error computing val metrics: {e}")
                metrics["val_accuracy"] = 0.0
                metrics["val_loss"] = np.nan

        return metrics

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 1,
        batch_size: int = 32,
        patience: int = 0,
        restore_best_weights: bool = False,
        monitor: str = "val_accuracy",
        verbose: bool = True,
        **kwargs,
    ) -> dict:
        self._train_loss_history = []
        self._train_acc_history = []
        self._val_loss_history = []
        self._val_acc_history = []

        model_type = self._model.__class__.__name__
        if verbose:
            self._logger.info(
                f"Training {model_type} (RandomForest wrapper) - fit once"
            )
            self._logger.info(
                "Note: epochs/early stopping are ignored for RandomForest."
            )

        self._model.fit(X_train, y_train, **kwargs)

        m = self._compute_metrics(X_train, y_train, X_val, y_val)

        self._train_loss_history.append(m.get("train_loss", np.nan))
        self._train_acc_history.append(m.get("train_accuracy", 0.0))

        if X_val is not None and y_val is not None:
            self._val_loss_history.append(m.get("val_loss", np.nan))
            self._val_acc_history.append(m.get("val_accuracy", 0.0))

        history = {
            "train_loss": self._train_loss_history,
            "train_accuracy": self._train_acc_history,
        }
        if X_val is not None and y_val is not None:
            history.update(
                {
                    "val_loss": self._val_loss_history,
                    "val_accuracy": self._val_acc_history,
                    "best_epoch": 0,
                    "best_val_loss": (
                        self._val_loss_history[0] if self._val_loss_history else None
                    ),
                }
            )

        self._is_trained = True
        self._training_history = history
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained and not self._is_loaded:
            raise ValueError("please train model or load a pre-trained model")
        return self._model.predict(X)

    @property
    def tunable_params(self) -> set:
        supported = set(self._model.get_params().keys())
        candidates = {
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "min_samples_split",
            "max_features",
            "bootstrap",
            "class_weight",
            "criterion",
            "max_leaf_nodes",
            "max_samples",
        }
        return supported.intersection(candidates)

    @property
    def supports_epochs(self) -> bool:
        return False

    def clone_with_params(self, params: dict) -> BaseMLModel:
        model_copy = deepcopy(self)

        fresh = clone(self._model)
        valid = set(fresh.get_params().keys())
        filtered = {k: v for k, v in params.items() if k in valid}

        dropped = set(params) - set(filtered)
        if dropped:
            model_copy._logger.warning(
                f"Dropped invalid params for {fresh.__class__.__name__}: {dropped}"
            )

        fresh.set_params(**filtered)
        model_copy._model = fresh

        model_copy._model_params = {
            **getattr(model_copy, "_model_params", {}),
            **filtered,
        }

        model_copy._is_trained = False
        model_copy._is_loaded = False
        model_copy._training_history = {}
        model_copy._train_loss_history = []
        model_copy._train_acc_history = []
        model_copy._val_loss_history = []
        model_copy._val_acc_history = []
        return model_copy

    def save_model(self, filepath: Path) -> str:
        filepath = Path(filepath)
        if filepath.suffix != ".joblib":
            filepath = filepath.with_suffix(".joblib")
        joblib.dump(self._model, filepath)
        return str(filepath)

    def load_model(self, filepath: str) -> None:
        filepath = Path(filepath)
        self._model = joblib.load(filepath)
        self._is_loaded = True
        self._is_trained = True

    def get_params(self) -> dict:
        params = super().get_params()
        params.update(
            {
                "sklearn_model_class": self._model.__class__.__name__,
                "model_params": self._model.get_params(),
            }
        )
        return params
