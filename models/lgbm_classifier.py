from typing import Optional, Dict, Any
from pathlib import Path
import joblib
import numpy as np

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss

from core.base_model import BaseMLModel


class LgbmClassifier(BaseMLModel):
    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = -1,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        class_weight: Optional[Any] = None,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        verbose: int = -1,
        **kwargs,
    ):
        super().__init__(model_name="LGBMClassifier", random_state=random_state)

        self._model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        self._is_trained = False
        self._is_loaded = False

        self._train_loss_history = []
        self._train_acc_history = []
        self._val_loss_history = []
        self._val_acc_history = []
        self._training_history = {}

    def build_model(self) -> Any:
        return self._model

    def _safe_log_loss(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        try:
            labels = None
            if hasattr(self._model, "classes_"):
                labels = list(self._model.classes_)
            return log_loss(y_true, y_proba, labels=labels)
        except Exception:
            return float("nan")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 1,
        batch_size: int = 32,
        patience: int = 0,
        restore_best_weights: bool = False,
        monitor: str = "val_loss",
        verbose: bool = True,
        **kwargs,
    ) -> dict:
        self._train_loss_history = []
        self._train_acc_history = []
        self._val_loss_history = []
        self._val_acc_history = []

        if verbose:
            self._logger.info("Training LGBMClassifier - fit once")
            self._logger.info(
                "Note: epochs/early stopping are ignored for LightGBM in this wrapper."
            )

        self._model.fit(X_train, y_train, **kwargs)

        y_pred_train = self._model.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)

        train_proba = self._model.predict_proba(X_train)
        train_loss = self._safe_log_loss(y_train, train_proba)

        self._train_acc_history.append(train_acc)
        self._train_loss_history.append(train_loss)

        has_val = X_val is not None and y_val is not None
        if has_val:
            y_pred_val = self._model.predict(X_val)
            val_acc = accuracy_score(y_val, y_pred_val)

            val_proba = self._model.predict_proba(X_val)
            val_loss = self._safe_log_loss(y_val, val_proba)

            self._val_acc_history.append(val_acc)
            self._val_loss_history.append(val_loss)

        history = {
            "train_loss": self._train_loss_history,
            "train_accuracy": self._train_acc_history,
        }

        if has_val:
            history.update(
                {
                    "val_loss": self._val_loss_history,
                    "val_accuracy": self._val_acc_history,
                    "best_epoch": 0,
                    "best_val_loss": self._val_loss_history[0],
                }
            )

        self._is_trained = True
        self._training_history = history
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained and not self._is_loaded:
            raise RuntimeError("Model is not trained")
        return self._model.predict(X)

    @property
    def tunable_params(self) -> set:
        return {
            "n_estimators",
            "learning_rate",
            "max_depth",
            "num_leaves",
            "subsample",
            "colsample_bytree",
            "class_weight",
        }

    @property
    def supports_epochs(self) -> bool:
        return False

    def clone_with_params(self, params: Dict[str, Any]) -> BaseMLModel:
        base = self.get_params()
        base.update(params)
        return LgbmClassifier(**base)

    def save_model(self, path: Path) -> str:
        path = Path(path).with_suffix(".joblib")
        joblib.dump(self._model, path)
        return str(path)

    def load_model(self, path: str) -> None:
        self._model = joblib.load(path)
        self._is_loaded = True
        self._is_trained = True
