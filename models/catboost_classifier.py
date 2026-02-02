from typing import Optional, Dict, Any
from pathlib import Path
import joblib
import numpy as np

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss

from core.base_model import BaseMLModel


class CatboostClassifier(BaseMLModel):
    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.05,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        loss_function: str = "MultiClass",
        random_state: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(model_name="CatBoostClassifier", random_state=random_state)

        self._model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            loss_function=loss_function,
            random_seed=random_state,
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

    def _safe_log_loss(self, y_true, y_proba):
        try:
            return log_loss(y_true, y_proba)
        except Exception:
            return float("nan")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs,
    ) -> dict:
        self._train_loss_history.clear()
        self._train_acc_history.clear()
        self._val_loss_history.clear()
        self._val_acc_history.clear()

        self._logger.info("Training CatBoostClassifier - fit once")

        if X_val is not None and y_val is not None:
            self._model.fit(X_train, y_train, eval_set=(X_val, y_val))
        else:
            self._model.fit(X_train, y_train)

        y_pred = self._model.predict(X_train)
        y_proba = self._model.predict_proba(X_train)

        self._train_acc_history.append(accuracy_score(y_train, y_pred))
        self._train_loss_history.append(self._safe_log_loss(y_train, y_proba))

        has_val = X_val is not None and y_val is not None
        if has_val:
            y_pred = self._model.predict(X_val)
            y_proba = self._model.predict_proba(X_val)

            self._val_acc_history.append(accuracy_score(y_val, y_pred))
            self._val_loss_history.append(self._safe_log_loss(y_val, y_proba))

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

        self._training_history = history
        self._is_trained = True
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained and not self._is_loaded:
            raise RuntimeError("Model not trained")
        return self._model.predict(X)

    @property
    def tunable_params(self) -> set:
        return {
            "iterations",
            "epochs",
            "learning_rate",
            "depth",
            "max_depth",
            "l2_leaf_reg",
        }

    @property
    def supports_epochs(self) -> bool:
        return False

    def clone_with_params(self, params: Dict[str, Any]) -> BaseMLModel:
        base = self.get_params()
        base.update(params)
        return CatboostClassifier(**base)

    def save_model(self, path: Path) -> str:
        path = Path(path).with_suffix(".joblib")
        joblib.dump(self._model, path)
        return str(path)

    def load_model(self, path: str) -> None:
        self._model = joblib.load(path)
        self._is_loaded = True
        self._is_trained = True
