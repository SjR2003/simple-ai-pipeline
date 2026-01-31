import abc
from typing import Any, Dict, List
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class BaseMLModel(abc.ABC):
    def __init__(self, model_name: str = "BaseModel", random_state: int = 42):
        self._model_name = model_name
        self._random_state = random_state
        self._model = None
        self._is_trained = False
        self._is_loaded = False
        self._training_history = []

        self._set_random_seeds()

    def _set_random_seeds(self):
        np.random.seed(self._random_state)
        torch.manual_seed(self._random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self._random_state)

    @abc.abstractmethod
    def build_model(self, **kwargs) -> Any:
        raise NotImplementedError(" please implement this method")

    @abc.abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> Dict:
        raise NotImplementedError(" please implement this method")

    @abc.abstractmethod
    def predict(self, X) -> np.ndarray:
        raise NotImplementedError(" please implement this method")

    def eval(self, X_test, y_test, metrics: List[str] = None) -> dict:
        if not self._is_trained and not self._is_loaded:
            raise ValueError(
                "please train model or use load model to load pre train model"
            )

        y_pred = self.predict(X_test)

        results = {}
        default_metrics = ["accuracy", "precision", "recall", "f1"]
        if metrics is None:
            metrics = default_metrics

        y_test_np = np.array(y_test)
        y_pred_np = np.array(y_pred)

        for metric in metrics:
            if metric == "accuracy":
                results["accuracy"] = accuracy_score(y_test_np, y_pred_np)
            elif metric == "precision":
                results["precision"] = precision_score(
                    y_test_np, y_pred_np, average="weighted", zero_division=0
                )
            elif metric == "recall":
                results["recall"] = recall_score(
                    y_test_np, y_pred_np, average="weighted", zero_division=0
                )
            elif metric == "f1":
                results["f1"] = f1_score(
                    y_test_np, y_pred_np, average="weighted", zero_division=0
                )

        return results

    def save_model(self, filepath: str):
        raise NotImplementedError(" please implement this method")

    def load_model(self, filepath: str):
        self._is_loaded = True
        raise NotImplementedError(" please implement this method")

    def get_params(self) -> Dict:
        return {
            "model_name": self._model_name,
            "random_state": self._random_state,
            "is_trained": self._is_trained,
            "is_loaded": self._is_loaded,
        }

    def __str__(self):
        return f"{self._model_name} (Trained: {self._is_trained})"
