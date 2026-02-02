import abc
import torch
import logging
import numpy as np
from typing import Any, Dict, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


class BaseMLModel(abc.ABC):
    def __init__(self, model_name: str = "BaseModel", random_state: int = 42):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._model_name = model_name
        self._random_state = random_state
        self._model = None
        self._is_trained = False
        self._is_loaded = False
        self._training_history = {}

        self._train_loss_history = []
        self._val_loss_history = []
        self._train_acc_history = []
        self._val_acc_history = []

        self._set_random_seeds()

    def _set_random_seeds(self):
        np.random.seed(self._random_state)
        torch.manual_seed(self._random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self._random_state)
            torch.cuda.manual_seed_all(self._random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)

    @abc.abstractmethod
    def build_model(self, **kwargs) -> Any:
        raise NotImplementedError(" please implement this method")

    @abc.abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> Dict:
        raise NotImplementedError(" please implement this method")

    @abc.abstractmethod
    def predict(self, X) -> np.ndarray:
        raise NotImplementedError(" please implement this method")

    @abc.abstractmethod
    def clone_with_params(self, params: dict):
        raise NotImplementedError(" please implement this method")

    @property
    def tunable_params(self) -> set:
        return set()

    def eval(self, X_test, y_test, metrics: List[str] = None) -> Dict[str, Any]:
        if not self._is_trained and not self._is_loaded:
            raise ValueError(
                "please train model or use load model to load pre train model"
            )

        y_pred = self.predict(X_test)

        results: Dict[str, Any] = {}
        default_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "confusion",
            "per_class_metrics",
        ]
        if metrics is None:
            metrics = default_metrics

        y_test_np = np.asarray(y_test)
        y_pred_np = np.asarray(y_pred)

        labels = np.unique(np.concatenate([y_test_np, y_pred_np], axis=0))

        for metric in metrics:
            if metric == "accuracy":
                results["accuracy"] = accuracy_score(y_test_np, y_pred_np)

            elif metric == "precision":
                results["precision"] = precision_score(
                    y_test_np, y_pred_np, average="macro", zero_division=0
                )

            elif metric == "recall":
                results["recall"] = recall_score(
                    y_test_np, y_pred_np, average="macro", zero_division=0
                )

            elif metric == "f1_score":
                results["f1_score"] = f1_score(
                    y_test_np, y_pred_np, average="macro", zero_division=0
                )

            elif metric == "confusion":
                cm = confusion_matrix(y_test_np, y_pred_np, labels=labels)
                results["confusion"] = {
                    "labels": labels.tolist(),
                    "matrix": cm.tolist(),
                }

            elif metric == "per_class_metrics":
                p, r, f1, support = precision_recall_fscore_support(
                    y_test_np, y_pred_np, labels=labels, average=None, zero_division=0
                )

                per_class_acc = {}
                for i, lab in enumerate(labels):
                    tp = int(((y_test_np == lab) & (y_pred_np == lab)).sum())
                    tn = int(((y_test_np != lab) & (y_pred_np != lab)).sum())
                    per_class_acc[str(lab)] = (
                        (tp + tn) / len(y_test_np) if len(y_test_np) else 0.0
                    )

                results["per_class_metrics"] = {
                    "labels": labels.tolist(),
                    "precision": {
                        str(lab): float(p[i]) for i, lab in enumerate(labels)
                    },
                    "recall": {str(lab): float(r[i]) for i, lab in enumerate(labels)},
                    "f1_score": {
                        str(lab): float(f1[i]) for i, lab in enumerate(labels)
                    },
                    "support": {
                        str(lab): int(support[i]) for i, lab in enumerate(labels)
                    },
                    "accuracy_ovr": per_class_acc,
                }

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
