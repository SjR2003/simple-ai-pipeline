import numpy as np
import joblib
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

from core.base_model import BaseMLModel


class SklearnClassifier(BaseMLModel):
    def __init__(self, sklearn_model: BaseEstimator, **kwargs):
        model_name = f"Sklearn_{sklearn_model.__class__.__name__}"
        super().__init__(model_name=model_name, **kwargs)

        self.sklearn_model = sklearn_model
        self._model = self.build_model()

    def build_model(self) -> BaseEstimator:
        return self.sklearn_model

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> dict:
        self._model.fit(X_train, y_train, **kwargs)
        self._is_trained = True
        y_train_pred = self.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        if X_val is not None and y_val is not None:
            y_val_pred = self.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
        else:
            val_accuracy = None

        history = {"train_accuracy": train_accuracy, "val_accuracy": val_accuracy}

        self._training_history = history
        return history

    def predict(self, X) -> np.ndarray:
        if not self._is_trained and not self._is_loaded:
            raise ValueError(
                "please train model or use load model to load pre train model"
            )

        return self._model.predict(X)

    def save_model(self, filepath: str):
        joblib.dump(self._model, filepath)

    def load_model(self, filepath: str):
        self._is_loaded = True
        self._model = joblib.load(filepath)
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
