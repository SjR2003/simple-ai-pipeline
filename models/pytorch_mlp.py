from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from core.base_model import BaseMLModel


class PytorchMlp(BaseMLModel):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [64, 32],
        output_size: int = 2,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        **kwargs,
    ):
        super().__init__(model_name="PytorchMlp", **kwargs)

        self._input_size = input_size
        self._hidden_sizes = hidden_sizes
        self._output_size = output_size
        self._dropout_rate = dropout_rate
        self._learning_rate = learning_rate

        self._model = self.build_model()

        if output_size == 1:
            self._criterion = nn.BCEWithLogitsLoss()
        else:
            self._criterion = nn.CrossEntropyLoss()
        self._optimizer = optim.Adam(self._model.parameters(), lr=learning_rate)

        self._train_loss_history = []
        self._val_loss_history = []
        self._train_acc_history = []
        self._val_acc_history = []

    def build_model(self) -> nn.Module:
        layers = []

        layers.append(nn.Linear(self._input_size, self._hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self._dropout_rate))

        for i in range(len(self._hidden_sizes) - 1):
            layers.append(nn.Linear(self._hidden_sizes[i], self._hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self._dropout_rate))

        layers.append(nn.Linear(self._hidden_sizes[-1], self._output_size))

        return nn.Sequential(*layers)

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> dict:
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)

        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
            has_val = True
        else:
            has_val = False

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in tqdm(range(epochs), desc="epoch: "):
            self._model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                self._optimizer.zero_grad()

                outputs = self._model(batch_X)
                loss = self._criterion(outputs, batch_y)

                loss.backward()
                self._optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total

            self._train_loss_history.append(avg_train_loss)
            self._train_acc_history.append(train_accuracy)

            if has_val:
                val_loss, val_accuracy = self._validate(X_val_tensor, y_val_tensor)
                self._val_loss_history.append(val_loss)
                self._val_acc_history.append(val_accuracy)
            else:
                val_loss, val_accuracy = None, None

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                val_info = (
                    f", Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
                    if has_val
                    else ""
                )
                print(
                    f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}{val_info}"
                )

        self._is_trained = True

        history = {
            "train_loss": self._train_loss_history,
            "train_accuracy": self._train_acc_history,
        }

        if has_val:
            history["val_loss"] = self._val_loss_history
            history["val_accuracy"] = self._val_acc_history

        self._training_history = history
        return history

    def _validate(self, X_val, y_val):
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(X_val)
            loss = self._criterion(outputs, y_val)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_val).sum().item()
            accuracy = correct / y_val.size(0)

        return loss.item(), accuracy

    def predict(self, X) -> np.ndarray:
        if not self._is_trained and not self._is_loaded:
            raise ValueError(
                "please train model or use load model to load pre train model"
            )
        self._model.eval()

        X_tensor = torch.FloatTensor(X)

        with torch.no_grad():
            outputs = self._model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)

        return predicted.numpy()

    def save_model(self, filepath: str):
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "model_params": {
                    "input_size": self._input_size,
                    "hidden_sizes": self._hidden_sizes,
                    "output_size": self._output_size,
                    "dropout_rate": self._dropout_rate,
                    "learning_rate": self._learning_rate,
                },
                "training_history": self._training_history,
                "is_trained": self._is_trained,
            },
            filepath,
        )

    def load_model(self, filepath: str):
        self._model_loaded = True
        checkpoint = torch.load(filepath)

        model_params = checkpoint["model_params"]
        self._input_size = model_params["input_size"]
        self._hidden_sizes = model_params["hidden_sizes"]
        self._output_size = model_params["output_size"]
        self._dropout_rate = model_params["dropout_rate"]
        self._learning_rate = model_params["learning_rate"]

        self._model = self.build_model()
        self._model.load_state_dict(checkpoint["model_state_dict"])

        self._optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate)
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self._training_history = checkpoint["training_history"]
        self._is_trained = checkpoint["is_trained"]

    def get_params(self) -> dict:
        params = super().get_params()
        params.update(
            {
                "input_size": self._input_size,
                "hidden_sizes": self._hidden_sizes,
                "output_size": self._output_size,
                "dropout_rate": self._dropout_rate,
                "learning_rate": self._learning_rate,
            }
        )
        return params
