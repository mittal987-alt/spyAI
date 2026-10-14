
from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.config import CHECKPOINT_DIR, EARLY_STOPPING_PATIENCE


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class DiagnosticNetwork(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_units: int = 64,
        num_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def create_synthetic_dataset(
    n_samples: int = 3000,
    n_features: int = 20,
    random_state: int = 42,
) -> dict:
    if n_samples < 100:
        raise ValueError("n_samples must be at least 100")
    if n_features < 4:
        raise ValueError("n_features must be at least 4")

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=max(0, n_features // 5),
        n_repeated=0,
        n_classes=2,
        random_state=random_state,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.long),
        "X_val": torch.tensor(X_val, dtype=torch.float32),
        "y_val": torch.tensor(y_val, dtype=torch.long),
        "feature_count": n_features,
        "num_classes": 2,
    }


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in loader:
            logits = model(features)
            loss = criterion(logits, labels)

            count = labels.size(0)
            total_loss += loss.item() * count
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += count

    if total == 0:
        raise ValueError("Cannot evaluate an empty dataset")

    return total_loss / total, correct / total


def train_model(
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    random_state: int = 42,
    n_samples: int = 3000,
    n_features: int = 20,
    hidden_units: int = 64,
    dropout: float = 0.2,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
) -> dict:
    if epochs < 2:
        raise ValueError("epochs must be at least 2")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if hidden_units < 2:
        raise ValueError("hidden_units must be at least 2")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive")

    set_seed(random_state)
    started_at = time.perf_counter()

    data = create_synthetic_dataset(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
    )

    generator = torch.Generator()
    generator.manual_seed(random_state)

    train_loader = DataLoader(
        TensorDataset(data["X_train"], data["y_train"]),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        TensorDataset(data["X_val"], data["y_val"]),
        batch_size=batch_size,
        shuffle=False,
    )

    model = DiagnosticNetwork(
        input_features=n_features,
        hidden_units=hidden_units,
        num_classes=data["num_classes"],
        dropout=dropout,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None
    stopped_early = False

    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            count = labels.size(0)
            total_loss += loss.item() * count
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += count

        if total == 0:
            raise ValueError("Cannot train on an empty dataset")

        train_loss = total_loss / total
        train_accuracy = correct / total
        val_loss, val_accuracy = _evaluate(model, val_loader, criterion)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_accuracy"].append(float(train_accuracy))
        history["val_accuracy"].append(float(val_accuracy))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            best_state = copy.deepcopy(model.state_dict())

            # Save the best weights and their associated metadata.
            torch.save(
                {
                    "model_state_dict": best_state,
                    "epoch": best_epoch,
                    "validation_loss": float(best_val_loss),
                    "input_features": n_features,
                    "hidden_units": hidden_units,
                    "dropout": dropout,
                    "random_state": random_state,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            stopped_early = True
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")

    # Ensure downstream diagnostics use the best model, not the last epoch.
    model.load_state_dict(best_state)
    model.eval()

    elapsed_seconds = time.perf_counter() - started_at

    training_summary = {
        "requested_epochs": epochs,
        "epochs_completed": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "best_validation_loss": float(best_val_loss),
        "early_stopping_triggered": stopped_early,
        "early_stopping_patience": early_stopping_patience,
        "training_time_seconds": round(elapsed_seconds, 4),
        "checkpoint_path": str(checkpoint_path.resolve()),
    }

    return {
        "model": model,
        "history": history,
        "X_train": data["X_train"],
        "X_val": data["X_val"],
        "y_val": data["y_val"],
        "dataset": {
            "training_samples": len(data["y_train"]),
            "validation_samples": len(data["y_val"]),
            "feature_count": n_features,
            "num_classes": data["num_classes"],
        },
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "random_state": random_state,
            "hidden_units": hidden_units,
            "dropout": dropout,
            "early_stopping_patience": early_stopping_patience,
        },
        "training_summary": training_summary,
    }
