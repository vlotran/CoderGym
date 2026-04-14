"""
Dropout Regularization Study Task
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

torch.manual_seed(42)
np.random.seed(42)

INPUT_DIM = 20
HIDDEN_DIM = 64
OUTPUT_DIM = 2
TRAIN_SAMPLES = 800
VAL_SAMPLES = 200
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.001


def get_task_metadata():
    return {
        "task_name": "dropout_regularization_study",
        "dropout_rates": [0.0, 0.2, 0.5, 0.7],
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders():
    X, y = make_classification(
        n_samples=1000,
        n_features=INPUT_DIM,
        n_informative=10,
        n_redundant=5,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_ds, batch_size=BATCH_SIZE),
    )


class DropoutClassifier(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        )

    def forward(self, x):
        return self.net(x)


def build_model(dropout_rate, device=None):
    if device is None:
        device = get_device()
    return DropoutClassifier(dropout_rate).to(device)


def train(model, train_loader, val_loader, device=None):
    if device is None:
        device = get_device()

    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []

    for _ in range(EPOCHS):
        model.train()
        total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()
            total += loss.item()

        train_losses.append(total / len(train_loader))

        model.eval()
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                total += loss_fn(model(X), y).item()

        val_losses.append(total / len(val_loader))

    return train_losses, val_losses


def evaluate(model, data_loader, device=None):
    if device is None:
        device = get_device()

    model.eval()
    preds, targets = [], []
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            total_loss += loss_fn(out, y).item()

            preds.extend(torch.argmax(out, 1).cpu().numpy())
            targets.extend(y.cpu().numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    acc = accuracy_score(targets, preds)
    mse = mean_squared_error(targets, preds)
    r2 = r2_score(targets, preds)

    return {
        "accuracy": float(acc),
        "loss": float(total_loss / len(data_loader)),
        "mse": float(mse),
        "r2": float(r2),
    }


def predict(model, X):
    with torch.no_grad():
        out = model(torch.FloatTensor(X))
        return torch.argmax(out, 1).numpy()


def save_artifacts(results, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "dropout_study.json"), "w") as f:
        json.dump(results, f, indent=2)


def main():
    device = get_device()
    train_loader, val_loader = make_dataloaders()

    dropout_rates = [0.0, 0.2, 0.5, 0.7]
    results = {}

    for rate in dropout_rates:
        set_seed(42)
        model = build_model(rate, device)

        train_losses, val_losses = train(model, train_loader, val_loader, device)

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        results[f"dropout_{rate}"] = {
            "train": train_metrics,
            "val": val_metrics,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

        print(f"\nDropout {rate}")
        print("Train:", train_metrics)
        print("Val  :", val_metrics)

    save_artifacts(results)

    best_r2 = max(r["val"]["r2"] for r in results.values())
    best_mse = min(r["val"]["mse"] for r in results.values())

    print("\nQuality Checks:")
    print(f"Best R2: {best_r2:.4f}")
    print(f"Best MSE: {best_mse:.4f}")

    assert best_r2 > 0.3
    assert best_mse < 0.5

    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())