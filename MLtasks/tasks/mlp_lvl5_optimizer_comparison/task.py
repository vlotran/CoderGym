"""
Optimizer Comparison Task 

This script trains identical neural networks using different optimizers
(SGD, SGD+Momentum, Adam, RMSprop) and compares their convergence behavior.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score

# Seeds
torch.manual_seed(42)
np.random.seed(42)

# Constants
INPUT_DIM = 2
HIDDEN_DIM = 16
OUTPUT_DIM = 2
TRAIN_SAMPLES = 800
VAL_SAMPLES = 200
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.01


def get_task_metadata():
    return {
        "task_name": "optimizer_comparison",
        "description": "Compare optimizers on same architecture",
        "optimizers": ["SGD", "SGD_Momentum", "Adam", "RMSprop"],
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders():
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_ds, batch_size=BATCH_SIZE),
        X_train, y_train,
        X_val, y_val
    )


class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        )

    def forward(self, x):
        return self.net(x)


def build_model(device=None):
    if device is None:
        device = get_device()
    return SimpleClassifier().to(device)


def get_optimizer(model, name):
    if name == "SGD":
        return optim.SGD(model.parameters(), lr=LEARNING_RATE)
    if name == "SGD_Momentum":
        return optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    if name == "Adam":
        return optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    raise ValueError(name)


def train(model, train_loader, val_loader, optimizer_name, device=None):
    if device is None:
        device = get_device()

    opt = get_optimizer(model, optimizer_name)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []

    for _ in range(EPOCHS):
        model.train()
        total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
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
    preds, targets, probs = [], [], []

    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            out = model(X)
            p = torch.softmax(out, dim=1)

            preds.extend(torch.argmax(out, 1).cpu().numpy())
            probs.extend(p.cpu().numpy())
            targets.extend(y.numpy())

    preds = np.array(preds)
    targets = np.array(targets)
    probs = np.array(probs)

    acc = accuracy_score(targets, preds)
    loss = log_loss(targets, probs)


    mse = mean_squared_error(targets, preds)
    r2 = r2_score(targets, preds)

    return {
        "accuracy": float(acc),
        "log_loss": float(loss),
        "mse": float(mse),
        "r2": float(r2),
    }


def predict(model, X):
    model.eval()
    with torch.no_grad():
        out = model(torch.FloatTensor(X))
        return torch.argmax(out, 1).numpy()


def save_artifacts(results, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


def main():
    print("Running Optimizer Comparison...")

    device = get_device()
    train_loader, val_loader, X_train, y_train, X_val, y_val = make_dataloaders()

    optimizers = ["SGD", "SGD_Momentum", "Adam", "RMSprop"]
    results = {}

    for name in optimizers:
        set_seed(42)
        model = build_model(device)

        train(model, train_loader, val_loader, name, device)

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        results[name] = {
            "train": train_metrics,
            "val": val_metrics
        }

        print(f"\n{name}")
        print("Train:", train_metrics)
        print("Val  :", val_metrics)

    save_artifacts(results)

    best_r2 = max(r["val"]["r2"] for r in results.values())
    best_mse = min(r["val"]["mse"] for r in results.values())

    print("\nQuality Checks:")
    print(f"Best R2: {best_r2:.4f}")
    print(f"Best MSE: {best_mse:.4f}")

    assert best_r2 > 0.5, "R2 too low"
    assert best_mse < 0.3, "MSE too high"

    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())