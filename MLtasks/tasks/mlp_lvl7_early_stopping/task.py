"""
Early Stopping Implementation Task

This script demonstrates early stopping to prevent overfitting by monitoring
validation loss and stopping training when it stops improving.
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
from sklearn.metrics import accuracy_score

torch.manual_seed(42)
np.random.seed(42)

INPUT_DIM = 30
HIDDEN_DIM = 128
OUTPUT_DIM = 3
TRAIN_SAMPLES = 600
VAL_SAMPLES = 200
BATCH_SIZE = 32
MAX_EPOCHS = 200
LEARNING_RATE = 0.001
PATIENCE = 15
MIN_DELTA = 0.001


def get_task_metadata():
    return {
        "task_name": "early_stopping",
        "description": "Implement early stopping to prevent overfitting",
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "max_epochs": MAX_EPOCHS
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders():
    X, y = make_classification(
        n_samples=TRAIN_SAMPLES + VAL_SAMPLES,
        n_features=INPUT_DIM,
        n_informative=15,
        n_redundant=10,
        n_classes=OUTPUT_DIM,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VAL_SAMPLES / (TRAIN_SAMPLES + VAL_SAMPLES),
        random_state=42
    )

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


class EarlyStoppingClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_model(device):
    return EarlyStoppingClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)


class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0


def train(model, train_loader, val_loader, use_es, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    es = EarlyStopping() if use_es else None

    train_accs, val_accs = [], []
    train_losses, val_losses = [], []

    stopped_epoch = MAX_EPOCHS

    for epoch in range(MAX_EPOCHS):
        model.train()
        correct, total, loss_sum = 0, 0, 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_losses.append(loss_sum / len(train_loader))
        train_accs.append(correct / total)

        model.eval()
        correct, total, loss_sum = 0, 0, 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)

                loss_sum += loss.item()
                preds = out.argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_losses.append(loss_sum / len(val_loader))
        val_accs.append(correct / total)

        print(
            f"Epoch [{epoch+1}/{MAX_EPOCHS}] "
            f"Train Loss: {train_losses[-1]:.4f} "
            f"Train Acc: {train_accs[-1]:.4f} "
            f"Val Loss: {val_losses[-1]:.4f} "
            f"Val Acc: {val_accs[-1]:.4f}"
        )

        if use_es:
            es(val_losses[-1], epoch)
            if es.early_stop:
                stopped_epoch = epoch + 1
                print(f"Early stopping triggered at epoch {stopped_epoch}")
                break

    return train_losses, val_losses, train_accs, val_accs, stopped_epoch


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x).argmax(1).cpu().numpy()
            preds.extend(out)
            targets.extend(y.numpy())

    return {"accuracy": float(accuracy_score(targets, preds))}


def save_artifacts(results):
    os.makedirs("output", exist_ok=True)
    with open("output/early_stopping_results.json", "w") as f:
        json.dump(results, f, indent=2)


def main():
    device = get_device()
    train_loader, val_loader = make_dataloaders()

    results = {}

    set_seed(42)
    model1 = build_model(device)

    t1, v1, ta1, va1, e1 = train(model1, train_loader, val_loader, False, device)
    print("\nTraining complete WITHOUT early stopping\n")

    results["no_es"] = {
        "epochs": e1,
        "train_acc": float(ta1[-1]),
        "val_acc": float(va1[-1]),
        "gap": float(ta1[-1] - va1[-1]),
    }

    set_seed(42)
    model2 = build_model(device)

    t2, v2, ta2, va2, e2 = train(model2, train_loader, val_loader, True, device)
    print("\nTraining complete WITH early stopping\n")

    results["es"] = {
        "epochs": e2,
        "train_acc": float(ta2[-1]),
        "val_acc": float(va2[-1]),
        "gap": float(ta2[-1] - va2[-1]),
    }

    save_artifacts(results)

    gap_no_es = results["no_es"]["gap"]
    gap_es = results["es"]["gap"]

    checks = {
        "early_stop_triggered": results["es"]["epochs"] < MAX_EPOCHS,
        "better_generalization": gap_es < gap_no_es,
        "valid_accuracy": results["es"]["val_acc"] > 0.6,
    }

    passed = all(checks.values())

    print("\nFINAL CHECKS:", checks)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())