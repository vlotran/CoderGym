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

INPUT_DIM = 50
HIDDEN_DIM = 128
OUTPUT_DIM = 5
TRAIN_SAMPLES = 1000
VAL_SAMPLES = 300
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.01


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_dataloaders():
    X, y = make_classification(
        n_samples=TRAIN_SAMPLES + VAL_SAMPLES,
        n_features=INPUT_DIM,
        n_informative=30,
        n_redundant=15,
        n_classes=OUTPUT_DIM,
        n_clusters_per_class=2,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SAMPLES / (TRAIN_SAMPLES + VAL_SAMPLES), random_state=42
    )

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


class MLP(nn.Module):
    def __init__(self, use_bn=False):
        super().__init__()

        if use_bn:
            self.net = nn.Sequential(
                nn.Linear(INPUT_DIM, HIDDEN_DIM),
                nn.BatchNorm1d(HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                nn.BatchNorm1d(HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                nn.BatchNorm1d(HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(INPUT_DIM, HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
            )

    def forward(self, x):
        return self.net(x)


def train(model, train_loader, val_loader, device):
    model.to(device)
    opt = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    train_accs, val_accs = [], []

    print_every = 10

    for epoch in range(EPOCHS):
        model.train()
        correct, total, loss_sum = 0, 0, 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

            loss_sum += loss.item()
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        train_accs.append(train_acc)

        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        val_acc = correct / total
        val_accs.append(val_acc)

        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")

    return train_accs, val_accs


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
    with open("output/batchnorm_results.json", "w") as f:
        json.dump(results, f, indent=2)


def main():
    device = get_device()
    print("Device:", device)

    train_loader, val_loader = make_dataloaders()

    results = {}

    print("\nTraining WITHOUT BatchNorm")
    set_seed(42)
    model_no_bn = MLP(use_bn=False)
    tr1, va1 = train(model_no_bn, train_loader, val_loader, device)

    no_bn_acc = evaluate(model_no_bn, val_loader, device)["accuracy"]

    print("\nTraining WITH BatchNorm")
    set_seed(42)
    model_bn = MLP(use_bn=True)
    tr2, va2 = train(model_bn, train_loader, val_loader, device)

    bn_acc = evaluate(model_bn, val_loader, device)["accuracy"]

    results["without_batchnorm"] = {
        "final_val_acc": float(no_bn_acc),
        "train_acc": float(tr1[-1]),
        "val_acc_curve": [float(x) for x in va1]
    }

    results["with_batchnorm"] = {
        "final_val_acc": float(bn_acc),
        "train_acc": float(tr2[-1]),
        "val_acc_curve": [float(x) for x in va2]
    }

    save_artifacts(results)

    improvement = bn_acc - no_bn_acc

    print("\nFINAL RESULTS")
    print("No BN:", no_bn_acc)
    print("BN:", bn_acc)
    print("Improvement:", improvement)

    check1 = bn_acc >= no_bn_acc
    check2 = max(va2) >= max(va1)

    passed = check1 and check2

    print("\nCHECKS:")
    print("BN improves accuracy:", check1)
    print("BN improves peak performance:", check2)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())