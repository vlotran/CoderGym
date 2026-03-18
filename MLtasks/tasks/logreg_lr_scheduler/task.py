"""
Logistic Regression with Learning Rate Scheduler
Binary classification with StepLR scheduler for adaptive learning rate.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def get_task_metadata() -> Dict[str, Any]:
    return {
        "name": "Logistic_Regression_LR_Scheduler",
        "description": "Logistic regression with learning rate scheduler on synthetic data",
        "task_type": "binary_classification",
        "dataset": "synthetic (sklearn.make_classification)",
        "algorithm": "Logistic Regression with StepLR Scheduler",
        "metrics": ["accuracy", "loss"],
        "default_epochs": 800,
        "default_batch_size": 32
    }

def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_dataloaders(n_samples: int = 1000, n_features: int = 20, batch_size: int = 32, val_split: float = 0.2) -> Tuple:
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=15, n_redundant=5, n_classes=2, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=SEED, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val).unsqueeze(1)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, scaler, n_features

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim: int):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))

def build_model(input_dim: int, device=None) -> LogisticRegressionModel:
    if device is None:
        device = get_device()
    return LogisticRegressionModel(input_dim).to(device)

def train(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    total_samples = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = F.binary_cross_entropy(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        total_samples += X_batch.size(0)
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    return total_loss / total_samples, current_lr

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = F.binary_cross_entropy(y_pred, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            y_pred_class = (y_pred >= 0.5).float()
            correct += (y_pred_class == y_batch).sum().item()
            total += X_batch.size(0)
    accuracy = correct / total
    avg_loss = total_loss / total
    return {"loss": avg_loss, "accuracy": accuracy, "mse": avg_loss, "r2": accuracy}

def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        return model(X)

def save_artifacts(model, metrics, metadata):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    ml_tasks_path = os.path.join(repo_root, "ml_tasks.json")
    task_entry = {
        "series": "Logistic Regression",
        "level": 8,
        "id": "logreg_lr_scheduler",
        "algorithm": metadata.get("algorithm", "Logistic Regression with LR Scheduler"),
        "description": metadata.get("description", ""),
        "interface_protocol": "pytorch_task_v1",
        "requirements": {
            "data": metadata.get("dataset", ""),
            "validation": "Accuracy > 0.80, LR decreases over time"
        }
    }
    if os.path.exists(ml_tasks_path):
        try:
            with open(ml_tasks_path, "r") as f:
                ml_tasks_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print("Error reading ml_tasks.json")
            return
    else:
        print(f"ml_tasks.json not found at {ml_tasks_path}")
        return
    if "tasks" not in ml_tasks_data:
        ml_tasks_data["tasks"] = []
    task_id = task_entry["id"]
    updated = False
    for i, task in enumerate(ml_tasks_data["tasks"]):
        if task.get("id") == task_id:
            ml_tasks_data["tasks"][i] = task_entry
            updated = True
            print(f"Updated task '{task_id}' in ml_tasks.json")
            break
    if not updated:
        ml_tasks_data["tasks"].append(task_entry)
        print(f"Added new task '{task_id}' to ml_tasks.json")
    with open(ml_tasks_path, "w") as f:
        json.dump(ml_tasks_data, f, indent=4)
    print(f"ml_tasks.json updated: {ml_tasks_path}")

def main():
    metadata = get_task_metadata()
    print(f"Starting {metadata['name']} task...")
    device = get_device()
    print(f"Using device: {device}")
    n_samples = 1000
    n_features = 20
    batch_size = 32
    epochs = 800
    initial_lr = 0.1
    step_size = 200
    gamma = 0.5
    print("Creating dataloaders...")
    train_loader, val_loader, scaler, n_feat = make_dataloaders(n_samples=n_samples, n_features=n_features, batch_size=batch_size)
    print(f"Dataset: Synthetic binary classification")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    print(f"Features: {n_features}")
    print("Building logistic regression model...")
    model = build_model(n_features, device=device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params}")
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    print(f"\nStarting training with StepLR scheduler")
    print(f"Initial LR: {initial_lr}, Step size: {step_size}, Gamma: {gamma}")
    best_val_acc = 0
    best_model_state = None
    lr_history = []
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, current_lr = train(model, train_loader, optimizer, scheduler, device)
        lr_history.append(current_lr)
        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_state = model.state_dict().copy()
        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | LR: {current_lr:.6f}")
    training_time = time.time() - start_time
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print("\n" + "="*60)
    print("Final Evaluation Results")
    print("="*60)
    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    print(f"\nTrain Set:")
    print(f"  Loss: {train_metrics['loss']:.4f}")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"\nValidation Set:")
    print(f"  Loss: {val_metrics['loss']:.4f}")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"\nLearning Rate Schedule:")
    print(f"  Initial LR: {lr_history[0]:.6f}")
    print(f"  Final LR: {lr_history[-1]:.6f}")
    print(f"  LR reduced {len(set(lr_history)) - 1} times")
    final_metrics = {"train_loss": train_metrics['loss'], "train_accuracy": train_metrics['accuracy'], "val_loss": val_metrics['loss'], "val_accuracy": val_metrics['accuracy'], "training_time_seconds": training_time, "initial_lr": initial_lr, "final_lr": lr_history[-1], "step_size": step_size, "gamma": gamma}
    print("\n" + "="*60)
    print("Quality Threshold Checks")
    print("="*60)
    quality_passed = True
    if val_metrics['accuracy'] < 0.80:
        print(f"❌ FAIL: Validation accuracy {val_metrics['accuracy']:.4f} < 0.80")
        quality_passed = False
    else:
        print(f"✅ PASS: Validation accuracy {val_metrics['accuracy']:.4f} >= 0.80")
    if lr_history[-1] >= lr_history[0]:
        print(f"❌ FAIL: Learning rate did not decrease (Final: {lr_history[-1]:.6f} >= Initial: {lr_history[0]:.6f})")
        quality_passed = False
    else:
        print(f"✅ PASS: Learning rate decreased from {lr_history[0]:.6f} to {lr_history[-1]:.6f}")
    if training_time > 120:
        print(f"⚠️  WARNING: Training took {training_time:.1f}s (> 2 minutes)")
    else:
        print(f"✅ PASS: Training time {training_time:.1f}s is reasonable")
    print("\nSaving artifacts...")
    save_artifacts(model, final_metrics, metadata)
    print("\n" + "="*60)
    if quality_passed:
        print("✅ ALL QUALITY CHECKS PASSED")
        print("="*60)
        return 0
    else:
        print("❌ SOME QUALITY CHECKS FAILED")
        print("="*60)
        return 1

if __name__ == '__main__':
    sys.exit(main())