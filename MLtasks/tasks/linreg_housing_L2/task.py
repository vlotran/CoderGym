"""
Linear Regression with L2 Regularization on California Housing Dataset
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
from sklearn.datasets import fetch_california_housing
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
        "name": "California_Housing_Linear_Regression_L2",
        "description": "Linear regression with L2 regularization (Ridge) on California Housing dataset",
        "task_type": "regression",
        "dataset": "sklearn.datasets.fetch_california_housing",
        "algorithm": "Ridge Regression",
        "metrics": ["mse", "rmse", "r2"],
        "default_epochs": 1000,
        "default_batch_size": 64
    }

def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_dataloaders(batch_size: int = 64, val_split: float = 0.2) -> Tuple:
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=SEED)
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
    return train_loader, val_loader, scaler, X.shape[1], data.feature_names

class LinearRegressionL2(nn.Module):
    def __init__(self, input_dim: int):
        super(LinearRegressionL2, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

def build_model(input_dim: int, device=None) -> LinearRegressionL2:
    if device is None:
        device = get_device()
    return LinearRegressionL2(input_dim).to(device)

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    total_samples = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = F.mse_loss(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        total_samples += X_batch.size(0)
    return total_loss / total_samples

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = F.mse_loss(y_pred, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            all_preds.append(y_pred.cpu())
            all_targets.append(y_batch.cpu())
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    n_samples = len(loader.dataset)
    mse = total_loss / n_samples
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    rmse = np.sqrt(mse)
    return {"mse": mse, "r2": r2, "rmse": rmse, "loss": mse}

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
        "series": "Linear Regression",
        "level": 7,
        "id": "linreg_housing_l2",
        "algorithm": metadata.get("algorithm", "Ridge Regression"),
        "description": metadata.get("description", ""),
        "interface_protocol": "pytorch_task_v1",
        "requirements": {
            "data": metadata.get("dataset", ""),
            "validation": "R² > 0.5, MSE < 2.0"
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
    epochs = 1000
    batch_size = 64
    lr = 0.01
    weight_decay = 0.001
    print("Creating dataloaders...")
    train_loader, val_loader, scaler, n_features, feature_names = make_dataloaders(batch_size=batch_size)
    print(f"Dataset: California Housing")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    print(f"Features: {n_features}")
    print("Building linear regression model with L2 regularization...")
    model = build_model(n_features, device=device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params}")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(f"\nStarting training with L2 weight_decay={weight_decay}...")
    best_val_r2 = -float('inf')
    best_model_state = None
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        if val_metrics['r2'] > best_val_r2:
            best_val_r2 = val_metrics['r2']
            best_model_state = model.state_dict().copy()
        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Train R²: {train_metrics['r2']:.4f} | Val R²: {val_metrics['r2']:.4f}")
    training_time = time.time() - start_time
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print("\n" + "="*60)
    print("Final Evaluation Results")
    print("="*60)
    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    print(f"\nTrain Set:")
    print(f"  MSE: {train_metrics['mse']:.4f}")
    print(f"  RMSE: {train_metrics['rmse']:.4f}")
    print(f"  R² Score: {train_metrics['r2']:.4f}")
    print(f"\nValidation Set:")
    print(f"  MSE: {val_metrics['mse']:.4f}")
    print(f"  RMSE: {val_metrics['rmse']:.4f}")
    print(f"  R² Score: {val_metrics['r2']:.4f}")
    final_metrics = {"train_mse": train_metrics['mse'], "train_rmse": train_metrics['rmse'], "train_r2": train_metrics['r2'], "val_mse": val_metrics['mse'], "val_rmse": val_metrics['rmse'], "val_r2": val_metrics['r2'], "training_time_seconds": training_time, "weight_decay": weight_decay}
    print("\n" + "="*60)
    print("Quality Threshold Checks")
    print("="*60)
    quality_passed = True
    if val_metrics['r2'] < 0.5:
        print(f"❌ FAIL: Validation R² {val_metrics['r2']:.4f} < 0.5")
        quality_passed = False
    else:
        print(f"✅ PASS: Validation R² {val_metrics['r2']:.4f} >= 0.5")
    if val_metrics['mse'] > 2.0:
        print(f"❌ FAIL: Validation MSE {val_metrics['mse']:.4f} > 2.0")
        quality_passed = False
    else:
        print(f"✅ PASS: Validation MSE {val_metrics['mse']:.4f} <= 2.0")
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