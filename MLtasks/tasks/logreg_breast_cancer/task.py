"""
Binary Logistic Regression on Breast Cancer Dataset
Implement binary classification for cancer diagnosis using sklearn breast cancer dataset.
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
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def get_task_metadata() -> Dict[str, Any]:
    """Return task metadata."""
    return {
        "name": "Breast_Cancer_Binary_Classification",
        "description": "Binary logistic regression on Wisconsin breast cancer dataset",
        "task_type": "binary_classification",
        "dataset": "sklearn.datasets.load_breast_cancer",
        "algorithm": "Logistic Regression",
        "metrics": ["accuracy", "precision", "recall", "f1"],
        "default_epochs": 1000,
        "default_batch_size": 32
    }


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get computation device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(batch_size: int = 32, val_split: float = 0.2) -> Tuple:
    """Load breast cancer dataset and create dataloaders."""
    data = load_breast_cancer()
    X, y = data.data, data.target
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
    return train_loader, val_loader, scaler, X.shape[1]


class LogisticRegressionModel(nn.Module):
    """Logistic Regression model for binary classification."""
    def __init__(self, input_dim: int):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))


def build_model(input_dim: int, device=None) -> LogisticRegressionModel:
    """Build the logistic regression model."""
    if device is None:
        device = get_device()
    model = LogisticRegressionModel(input_dim).to(device)
    return model


def train(model: LogisticRegressionModel, train_loader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """Train the model for one epoch."""
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
    return total_loss / total_samples


def evaluate(model: LogisticRegressionModel, loader, device: torch.device) -> Dict[str, float]:
    """Evaluate the model on given data loader."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = F.binary_cross_entropy(y_pred, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            y_pred_class = (y_pred >= 0.5).float()
            correct += (y_pred_class == y_batch).sum().item()
            total += X_batch.size(0)
            true_positives += ((y_pred_class == 1) & (y_batch == 1)).sum().item()
            false_positives += ((y_pred_class == 1) & (y_batch == 0)).sum().item()
            false_negatives += ((y_pred_class == 0) & (y_batch == 1)).sum().item()
    accuracy = correct / total
    avg_loss = total_loss / total
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return {"loss": avg_loss, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "mse": avg_loss, "r2": accuracy}


def predict(model: LogisticRegressionModel, X: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Get predictions for input data."""
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        return model(X)


def save_artifacts(model: LogisticRegressionModel, metrics: Dict[str, float], metadata: Dict[str, Any]) -> None:
    """Append task metadata to ml_tasks.json."""
    # Find ml_tasks.json in repo root (MLtasks/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))  # Go up to MLtasks/
    ml_tasks_path = os.path.join(repo_root, "ml_tasks.json")
    
    # Create new task entry matching the existing format
    task_entry = {
        "series": "Logistic Regression",
        "level": 5,
        "id": "logreg_breast_cancer",
        "algorithm": metadata.get("algorithm", "Logistic Regression"),
        "description": metadata.get("description", ""),
        "interface_protocol": "pytorch_task_v1",
        "requirements": {
            "data": metadata.get("dataset", ""),
            "validation": "Accuracy > 0.90, F1 > 0.85"
        }
    }
    
    # Load existing ml_tasks.json
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
    
    # Ensure tasks array exists
    if "tasks" not in ml_tasks_data:
        ml_tasks_data["tasks"] = []
    
    # Check if task with same id already exists
    task_id = task_entry["id"]  # Use "id" instead of "name"
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
    
    # Save back to ml_tasks.json
    with open(ml_tasks_path, "w") as f:
        json.dump(ml_tasks_data, f, indent=4)
    
    print(f"ml_tasks.json updated: {ml_tasks_path}")


def main():
    """Main training and evaluation function."""
    metadata = get_task_metadata()
    print(f"Starting {metadata['name']} task...")
    device = get_device()
    print(f"Using device: {device}")
    epochs = 1000
    batch_size = 32
    lr = 0.01
    print("Creating dataloaders...")
    train_loader, val_loader, scaler, n_features = make_dataloaders(batch_size=batch_size)
    print(f"Dataset: Breast Cancer Wisconsin")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    print(f"Features: {n_features}")
    print("Building logistic regression model...")
    model = build_model(n_features, device=device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params}")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    print("\nStarting training...")
    best_val_acc = 0
    best_model_state = None
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_state = model.state_dict().copy()
        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
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
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall: {train_metrics['recall']:.4f}")
    print(f"  F1 Score: {train_metrics['f1']:.4f}")
    print(f"\nValidation Set:")
    print(f"  Loss: {val_metrics['loss']:.4f}")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall: {val_metrics['recall']:.4f}")
    print(f"  F1 Score: {val_metrics['f1']:.4f}")
    final_metrics = {"train_loss": train_metrics['loss'], "train_accuracy": train_metrics['accuracy'], "train_f1": train_metrics['f1'], "val_loss": val_metrics['loss'], "val_accuracy": val_metrics['accuracy'], "val_precision": val_metrics['precision'], "val_recall": val_metrics['recall'], "val_f1": val_metrics['f1'], "training_time_seconds": training_time, "epochs": epochs, "batch_size": batch_size, "learning_rate": lr, "num_parameters": num_params}
    print("\n" + "="*60)
    print("Quality Threshold Checks")
    print("="*60)
    quality_passed = True
    if val_metrics['accuracy'] < 0.90:
        print(f"❌ FAIL: Validation accuracy {val_metrics['accuracy']:.4f} < 0.90")
        quality_passed = False
    else:
        print(f"✅ PASS: Validation accuracy {val_metrics['accuracy']:.4f} >= 0.90")
    if val_metrics['f1'] < 0.85:
        print(f"❌ FAIL: Validation F1 {val_metrics['f1']:.4f} < 0.85")
        quality_passed = False
    else:
        print(f"✅ PASS: Validation F1 {val_metrics['f1']:.4f} >= 0.85")
    if train_metrics['accuracy'] < val_metrics['accuracy'] * 0.95:
        print(f"⚠️  WARNING: Training accuracy {train_metrics['accuracy']:.4f} < Validation accuracy {val_metrics['accuracy']:.4f} * 0.95")
    else:
        print(f"✅ PASS: No severe underfitting detected")
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