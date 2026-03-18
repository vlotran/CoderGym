"""
Multi-class Softmax Regression on Iris Dataset
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
from sklearn.datasets import load_iris
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
        "name": "Iris_Multiclass_Softmax_Regression",
        "description": "Multi-class classification using softmax regression on Iris dataset",
        "task_type": "multiclass_classification",
        "dataset": "sklearn.datasets.load_iris",
        "algorithm": "Softmax Regression",
        "metrics": ["accuracy", "macro_f1"],
        "num_classes": 3,
        "default_epochs": 500,
        "default_batch_size": 16
    }

def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_dataloaders(batch_size: int = 16, val_split: float = 0.3) -> Tuple:
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=SEED, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, scaler, X.shape[1], len(np.unique(y)), data.target_names

class SoftmaxRegressionModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super(SoftmaxRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

def build_model(input_dim: int, num_classes: int, device=None) -> SoftmaxRegressionModel:
    if device is None:
        device = get_device()
    return SoftmaxRegressionModel(input_dim, num_classes).to(device)

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    total_samples = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = F.cross_entropy(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        total_samples += X_batch.size(0)
    return total_loss / total_samples

def evaluate(model, loader, device, num_classes=3):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = F.cross_entropy(logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            _, predictions = torch.max(logits, dim=1)
            correct += (predictions == y_batch).sum().item()
            total += X_batch.size(0)
            for true, pred in zip(y_batch.cpu().numpy(), predictions.cpu().numpy()):
                confusion_matrix[true, pred] += 1
    accuracy = correct / total
    avg_loss = total_loss / total
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)
    macro_precision = np.mean(per_class_precision)
    macro_recall = np.mean(per_class_recall)
    macro_f1 = np.mean(per_class_f1)
    return {"loss": avg_loss, "accuracy": accuracy, "macro_precision": macro_precision, "macro_recall": macro_recall, "macro_f1": macro_f1, "mse": avg_loss, "r2": accuracy}

def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        logits = model(X)
        return F.softmax(logits, dim=1)

def save_artifacts(model, metrics, metadata):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    ml_tasks_path = os.path.join(repo_root, "ml_tasks.json")
    task_entry = {
        "series": "Logistic Regression",
        "level": 6,
        "id": "softmax_iris_multiclass",
        "algorithm": metadata.get("algorithm", "Softmax Regression"),
        "description": metadata.get("description", ""),
        "interface_protocol": "pytorch_task_v1",
        "requirements": {
            "data": metadata.get("dataset", ""),
            "validation": "Accuracy > 0.85, Macro-F1 > 0.85"
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
    epochs = 500
    batch_size = 16
    lr = 0.01
    print("Creating dataloaders...")
    train_loader, val_loader, scaler, n_features, n_classes, class_names = make_dataloaders(batch_size=batch_size)
    print(f"Dataset: Iris")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    print(f"Features: {n_features}, Classes: {n_classes}")
    print("Building softmax regression model...")
    model = build_model(n_features, n_classes, device=device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("\nStarting training...")
    best_val_acc = 0
    best_model_state = None
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        train_metrics = evaluate(model, train_loader, device, n_classes)
        val_metrics = evaluate(model, val_loader, device, n_classes)
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_state = model.state_dict().copy()
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
    training_time = time.time() - start_time
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print("\n" + "="*60)
    print("Final Evaluation Results")
    print("="*60)
    train_metrics = evaluate(model, train_loader, device, n_classes)
    val_metrics = evaluate(model, val_loader, device, n_classes)
    print(f"\nTrain Set:")
    print(f"  Loss: {train_metrics['loss']:.4f}")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {train_metrics['macro_f1']:.4f}")
    print(f"\nValidation Set:")
    print(f"  Loss: {val_metrics['loss']:.4f}")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {val_metrics['macro_f1']:.4f}")
    final_metrics = {"train_loss": train_metrics['loss'], "train_accuracy": train_metrics['accuracy'], "val_loss": val_metrics['loss'], "val_accuracy": val_metrics['accuracy'], "val_macro_f1": val_metrics['macro_f1'], "training_time_seconds": training_time}
    print("\n" + "="*60)
    print("Quality Threshold Checks")
    print("="*60)
    quality_passed = True
    if val_metrics['accuracy'] < 0.85:
        print(f"❌ FAIL: Validation accuracy {val_metrics['accuracy']:.4f} < 0.85")
        quality_passed = False
    else:
        print(f"✅ PASS: Validation accuracy {val_metrics['accuracy']:.4f} >= 0.85")
    if val_metrics['macro_f1'] < 0.85:
        print(f"❌ FAIL: Validation Macro-F1 {val_metrics['macro_f1']:.4f} < 0.85")
        quality_passed = False
    else:
        print(f"✅ PASS: Validation Macro-F1 {val_metrics['macro_f1']:.4f} >= 0.85")
    if training_time > 60:
        print(f"⚠️  WARNING: Training took {training_time:.1f}s (> 1 minute)")
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