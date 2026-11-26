from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from spatial_moire.baseline.metrics import compute_binary_metrics
from spatial_moire.config import BEST_MODEL_FILENAME, LEARNING_RATE, TRAIN_EPOCHS, WEIGHT_DECAY


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> dict:
    model.train()
    running_loss = 0.0
    correct = 0
    seen = 0
    for images, labels in loader:
        images = images.to(device=device, non_blocking=True)
        targets = labels.to(device=device, dtype=torch.float32, non_blocking=True).unsqueeze(dim=1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = images.size(dim=0)
        running_loss += loss.item() * batch_size
        correct += int(((logits >= 0.0) == (targets >= 0.5)).sum().item())
        seen += batch_size
    return {"loss": running_loss / max(seen, 1), "accuracy": correct / max(seen, 1)}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    running_loss = 0.0
    correct = 0
    seen = 0
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device=device, non_blocking=True)
        targets = labels.to(device=device, dtype=torch.float32, non_blocking=True).unsqueeze(dim=1)
        logits = model(images)
        loss = criterion(logits, targets)
        batch_size = images.size(dim=0)
        running_loss += loss.item() * batch_size
        probabilities = torch.sigmoid(logits).squeeze(dim=1)
        correct += int(((logits >= 0.0) == (targets >= 0.5)).sum().item())
        seen += batch_size
        all_labels.append(labels.numpy())
        all_probabilities.append(probabilities.detach().cpu().numpy())
    labels_array = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
    probabilities_array = np.concatenate(all_probabilities) if all_probabilities else np.array([])
    return {
        "loss": running_loss / max(seen, 1),
        "accuracy": correct / max(seen, 1),
        "labels": labels_array,
        "probabilities": probabilities_array,
    }


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epochs: int = TRAIN_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
        "learning_rate": [],
    }
    best_f1 = -1.0
    best_epoch = 0
    checkpoint_path = output_dir / BEST_MODEL_FILENAME
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        val_stats = evaluate(model=model, loader=val_loader, criterion=criterion, device=device)
        val_metrics = compute_binary_metrics(labels=val_stats["labels"], probabilities=val_stats["probabilities"])
        lr_used = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.perf_counter() - started
        history["epoch"].append(epoch)
        history["train_loss"].append(train_stats["loss"])
        history["train_accuracy"].append(train_stats["accuracy"])
        history["val_loss"].append(val_stats["loss"])
        history["val_accuracy"].append(val_stats["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["learning_rate"].append(lr_used)
        print(
            "epoch {:02d}/{} | train_loss {:.4f} | train_acc {:.4f} | val_loss {:.4f} | "
            "val_acc {:.4f} | val_f1 {:.4f} | lr {:.2e} | {:.1f}s".format(
                epoch,
                epochs,
                train_stats["loss"],
                train_stats["accuracy"],
                val_stats["loss"],
                val_stats["accuracy"],
                val_metrics["f1"],
                lr_used,
                elapsed,
            ),
            flush=True,
        )
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save(obj={"epoch": epoch, "val_f1": best_f1, "model_state": model.state_dict()}, f=checkpoint_path)
    history["best_val_f1"] = best_f1
    history["best_epoch"] = best_epoch
    return history
