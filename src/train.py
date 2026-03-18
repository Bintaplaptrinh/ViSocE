"""Training and validation utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.amp.autocast_mode import autocast
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler, device, amp_enabled: bool = True):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=amp_enabled):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def collect_probs_labels(model, loader, device, num_labels: int, amp_enabled: bool = True):
    model.eval()
    all_probs = []
    all_labels = []

    for batch in tqdm(loader, desc="Infer", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()

        with autocast("cuda", enabled=amp_enabled):
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels)

    probs = np.vstack(all_probs) if all_probs else np.empty((0, num_labels), dtype=np.float32)
    trues = np.vstack(all_labels) if all_labels else np.empty((0, num_labels), dtype=np.float32)
    return probs, trues


def compute_macro_f1(y_true, y_prob, thresholds):
    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(int)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def early_stopping_train_loop(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    epochs: int,
    thresholds,
    checkpoint_path,
    amp_enabled: bool,
    min_delta: float,
    patience: int,
):
    best_macro_f1 = -1.0
    patience_count = 0
    num_labels = len(thresholds)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
        )

        val_probs, val_true = collect_probs_labels(
            model=model,
            loader=val_loader,
            device=device,
            num_labels=num_labels,
            amp_enabled=amp_enabled,
        )
        val_f1_macro = compute_macro_f1(val_true, val_probs, thresholds)

        print(f"Epoch {epoch:02d} | loss={loss:.4f} | val_f1_macro={val_f1_macro:.4f}")

        if val_f1_macro > best_macro_f1 + min_delta:
            best_macro_f1 = val_f1_macro
            patience_count = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best checkpoint to: {checkpoint_path}")
        else:
            patience_count += 1
            print(f"No improvement. Patience {patience_count}/{patience}")

        if patience_count >= patience:
            print("Early stopping triggered.")
            break

    return best_macro_f1
