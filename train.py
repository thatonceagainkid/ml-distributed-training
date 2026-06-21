# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:34:23 2026

@author: clayt
"""
# =========================================================
# train.py — CIFAR-10 ResNet18 (Notebook Export Version)
# =========================================================

import os
import csv
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import confusion_matrix


# =========================================================
# REPRODUCIBILITY
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# OPTIONAL DISTRIBUTED SUPPORT (SAFE MINIMAL ADDITION)
# =========================================================
def is_distributed():
    return "RANK" in os.environ


def get_rank():
    return int(os.environ.get("RANK", 0))


def setup_device():
    if torch.cuda.is_available():
        if is_distributed():
            torch.cuda.set_device(get_rank())
            return torch.device(f"cuda:{get_rank()}")
        return torch.device("cuda")
    return torch.device("cpu")


# =========================================================
# TRAIN ONE EPOCH
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    preds_all, labels_all = [], []

    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds_all.extend(torch.argmax(out, 1).cpu().numpy())
        labels_all.extend(y.cpu().numpy())

    acc = np.mean(np.array(preds_all) == np.array(labels_all))
    return total_loss / len(loader), acc


# =========================================================
# EVALUATION
# =========================================================
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Training"):
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item()

            preds_all.extend(torch.argmax(out, 1).cpu().numpy())
            labels_all.extend(y.cpu().numpy())

    acc = np.mean(np.array(preds_all) == np.array(labels_all))
    return total_loss / len(loader), acc, preds_all, labels_all


# =========================================================
# MAIN
# =========================================================
def main(args):

    set_seed(args.seed)

    device = setup_device()

    print("====================================")
    print("TRAINING START")
    print("Device:", device)
    print("Distributed:", is_distributed())
    print("====================================")

    # -------------------------
    # OUTPUT DIR
    # -------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "metrics.csv"

    # clean CSV each run
    if csv_path.exists() and get_rank() == 0:
        os.remove(csv_path)

    if get_rank() == 0:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    # -------------------------
    # DATA
    # -------------------------
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    train_data = datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # -------------------------
    # MODEL (EXACT SAME AS NOTEBOOK)
    # -------------------------
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for p in model.parameters():
        p.requires_grad = False

    for p in model.layer3.parameters():
        p.requires_grad = True

    for p in model.layer4.parameters():
        p.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    # -------------------------
    # LOSS / OPTIM / SCHED
    # -------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW([
        {"params": model.layer3.parameters(), "lr": 1e-5},
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # -------------------------
    # TRAIN LOOP
    # -------------------------
    best_acc = 0

    for epoch in range(args.epochs):

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, preds, labels = evaluate(model, test_loader, criterion, device)

        scheduler.step()

        if get_rank() == 0:
            print(f"""
Epoch {epoch+1}/{args.epochs}
Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}
Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}
""")

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc])

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), output_dir / "best_model.pth")
                print("Saved best model ✔")

    if get_rank() == 0:
        torch.save(model.state_dict(), output_dir / "last_model.pth")
        print("Training complete ✔ Best Acc:", best_acc)

        # -------------------------
        # CONFUSION MATRIX
        # -------------------------
        cm = confusion_matrix(labels, preds)
        print("Confusion Matrix:\n", cm)


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    main(args)