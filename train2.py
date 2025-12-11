"""
ADSD - Adaptive Disease Signature Diffusion Model
==================================================
Uses diffusion-based augmentation to generate biologically realistic
disease lesions for improved generalization.

Key Components:
- Denoising Diffusion Probabilistic Model (DDPM) for lesion generation
- Disease signature extraction and transfer
- Adaptive augmentation based on class distribution
- CNN classifier with diffusion-augmented training
"""

# ================================================================
# ADSD-FAST (Option A: MAX SPEED) - Single cell for Kaggle
# - Diffusion disabled (fast mode)
# - Lightweight signature augmentation (no heavy UNet / diffusion)
# - Early stop when train_acc >= 0.96 AND val_acc >= 0.96 (or val >= 0.96)
# - Uses EfficientNet-B0 backbone (pretrained) and AMP
# - Outputs: saved model, training curves, confusion matrix, class report
# ================================================================

import os
import time
import random
from collections import Counter
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, datasets, models

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ----------------------------
# Config (speed-optimized)
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# change these paths if your dataset is elsewhere
TRAIN_DIR = "/kaggle/input/leaf-dataset-vimal/dataset/train"
VALID_DIR = "/kaggle/input/leaf-dataset-vimal/dataset/valid"
TEST_DIR  = "/kaggle/input/leaf-dataset-vimal/dataset/test"

OUT_DIR = "/kaggle/working/adsd_fast"
MODEL_PATH = os.path.join(OUT_DIR, "ADSD_fast_model.pth")
os.makedirs(OUT_DIR, exist_ok=True)

CFG = {
    "image_size": 160,          # smaller = faster
    "batch_size": 32,
    "epochs": 40,               # but early stop will likely cut it short
    "lr": 2e-4,
    "weight_decay": 1e-5,
    "num_workers": 2,
    "target_acc": 0.96,         # 96% threshold
    "use_diffusion": False,     # Option A disables heavy diffusion
    "seed": 42
}

# reproducibility
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
random.seed(CFG["seed"])

# ----------------------------
# Lightweight Disease Signature Extractor (fast)
# ----------------------------
class TinySignatureExtractor(nn.Module):
    """Lightweight encoder that extracts a small disease signature vector"""
    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 80x80
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 40x40
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

# ----------------------------
# Lightweight augmenter using signatures (fast substitute for diffusion)
# ----------------------------
class LightweightAugmenter:
    def __init__(self, signature_extractor, device=DEVICE):
        self.sig = signature_extractor.to(device)
        self.device = device

    def augment(self, images, labels=None, strength=0.25):
        """
        images: tensor Bx3xHxW in [0,1] (we expect normalized later; here we receive pre-normalized tensors)
        returns augmented images (same shape)
        """
        with torch.no_grad():
            B = images.size(0)
            sig_vec = self.sig(images)  # B x D
            sv = (sig_vec - sig_vec.mean(dim=1, keepdim=True)) / (sig_vec.std(dim=1, keepdim=True) + 1e-6)
            D = sig_vec.size(1)
            maps = sv.unsqueeze(-1).unsqueeze(-1) * torch.ones((B, D, 1, 1), device=self.device)
            lesion_map = maps.mean(dim=1, keepdim=True)  # Bx1x1x1
            lesion_map = F.interpolate(lesion_map, size=(images.size(2), images.size(3)), mode="bilinear", align_corners=False)
            color_offsets = torch.tensor([0.9, 0.6, 0.6], device=self.device).view(1,3,1,1)
            lesion_rgb = lesion_map * color_offsets
            lesion_rgb = torch.tanh(lesion_rgb) * 0.5
            alpha = strength
            aug = (1-alpha) * images + alpha * (images + lesion_rgb)
            aug = torch.clamp(aug, 0.0, 1.0)
            return aug

# ----------------------------
# Fast classifier (EfficientNet-B0 backbone with small modifications)
# ----------------------------
class ADSD_FastClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.2):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = backbone.features
        feat_dim = backbone.classifier[1].in_features  # 1280
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        f = self.features(x)
        f = self.pool(f).flatten(1)
        out = self.classifier(f)
        return out

# ----------------------------
# Transforms (fast)
# ----------------------------
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((CFG["image_size"]+16, CFG["image_size"]+16)),
            transforms.RandomResizedCrop(CFG["image_size"], scale=(0.8,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ToTensor(),  # produces 0..1
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CFG["image_size"], CFG["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

# ----------------------------
# Helpers: class weights sampler
# ----------------------------
def get_class_weights(folder_dataset):
    labels = [lab for _, lab in folder_dataset.samples]
    counts = Counter(labels)
    total = len(labels)
    weights = [total / counts[lab] for _, lab in folder_dataset.samples]
    return weights

# ----------------------------
# Plotting helpers
# ----------------------------
def plot_training_curves(history, path):
    plt.figure(figsize=(10,3))
    plt.subplot(1,3,1); plt.plot(history['train_loss'], label='train'); plt.plot(history['val_loss'], label='val'); plt.title('Loss'); plt.legend()
    plt.subplot(1,3,2); plt.plot(history['train_acc'], label='train'); plt.plot(history['val_acc'], label='val'); plt.title('Accuracy'); plt.legend()
    plt.subplot(1,3,3); plt.plot(history['train_f1'], label='train'); plt.plot(history['val_f1'], label='val'); plt.title('F1'); plt.legend()
    plt.tight_layout(); plt.savefig(path); plt.close()

def plot_confusion(cm, classes, path):
    plt.figure(figsize=(10,8));
    import seaborn as sns
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xticks(rotation=90); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(path); plt.close()

# ----------------------------
# Training and evaluation functions
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, augmenter=None, use_augment=False):
    model.train()
    total_loss = 0.0
    preds=[]; labs=[]
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        # images come normalized; augment expects un-normalized [0,1] so we undo norm -> augment -> renorm
        if augmenter is not None and use_augment and torch.rand(1).item() < 0.5:
            mean = torch.tensor([0.485,0.456,0.406], device=DEVICE).view(1,3,1,1)
            std = torch.tensor([0.229,0.224,0.225], device=DEVICE).view(1,3,1,1)
            unnorm = imgs * std + mean
            aug = augmenter.augment(unnorm, labels, strength=0.2)
            imgs = (aug - mean) / std

        optimizer.zero_grad()
        with autocast():
            out = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        preds.extend(out.argmax(1).detach().cpu().numpy().tolist())
        labs.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(labs, preds)
    f1 = f1_score(labs, preds, average='weighted')
    return avg_loss, acc, f1

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    preds=[]; labs=[]
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE); labels = labels.to(DEVICE)
            out = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * imgs.size(0)
            preds.extend(out.argmax(1).cpu().numpy().tolist())
            labs.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(labs, preds)
    f1 = f1_score(labs, preds, average='weighted')
    return avg_loss, acc, f1, preds, labs

# ----------------------------
# Main (single-cell)
# ----------------------------
def main():
    print("ADSD-FAST (Option A) starting. Device:", DEVICE)
    # Datasets
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=get_transforms(train=True))
    valid_ds = datasets.ImageFolder(VALID_DIR, transform=get_transforms(train=False))
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=get_transforms(train=False))

    classes = train_ds.classes
    print("Classes:", len(classes))

    # Weighted sampler for class imbalance
    weights = get_class_weights(train_ds)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], sampler=sampler,
                              num_workers=CFG["num_workers"], pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=CFG["batch_size"], shuffle=False,
                              num_workers=CFG["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"], shuffle=False,
                              num_workers=CFG["num_workers"], pin_memory=True)

    # Models
    classifier = ADSD_FastClassifier(num_classes=len(classes), dropout=0.2).to(DEVICE)
    signature_extractor = TinySignatureExtractor(out_dim=48).to(DEVICE)
    augmenter = LightweightAugmenter(signature_extractor, device=DEVICE)

    # Optimizer / loss / scaler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scaler = GradScaler()

    history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[], 'train_f1':[], 'val_f1':[]}
    best_val = 0.0

    start = time.time()
    for epoch in range(1, CFG["epochs"]+1):
        use_augment = epoch >= 2  # start augmentation after 1 epoch
        tloss, tacc, tf1 = train_one_epoch(classifier, train_loader, criterion, optimizer, scaler, augmenter, use_augment)
        vloss, vacc, vf1, vpreds, vlabels = evaluate(classifier, valid_loader, criterion)

        history['train_loss'].append(tloss); history['val_loss'].append(vloss)
        history['train_acc'].append(tacc); history['val_acc'].append(vacc)
        history['train_f1'].append(tf1); history['val_f1'].append(vf1)

        print(f"Epoch {epoch}/{CFG['epochs']}  Train Acc: {tacc:.4f}  Val Acc: {vacc:.4f}  Train Loss: {tloss:.4f}  Val Loss: {vloss:.4f}")

        # Save best
        if vacc > best_val + 1e-6:
            best_val = vacc
            torch.save({'model_state': classifier.state_dict(), 'classes': classes, 'cfg': CFG}, MODEL_PATH)
            print("  -> Saved best model (val improved).")

        # Early stop based on 96% threshold
        if tacc >= CFG["target_acc"] and vacc >= CFG["target_acc"]:
            print(f"\nEARLY STOP: train_acc ({tacc:.4f}) >= {CFG['target_acc']} AND val_acc ({vacc:.4f}) >= {CFG['target_acc']}")
            break
        if vacc >= CFG["target_acc"]:
            print(f"\nEARLY STOP: val_acc ({vacc:.4f}) >= {CFG['target_acc']}")
            break

    elapsed = time.time() - start
    print(f"\nTraining finished in {elapsed/60:.2f} minutes. Best val acc: {best_val:.4f}")

    # Load best model & evaluate on train/val/test
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    classifier.load_state_dict(ckpt['model_state'])
    classifier.to(DEVICE); classifier.eval()

    tr_loss, tr_acc, tr_f1, tr_preds, tr_labels = evaluate(classifier, train_loader, criterion)
    v_loss, v_acc, v_f1, v_preds, v_labels = evaluate(classifier, valid_loader, criterion)
    te_loss, te_acc, te_f1, te_preds, te_labels = evaluate(classifier, test_loader, criterion)

    print("\n=== FINAL RESULTS ===")
    print(f"Train Acc: {tr_acc:.4f}  Val Acc: {v_acc:.4f}  Test Acc: {te_acc:.4f}")
    print(f"Train F1 : {tr_f1:.4f}  Val F1 : {v_f1:.4f}  Test F1 : {te_f1:.4f}")

    # Save metrics and plots
    plot_training_curves(history, os.path.join(OUT_DIR, "training_curves.png"))
    cm = confusion_matrix(v_labels, v_preds)
    plot_confusion(cm, classes, os.path.join(OUT_DIR, "confusion_matrix_val.png"))

    # Classification report (validation)
    report = classification_report(v_labels, v_preds, target_names=classes, digits=4)
    with open(os.path.join(OUT_DIR, "classification_report_val.txt"), "w") as f:
        f.write(report)

    # Save final model (also saved best), also save inference helper files
    torch.save({"model_state": classifier.state_dict(), "classes": classes, "cfg": CFG}, os.path.join(OUT_DIR, "final_model.pth"))

    print("\nArtifacts saved to:", OUT_DIR)
    return classifier, classes

if __name__ == "__main__":
    main()
