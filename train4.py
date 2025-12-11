"""
ECAM - Explainable Causal Attribution Module
=============================================
Uses counterfactual graph masking to identify causal disease regions
on the leaf for explainable predictions.

Key Components:
- Causal feature extraction with intervention
- Counterfactual masking for explanation
- Graph-based causal reasoning
- Attention-based region attribution
"""

# ============================================================
# ECAM_fast (Option A) - Single Kaggle cell (NO visualizations)
# - Fast/resnet18 backbone
# - causal_dim = 128
# - num_regions = 25 (5x5)
# - mixed precision (AMP)
# - early stop when train, val and test all >= 95%
# - saves:
#     ECAM_fast_best.pth
#     ECAM_fast_final.pth
#     ECAM_fast.pth (unified)
# ============================================================

import os
import random
import warnings
from collections import Counter
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# -------------------------
# User dataset paths (Kaggle)
# -------------------------
TRAIN_DIR = "/kaggle/input/leaf-dataset-vimal/dataset/train"
VALID_DIR = "/kaggle/input/leaf-dataset-vimal/dataset/valid"
TEST_DIR  = "/kaggle/input/leaf-dataset-vimal/dataset/test"

# -------------------------
# Config (Option A: speed)
# -------------------------
CONFIG = {
    "image_size": 224,
    "batch_size": 48,          # reasonably large for T4; reduce if OOM
    "epochs": 20,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "num_regions": 25,         # 5x5
    "causal_dim": 128,
    "dropout": 0.3,
    "patience": 6,
    "target_accuracy": 0.95,   # 95%
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 2,
    "save_dir": "/kaggle/working/ECAM_fast"
}
os.makedirs(CONFIG["save_dir"], exist_ok=True)
DEVICE = torch.device(CONFIG["device"])
print("Device:", DEVICE)

# -------------------------
# Transforms & loaders
# -------------------------
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
            transforms.RandomResizedCrop(CONFIG["image_size"], scale=(0.8,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1,0.1,0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
            transforms.CenterCrop(CONFIG["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])


# -------------------------
# Lightweight causal components (optimized)
# -------------------------
class CausalMaskGenerator(nn.Module):
    def __init__(self, in_ch, num_regions=25):
        super().__init__()
        grid = int(np.sqrt(num_regions))
        self.mask_net = nn.Sequential(
            nn.Conv2d(in_ch, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((grid, grid)),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.causal_strength = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, num_regions),
            nn.Softmax(dim=1)
        )

    def forward(self, features):
        mask = self.mask_net(features)                              # B x 1 x grid x grid
        strength = self.causal_strength(features).clamp(1e-8,1.0)   # B x num_regions
        return mask, strength


class CounterfactualInterventionModule:
    def __init__(self):
        pass

    def intervene(self, features, mask, mode='zero'):
        # features: B x C x H x W ; mask: B x 1 x gh x gw
        mask_resized = F.interpolate(mask, size=features.shape[2:], mode='bilinear', align_corners=False)
        if mode == 'zero':
            return features * (1 - mask_resized)
        elif mode == 'blur':
            blurred = F.avg_pool2d(features, kernel_size=3, stride=1, padding=1)
            return features * (1 - mask_resized) + blurred * mask_resized
        else:
            noise = torch.randn_like(features) * 0.05
            return features * (1 - mask_resized) + (features + noise) * mask_resized


class CausalGraphReasoning(nn.Module):
    def __init__(self, feat_dim, num_regions=25, hidden=128, top_k=6):
        super().__init__()
        self.num_regions = num_regions
        self.top_k = top_k
        self.node_lin = nn.Linear(feat_dim, hidden)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden*2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
        self.msg = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.out = nn.Linear(hidden, feat_dim)

    def forward(self, region_features):
        # region_features: B x N x D
        B, N, D = region_features.shape
        nodes = self.node_lin(region_features)            # B x N x H
        # compute pairwise scores efficiently
        # For efficiency: compute similarity matrix via dot product
        # nodes: B x N x H
        sim = torch.bmm(nodes, nodes.transpose(1,2))     # B x N x N
        # Keep top-k per node (excluding self)
        mask = torch.eye(N, device=sim.device).unsqueeze(0).expand(B, -1, -1) * -1e9
        sim = sim + mask
        topk_vals, topk_idx = sim.topk(min(self.top_k, N-1), dim=-1)  # B x N x K
        # Build aggregated messages
        gathered = []
        for k in range(topk_idx.size(-1)):
            idx = topk_idx[:,:,k]                                         # B x N
            batch_idx = torch.arange(B, device=idx.device).unsqueeze(-1).expand(-1, idx.size(1))
            neigh = nodes[batch_idx, idx]                                 # B x N x H
            gathered.append(neigh)
        neighs = torch.stack(gathered, dim=-2).mean(dim=-2)                # B x N x H
        msgs = self.msg(neighs)                                           # B x N x H
        updated = nodes + msgs
        out = self.out(updated)                                           # B x N x D
        return out, topk_vals


# -------------------------
# ECAM_fast model with ResNet18 backbone
# -------------------------
class ECAM_fast(nn.Module):
    def __init__(self, num_classes, num_regions=25, causal_dim=128, dropout=0.3):
        super().__init__()
        # ResNet18 backbone (fast)
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # remove last layers
        self.features = nn.Sequential(*list(backbone.children())[:-2])    # output C x H x W, C=512
        feat_ch = 512

        # lightweight reducer to causal_dim
        self.reduce = nn.Sequential(
            nn.Conv2d(feat_ch, causal_dim, kernel_size=1),
            nn.BatchNorm2d(causal_dim),
            nn.ReLU(inplace=True)
        )

        self.grid = int(np.sqrt(num_regions))
        self.num_regions = num_regions

        # causal modules
        self.mask_gen = CausalMaskGenerator(causal_dim, num_regions=num_regions)
        self.intervenor = CounterfactualInterventionModule()
        self.causal_graph = CausalGraphReasoning(causal_dim, num_regions=num_regions, hidden=causal_dim//2, top_k=6)

        # classification heads
        self.main_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(causal_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        self.causal_head = nn.Sequential(
            nn.Linear(causal_dim * num_regions, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, return_explanations=False):
        B = x.size(0)
        feat = self.features(x)                 # B x C x H x W
        feat = self.reduce(feat)                # B x causal_dim x h x w

        # Mask + strength
        mask, strength = self.mask_gen(feat)    # mask: B x1 x gh x gw ; strength: B x N

        # apply mask (attended features)
        attended = feat * F.interpolate(mask, size=feat.shape[2:], mode='bilinear', align_corners=False)

        # region pooling -> B x N x D
        pooled = F.adaptive_avg_pool2d(attended, (self.grid, self.grid))          # B x D x gh x gw
        B, D, gh, gw = pooled.shape
        regions = pooled.view(B, D, -1).permute(0,2,1).contiguous()              # B x N x D

        # causal graph reasoning
        causal_feats, edge_info = self.causal_graph(regions)                     # B x N x D

        # heads
        main_logits = self.main_head(attended)                                   # B x C
        causal_flat = causal_feats.view(B, -1)
        causal_logits = self.causal_head(causal_flat)                            # B x C

        logits = 0.75 * main_logits + 0.25 * causal_logits

        if return_explanations:
            return logits, {
                "mask": mask,
                "strength": strength,
                "edge_info": edge_info,
                "attended": attended
            }
        return logits

    def get_causal_explanation(self, x):
        self.eval()
        with torch.no_grad():
            logits, expl = self.forward(x, return_explanations=True)
            pred = logits.argmax(dim=1)
            conf = F.softmax(logits, dim=1).max(dim=1)[0]
        return {"prediction": pred, "confidence": conf, **expl}


# -------------------------
# Losses & training utilities
# -------------------------
def causal_consistency_loss(model, images, labels, criterion):
    # original
    model.eval()
    with torch.no_grad():
        logits, expl = model(images, return_explanations=True)
    model.train()

    # original loss (hard)
    orig_loss = criterion(logits, labels)

    # intervene (zero out causal mask)
    mask = expl["mask"]
    intervened_feat = model.intervenor.intervene(model.reduce(model.features(images)), mask, mode='zero')
    # run main head on intervened features (use classifier part only)
    main_logits = model.main_head(intervened_feat)
    # encourage divergence: KL(original || intervened)
    kl = F.kl_div(F.log_softmax(main_logits, dim=1), F.softmax(logits.detach(), dim=1), reduction='batchmean')
    # combine
    loss = orig_loss + 0.1 * kl
    return loss


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, use_causal=False):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with autocast():
            if use_causal:
                loss = causal_consistency_loss(model, imgs, labels, criterion)
            else:
                out = model(imgs)
                loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)

        # predictions for metrics (compute under no_grad to avoid extra graph)
        with torch.no_grad():
            out = model(imgs)
            preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, acc, f1


def evaluate(model, loader, criterion, device):
    model.eval()
    preds = []
    labs = []
    loss_sum = 0.0
    cnt = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            loss_sum += loss.item() * imgs.size(0)
            cnt += imgs.size(0)
            preds.extend(out.argmax(dim=1).cpu().numpy())
            labs.extend(labels.cpu().numpy())
    if cnt == 0: 
        return 0.0, 0.0, 0.0, [], []
    acc = accuracy_score(labs, preds)
    f1 = f1_score(labs, preds, average='weighted')
    return loss_sum / cnt, acc, f1, preds, labs


# -------------------------
# Main training (only runs when executed directly)
# -------------------------
if __name__ == '__main__':
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=get_transforms(train=True))
    valid_dataset = datasets.ImageFolder(VALID_DIR, transform=get_transforms(train=False))
    test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=get_transforms(train=False))

    print("Classes:", train_dataset.classes)
    print("Samples - train:", len(train_dataset), "valid:", len(valid_dataset), "test:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
                              num_workers=CONFIG["num_workers"], pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
                              num_workers=CONFIG["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=CONFIG["batch_size"], shuffle=False,
                              num_workers=CONFIG["num_workers"], pin_memory=True)

    # -------------------------
    # Prepare model / optimizer / scaler
    # -------------------------
    num_classes = len(train_dataset.classes)
    model = ECAM_fast(num_classes=num_classes, num_regions=CONFIG["num_regions"],
                      causal_dim=CONFIG["causal_dim"], dropout=CONFIG["dropout"]).to(DEVICE)

    # Use only a little weight decay, use AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, CONFIG["epochs"]))
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    print("Model params:", sum(p.numel() for p in model.parameters()))

    # -------------------------
    # Training loop with early stopping
    # -------------------------
    history = {"train_loss":[], "train_acc":[], "train_f1":[], "val_loss":[], "val_acc":[], "val_f1":[], "test_acc":[]}
    best_val = -1.0
    patience = 0
    best_state = None
    final_state = None

    for epoch in range(CONFIG["epochs"]):
        use_causal = epoch >= 3   # enable causal consistency loss after a couple epochs
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, optimizer, scaler, criterion, DEVICE, use_causal)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, valid_loader, criterion, DEVICE)
        test_loss, test_acc, test_f1, _, _ = evaluate(model, test_loader, criterion, DEVICE)

        scheduler.step()

        history["train_loss"].append(train_loss); history["train_acc"].append(train_acc); history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc); history["val_f1"].append(val_f1)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch+1}/{CONFIG['epochs']}  Train acc: {train_acc:.4f}  Val acc: {val_acc:.4f}  Test acc: {test_acc:.4f}")

        # save best by validation acc
        if val_acc > best_val + 1e-8:
            best_val = val_acc
            patience = 0
            best_state = model.state_dict()
            torch.save(best_state, os.path.join(CONFIG["save_dir"], "ECAM_fast_best.pth"))
            print(f"  -> saved best (val acc {val_acc:.4f})")
        else:
            patience += 1

        # early stop if all three >= target
        if train_acc >= CONFIG["target_accuracy"] and val_acc >= CONFIG["target_accuracy"] and test_acc >= CONFIG["target_accuracy"]:
            print(f"\n*** EARLY STOP: reached target {CONFIG['target_accuracy']*100:.1f}% on train, val, test")
            final_state = model.state_dict()
            break

        # patience early stop
        if patience >= CONFIG["patience"]:
            print("\nEarly stopping due to patience")
            final_state = model.state_dict()
            break

    # if not set, fill final_state with last model
    if final_state is None:
        final_state = model.state_dict()
    if best_state is None:
        best_state = final_state
        torch.save(best_state, os.path.join(CONFIG["save_dir"], "ECAM_fast_best.pth"))

    # save final and unified
    torch.save(final_state, os.path.join(CONFIG["save_dir"], "ECAM_fast_final.pth"))
    unified = {
        "best_state": best_state,
        "final_state": final_state,
        "classes": train_dataset.classes,
        "config": CONFIG
    }
    torch.save(unified, os.path.join(CONFIG["save_dir"], "ECAM_fast.pth"))
    print("Saved checkpoints to", CONFIG["save_dir"])

    # final evaluation on best checkpoint
    model.load_state_dict(best_state)
    _, train_acc, train_f1, _, _ = evaluate(model, train_loader, criterion, DEVICE)
    _, val_acc, val_f1, val_preds, val_labels = evaluate(model, valid_loader, criterion, DEVICE)
    _, test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion, DEVICE)

    print("\n=== FINAL (BEST MODEL) METRICS ===")
    print(f"Train Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
    print(f"Val   Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
    print(f"Test  Acc: {test_acc:.4f}, F1: {test_f1:.4f}")

    # save confusion matrix and classification report for validation
    cm = confusion_matrix(val_labels, val_preds)
    plt.figure(figsize=(10,10)); plt.imshow(cm, cmap="Blues"); plt.title("Val Confusion Matrix"); plt.colorbar()
    plt.tight_layout(); plt.savefig(os.path.join(CONFIG["save_dir"], "confusion_matrix_val.png")); plt.close()

    report = classification_report(val_labels, val_preds, target_names=train_dataset.classes, digits=4)
    with open(os.path.join(CONFIG["save_dir"], "classification_report_val.txt"), "w") as f:
        f.write(report)

    print("Artifacts saved (confusion matrix, classification report). Done.")
