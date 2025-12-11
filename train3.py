"""
S-ViT Lite - Spectral-Aware Vision Transformer (Lite)
======================================================
Fuses RGB + pseudo-NIR channels using a spectral attention head
for enhanced plant disease detection.

Key Components:
- Pseudo-NIR channel generation from RGB
- Spectral attention mechanism
- Lightweight Vision Transformer architecture
- Multi-spectral feature fusion
"""

# =====================================================================
#            S-ViT Lite FAST (MODIFIED VERSION) - FULL SCRIPT
#   Saves both separate files and a single unified file containing both.
# =====================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms, datasets
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# CONFIG (EXTREME SPEED MODE)
# =====================================================================

CONFIG = {
    'train_dir': '/kaggle/input/leaf-dataset-vimal/dataset/train',   # update if needed
    'valid_dir': '/kaggle/input/leaf-dataset-vimal/dataset/valid',
    'test_dir':  '/kaggle/input/leaf-dataset-vimal/dataset/test',

    'image_size': 160,
    'patch_size': 16,
    'embed_dim': 256,
    'heads': 4,
    'layers': 3,
    'mlp_ratio': 3,
    'dropout': 0.1,

    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'num_workers': 2,

    'patience': 7,
    'target_acc': 0.96,

    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': '/kaggle/working/SViT_Lite_Fast_Modified'
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)
DEVICE = torch.device(CONFIG['device'])
print("DEVICE:", DEVICE)

# =====================================================================
# PSEUDO-NIR GENERATOR (FAST)
# =====================================================================

class PseudoNIR(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        r = x[:,0:1]; g = x[:,1:2]; b = x[:,2:3]
        veg = torch.clamp(2*g - r - b, 0, 1)
        return 0.7*self.net(x) + 0.3*veg

# =====================================================================
# SPECTRAL ATTENTION (FAST)
# =====================================================================

class SpectralAttention(nn.Module):
    def __init__(self, ch=4):
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch, 1),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(ch, 1, 7, padding=3),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x * self.channel_gate(x)
        x = x * self.spatial_gate(x)
        return x

# =====================================================================
# PATCH EMBEDDING
# =====================================================================

class PatchEmbedding(nn.Module):
    def __init__(self, img=160, patch=16, ch=4, dim=256):
        super().__init__()
        self.num_patches = (img // patch) ** 2
        self.proj = nn.Conv2d(ch, dim, patch, patch)
    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)

# =====================================================================
# TRANSFORMER BLOCK (LIGHT)
# =====================================================================

class MHA(nn.Module):
    def __init__(self, dim, heads=4, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return x + self.drop(out)

class MLP(nn.Module):
    def __init__(self, dim, ratio=3, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim*ratio)
        self.fc2 = nn.Linear(dim*ratio, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        return x + self.drop(self.fc2(F.gelu(self.fc1(x))))

class Transformer(nn.Module):
    def __init__(self, dim, heads, ratio, drop):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.att = MHA(dim, heads, drop)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, ratio, drop)
    def forward(self, x):
        x = self.att(self.n1(x))
        x = self.mlp(self.n2(x))
        return x

# =====================================================================
# S-ViT LITE FAST MODEL (MODIFIED)
# =====================================================================

class SViT_Lite_Fast_Modified(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.nir = PseudoNIR()
        self.satt = SpectralAttention(4)

        self.patch = PatchEmbedding(CONFIG['image_size'], CONFIG['patch_size'], 4, CONFIG['embed_dim'])
        N = self.patch.num_patches
        D = CONFIG['embed_dim']

        self.cls = nn.Parameter(torch.zeros(1, 1, D))
        self.pos = nn.Parameter(torch.zeros(1, N + 1, D))

        self.blocks = nn.ModuleList([
            Transformer(D, CONFIG['heads'], CONFIG['mlp_ratio'], CONFIG['dropout'])
            for _ in range(CONFIG['layers'])
        ])

        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, num_classes)

        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        nir = self.nir(x)
        x = torch.cat([x, nir], 1)
        x = self.satt(x)

        x = self.patch(x)
        B = x.size(0)

        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], 1)
        x = x + self.pos

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x[:,0])

# =====================================================================
# DATASET + TRANSFORMS
# =====================================================================

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((CONFIG["image_size"]+16, CONFIG["image_size"]+16)),
            transforms.RandomResizedCrop(CONFIG["image_size"], scale=(0.8,1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2,0.2,0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

class SpectralDataset(Dataset):
    def __init__(self, root, transform=None):
        self.df = datasets.ImageFolder(root)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        img, lbl = self.df[i]
        if self.transform: img = self.transform(img)
        return img, lbl
    @property
    def classes(self): return self.df.classes

def get_class_weights(ds):
    labels = [lbl for _, lbl in ds.df.samples]
    c = Counter(labels)
    total = len(labels)
    return [total / c[lbl] for _, lbl in ds.df.samples]

# =====================================================================
# TRAIN/EVAL
# =====================================================================

def train_epoch(model, loader, crit, opt, scaler):
    model.train()
    preds=[]; labs=[]; total_loss=0.0
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        with autocast():
            out = model(x)
            loss = crit(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total_loss += loss.item() * x.size(0)
        preds += out.argmax(1).cpu().numpy().tolist()
        labs += y.cpu().numpy().tolist()
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(labs, preds)
    f1 = f1_score(labs, preds, average='weighted')
    return avg_loss, acc, f1

def evaluate(model, loader, crit):
    model.eval()
    preds=[]; labs=[]; total_loss=0.0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = crit(out, y)
            total_loss += loss.item() * x.size(0)
            preds += out.argmax(1).cpu().numpy().tolist()
            labs += y.cpu().numpy().tolist()
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(labs, preds)
    f1 = f1_score(labs, preds, average='weighted')
    return avg_loss, acc, f1, preds, labs

# =====================================================================
# PLOTTING HELPERS
# =====================================================================

def plot_curves(h, path):
    plt.figure(figsize=(10,3))
    plt.subplot(1,3,1); plt.plot(h['tl']); plt.plot(h['vl']); plt.title("Loss")
    plt.subplot(1,3,2); plt.plot(h['ta']); plt.plot(h['va']); plt.title("Acc")
    plt.subplot(1,3,3); plt.plot(h['tf']); plt.plot(h['vf']); plt.title("F1")
    plt.tight_layout(); plt.savefig(path); plt.close()

# =====================================================================
# MAIN
# =====================================================================

def main():
    print("\n=== LOADING DATA ===\n")
    train_ds = SpectralDataset(CONFIG['train_dir'], get_transforms(True))
    valid_ds = SpectralDataset(CONFIG['valid_dir'], get_transforms(False))
    test_ds  = SpectralDataset(CONFIG['test_dir'],  get_transforms(False))

    sampler = WeightedRandomSampler(get_class_weights(train_ds), len(train_ds))

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], sampler=sampler, num_workers=CONFIG['num_workers'], pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=CONFIG['batch_size'], shuffle=False, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG['batch_size'], shuffle=False, pin_memory=True)

    classes = train_ds.classes
    print("Classes:", len(classes))

    # Model
    model = SViT_Lite_Fast_Modified(len(classes)).to(DEVICE)
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt  = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 10)
    scaler = GradScaler()

    history = {'tl':[], 'ta':[], 'tf':[], 'vl':[], 'va':[], 'vf':[]}
    best_val = 0.0
    patience = 0

    print("\n=== TRAINING STARTED ===\n")
    for epoch in range(CONFIG['epochs']):
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, crit, opt, scaler)
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(model, valid_loader, crit)
        sched.step()

        history['tl'].append(tr_loss); history['ta'].append(tr_acc); history['tf'].append(tr_f1)
        history['vl'].append(val_loss); history['va'].append(val_acc); history['vf'].append(val_f1)

        print(f"Epoch {epoch+1}/{CONFIG['epochs']}  Train Acc: {tr_acc:.4f}  Val Acc: {val_acc:.4f}  Train Loss: {tr_loss:.4f}  Val Loss: {val_loss:.4f}")

        # Save best checkpoint (validation-driven)
        if val_acc > best_val + 1e-8:
            best_val = val_acc
            patience = 0
            best_path = os.path.join(CONFIG['save_dir'], "S-ViT_Lite_Fast_Modified_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  -> Saved best model (val_acc={val_acc:.4f}) to {best_path}")
        else:
            patience += 1

        # Early stop if target reached
        if (tr_acc >= CONFIG['target_acc'] and val_acc >= CONFIG['target_acc']) or (val_acc >= CONFIG['target_acc']):
            print(f"\n*** EARLY STOP: target {CONFIG['target_acc']*100:.0f}% reached (train={tr_acc:.4f}, val={val_acc:.4f}) ***\n")
            break

        if patience >= CONFIG['patience']:
            print(f"\nEarly stopping due to no improvement for {CONFIG['patience']} epochs.\n")
            break

    # After training, always save final state
    final_path = os.path.join(CONFIG['save_dir'], "S-ViT_Lite_Fast_Modified_final.pth")
    torch.save(model.state_dict(), final_path)
    print("Saved final model to:", final_path)

    # Ensure best model exists (if not, set best=model)
    best_path = os.path.join(CONFIG['save_dir'], "S-ViT_Lite_Fast_Modified_best.pth")
    if not os.path.exists(best_path):
        torch.save(model.state_dict(), best_path)
        print("No best checkpoint found – saved current final as best.")

    # Unified single-file save containing both best + final + metadata
    best_state = torch.load(best_path, map_location='cpu')
    final_state = torch.load(final_path, map_location='cpu')
    unified = {
        'best_model_state': best_state,
        'final_model_state': final_state,
        'classes': classes,
        'config': CONFIG,
        'best_val_acc': float(best_val)
    }
    unified_path = os.path.join(CONFIG['save_dir'], "S-ViT_Lite_Fast_Modified.pth")
    torch.save(unified, unified_path)
    print("Saved unified checkpoint containing best+final to:", unified_path)

    # Final evaluation on train/val/test using the best model (recommended for inference)
    model.load_state_dict(unified['best_model_state'])
    model.to(DEVICE).eval()

    tr_l, tr_acc, tr_f1, tr_preds, tr_labels = evaluate(model, train_loader, crit)
    v_l, v_acc, v_f1, v_preds, v_labels = evaluate(model, valid_loader, crit)
    te_l, te_acc, te_f1, te_preds, te_labels = evaluate(model, test_loader, crit)

    print("\n=== FINAL METRICS (using BEST checkpoint) ===")
    print(f"Train  Acc: {tr_acc:.4f}  F1: {tr_f1:.4f}")
    print(f"Valid  Acc: {v_acc:.4f}  F1: {v_f1:.4f}")
    print(f"Test   Acc: {te_acc:.4f}  F1: {te_f1:.4f}")

    # Save confusion matrix + classification report (validation)
    cm = confusion_matrix(v_labels, v_preds)
    plt.figure(figsize=(10,8)); plt.imshow(cm, cmap='Blues'); plt.title("Validation Confusion Matrix"); plt.colorbar()
    plt.tight_layout(); plt.savefig(os.path.join(CONFIG['save_dir'], "confusion_matrix.png")); plt.close()

    report = classification_report(v_labels, v_preds, target_names=classes, digits=4)
    with open(os.path.join(CONFIG['save_dir'], "classification_report_val.txt"), "w") as f:
        f.write(report)

    # Save training curves
    plot_curves(history, os.path.join(CONFIG['save_dir'], "training_curves.png"))

    print("\nAll artifacts saved in:", CONFIG['save_dir'])
    print("Files saved:")
    print(" -", best_path)
    print(" -", final_path)
    print(" -", unified_path)
    print("\nDone.")

if __name__ == "__main__":
    main()
