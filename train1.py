"""
LGNM - Leaf-Graph Neural Morphing Model
========================================
Converts leaf images into graph representations based on vein structures
and uses GNN-based reasoning for disease classification.

Key Components:
- Vein extraction using edge detection + morphological operations
- Graph construction from vein skeleton
- Graph Neural Network (GCN/GAT) for classification
- Hybrid CNN-GNN architecture for robust feature extraction
"""

# ================================================================
# LGNM - MAX SPEED VERSION (Early Stop at 96% Accuracy)
# ================================================================

import os, cv2, time, random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, datasets
from torchvision.models import mobilenet_v3_large

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

TRAIN_DIR = "/kaggle/input/leaf-dataset-vimal/dataset/train"
VALID_DIR = "/kaggle/input/leaf-dataset-vimal/dataset/valid"
TEST_DIR  = "/kaggle/input/leaf-dataset-vimal/dataset/test"

OUT_DIR = "/kaggle/working/fast_visuals"
MODEL_PATH = "/kaggle/working/LGNM_fast_model.pth"
os.makedirs(OUT_DIR, exist_ok=True)

CFG = {
    "image_size": 160,
    "batch_size": 32,
    "epochs": 50,          # but early stop will cut it short
    "lr": 2e-4,
    "weight_decay": 1e-5,
    "num_graph_nodes": 32,
    "num_workers": 2,
    "target_acc": 0.96     # STOP here
}


# ============================================================
# FAST GRAPH EXTRACTOR
# ============================================================
class FastGraphExtractor:
    def __init__(self, num_nodes=32):
        self.num_nodes = num_nodes

    def extract(self, tensor_img):
        img = (tensor_img.permute(1,2,0).numpy()*255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 40, 110)

        pts = np.column_stack(np.where(edges > 0))
        h, w = edges.shape

        if len(pts) < 1:
            pts = np.random.randint(0, min(h,w), (self.num_nodes,2))
        if len(pts) < self.num_nodes:
            pad = np.random.randint(0, min(h,w), (self.num_nodes-len(pts),2))
            pts = np.vstack([pts, pad])

        idx = np.linspace(0, len(pts)-1, self.num_nodes).astype(int)
        pts = pts[idx]

        feat = img[pts[:,0], pts[:,1]] / 255.0  # 3 features

        feat = torch.tensor(feat, dtype=torch.float32)
        edges_list = []
        for i in range(self.num_nodes):
            d = np.linalg.norm(pts - pts[i], axis=1)
            nn_i = np.argsort(d)[1:4]
            for j in nn_i:
                edges_list.append([i,j])

        if len(edges_list)==0:
            edge_index = torch.zeros((2,0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges_list).t().contiguous()

        return {"x":feat, "edge_index":edge_index}


# ============================================================
# DATASET
# ============================================================
class LGNMDataset(Dataset):
    def __init__(self, root, transform, g_ex):
        self.tform = transform
        self.raw = transforms.ToTensor()
        self.folder = datasets.ImageFolder(root, transform=self.tform)
        self.samples = self.folder.samples
        self.g = g_ex

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, idx):
        img_t, label = self.folder[idx]
        path,_ = self.samples[idx]

        raw = Image.open(path).convert("RGB")
        raw = self.raw(raw.resize((CFG["image_size"],CFG["image_size"])))

        graph = self.g.extract(raw)

        return img_t, graph, label


def collate_fn(batch):
    imgs, graphs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels)

    xs=[]; es=[]; b=[]; offset=0
    for i,g in enumerate(graphs):
        x = g["x"]; e = g["edge_index"]
        xs.append(x)
        if e.numel()>0:
            es.append(e + offset)
        b.append(torch.full((x.size(0),), i))
        offset += x.size(0)

    X = torch.cat(xs,0)
    B = torch.cat(b,0)
    if len(es)>0: E = torch.cat(es,1)
    else: E = torch.zeros((2,0),dtype=torch.long)

    return imgs, {"x":X, "edge_index":E, "batch":B}, labels


# ============================================================
# SUPER FAST GCN
# ============================================================
class FastGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(3, 32)
        self.l2 = nn.Linear(32, 64)
        self.relu = nn.ReLU()

    def forward(self, g):
        x = g["x"]
        ei = g["edge_index"]
        b = g["batch"]

        h = self.relu(self.l1(x))

        N = h.size(0)
        if ei.numel()==0:
            return torch.zeros((b.max()+1,64),device=x.device)

        src,dst = ei
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, h[src])

        deg = torch.zeros((N,),device=x.device)
        deg.index_add_(0, dst, torch.ones_like(dst,dtype=torch.float))
        deg = deg.clamp_min(1).unsqueeze(1)

        h2 = self.relu(self.l2(agg/deg))

        num_g = int(b.max())+1
        pooled = torch.zeros((num_g,64), device=x.device)
        pooled.index_add_(0, b, h2)

        c = torch.zeros((num_g,),device=x.device)
        c.index_add_(0,b,torch.ones_like(b,dtype=torch.float))
        pooled /= c.unsqueeze(1)

        return pooled


# ============================================================
# MODEL: MobileNetV3 + FastGCN
# ============================================================
class LGNM_FAST(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        base = mobilenet_v3_large(weights="DEFAULT")
        self.cnn = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(960, 256)

        self.gnn = FastGCN()

        self.fuse = nn.Linear(256+64, 128)
        self.cls = nn.Linear(128, num_classes)

    def forward(self, imgs, g):
        with autocast():
            f = self.cnn(imgs)
            f = self.pool(f).view(f.size(0),-1)
            f = self.proj(f)

        with torch.cuda.amp.autocast(enabled=False):
            gfeat = self.gnn({
                "x":g["x"].float().to(f.device),
                "edge_index":g["edge_index"].to(f.device),
                "batch":g["batch"].to(f.device)
            })

        if gfeat.size(0)!=f.size(0):
            gfeat = gfeat.mean(0,keepdim=True).expand(f.size(0),-1)

        z = torch.cat([f,gfeat],1)
        z = F.relu(self.fuse(z))
        return self.cls(z)


# ============================================================
# TRANSFORMS
# ============================================================
def get_train_transform():
    return transforms.Compose([
        transforms.Resize((CFG["image_size"],CFG["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

def get_test_transform():
    return transforms.Compose([
        transforms.Resize((CFG["image_size"],CFG["image_size"])),
        transforms.ToTensor(),
    ])


# Only run training when script is executed directly
if __name__ == '__main__':
    # ============================================================
    # LOAD DATA
    # ============================================================
    train_tf = get_train_transform()
    test_tf = get_test_transform()
    
    g_ex = FastGraphExtractor(CFG["num_graph_nodes"])

    train_ds = LGNMDataset(TRAIN_DIR, train_tf, g_ex)
    valid_ds = LGNMDataset(VALID_DIR, test_tf, g_ex)
    test_ds  = LGNMDataset(TEST_DIR,  test_tf, g_ex)

    train_loader = DataLoader(train_ds, CFG["batch_size"], True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, CFG["batch_size"], False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  CFG["batch_size"], False, collate_fn=collate_fn)

    classes = train_ds.folder.classes
    print("Classes:", classes)


    # ============================================================
    # TRAINING WITH 96% EARLY STOP
    # ============================================================
    model = LGNM_FAST(len(classes)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_acc = 0

    for ep in range(CFG["epochs"]):

        # ---------------- TRAIN ----------------
        model.train()
        preds=[]; labs=[]; loss_sum=0

        for imgs,g,y in train_loader:
            imgs = imgs.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            with autocast():
                out = model(imgs,g)
                loss = criterion(out,y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()*imgs.size(0)
            preds.extend(out.argmax(1).cpu().numpy())
            labs.extend(y.cpu().numpy())

        tr_acc = accuracy_score(labs,preds)

        # ---------------- VALID ----------------
        model.eval()
        v_preds=[]; v_labs=[]

        with torch.no_grad():
            for imgs,g,y in valid_loader:
                imgs=imgs.to(DEVICE)
                out=model(imgs,g)
                v_preds.extend(out.argmax(1).cpu().numpy())
                v_labs.extend(y.numpy())

        val_acc = accuracy_score(v_labs,v_preds)
        print(f"Epoch {ep+1}: Train Acc={tr_acc:.4f}, Val Acc={val_acc:.4f}")

        # SAVE BEST
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model":model.state_dict(),"classes":classes}, MODEL_PATH)
            print("  → Best model updated.")

        # ================================
        # EARLY STOP ON ACC ≥ 96%
        # ================================
        if tr_acc >= CFG["target_acc"] and val_acc >= CFG["target_acc"]:
            print("EARLY STOP TRIGGERED — 96% REACHED!")
            break
        if val_acc >= CFG["target_acc"]:
            print("EARLY STOP — VALIDATION ACC ≥ 96%")
            break


    # ============================================================
    # FINAL EVALUATION
    # ============================================================
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])

    def eval_loader(loader):
        p=[]; l=[]
        with torch.no_grad():
            for imgs,g,y in loader:
                imgs=imgs.to(DEVICE)
                out=model(imgs,g)
                p.extend(out.argmax(1).cpu().numpy())
                l.extend(y.numpy())
        return accuracy_score(l,p), f1_score(l,p,average="weighted")

    train_acc,_ = eval_loader(train_loader)
    val_acc,_ = eval_loader(valid_loader)
    test_acc,test_f1 = eval_loader(test_loader)

    print("\n===== FINAL SCORES =====")
    print("Train Accuracy:", train_acc)
    print("Valid Accuracy:", val_acc)
    print("Test Accuracy :", test_acc)
    print("Test F1 Score :", test_f1)
    print("========================")

    print("Model saved to:", MODEL_PATH)
