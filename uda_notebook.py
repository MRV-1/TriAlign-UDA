# %%
# ============================================================
# TriAlign-UDA — Final Notebook Code
# Hybrid Domain Adaptation for Histopathology Foundation Model Features
# ============================================================
# This file contains the code cells intended for the UDA notebook.
# Copy each # %% section into a separate notebook cell if desired.
# The protocol follows the manuscript setting:
#   - Frozen UNI2-H features
#   - Labeled source domain: NCT-CRC-HE-100K
#   - Unlabeled target domain: TCGA-COAD/READ
#   - External evaluation only: CRC-VAL-HE-7K
#   - Checkpoint selection only on source-validation Macro-F1
# ============================================================


# %%
# Notebook cell 1
# ============================================================
# 1) Imports, paths, config
# ============================================================
import os, json, time, random, math, warnings
from datetime import datetime
from pathlib import Path
from itertools import cycle

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm
from IPython.display import display
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,f1_score,precision_recall_fscore_support,confusion_matrix,)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

def to_wsl_path(p):
    p = str(p)

    # Already WSL/Linux path
    if p.startswith("/mnt/") or p.startswith("/home/"):
        return p

    # Windows path such as C:\home\merve\...
    if len(p) >= 3 and p[1:3] == ":\\":
        drive = p[0].lower()
        rest = p[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"

    return p

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def save_df(df, csv_path, xlsx_path=None):
    ensure_dir(os.path.dirname(csv_path))
    df.to_csv(csv_path, index=False)
    if xlsx_path is not None:
        try:
            df.to_excel(xlsx_path, index=False)
        except Exception as e:
            print(f"XLSX skipped: {e}")

# -------------------------
# Project/cache paths
# -------------------------
# These paths are intentionally configurable so the notebook can be shared on GitHub.
# Option 1: edit PROJECT_ROOT and CACHE_DIR directly below.
# Option 2: set environment variables before running the notebook:
#   export TRIALIGN_PROJECT_ROOT=/path/to/TriAlign-UDA
#   export TRIALIGN_CACHE_DIR=/path/to/uni2_h_feature_cache
PROJECT_ROOT = os.environ.get(
    "TRIALIGN_PROJECT_ROOT",
    to_wsl_path(r"C:\home\merve\UNI2-h\TriAlign-UDA")
)
CACHE_DIR = os.environ.get(
    "TRIALIGN_CACHE_DIR",
    "/mnt/c/home/merve/UNI2-h/cache/uni2_h"
)

NCT_X_PATH  = os.path.join(CACHE_DIR, "nct_rgb.npy")
NCT_Y_PATH  = os.path.join(CACHE_DIR, "nct_labels_trafiq_order.npy")
CRC_X_PATH  = os.path.join(CACHE_DIR, "crc7k_rgb.npy")
CRC_Y_PATH  = os.path.join(CACHE_DIR, "crc7k_labels_trafiq_order.npy")
TCGA_X_PATH = os.path.join(CACHE_DIR, "tcga_rgb.npy")
CLASS_NAMES_PATH = os.path.join(CACHE_DIR, "class_names_trafiq_order.npy")

# -------------------------
# Run configuration
# -------------------------
# This notebook implements the final paper protocol.
# Best checkpoint selection:
#   NCT-CRC-HE-100K is split into source-train and source-validation subsets.
#   The best checkpoint is selected only by source-validation Macro-F1.
# External evaluation:
#   CRC-VAL-HE-7K is never used during training, adaptation, hyperparameter tuning,
#   or checkpoint selection. It is used only for independent external evaluation.
RUN_ID = "trialign_uda_final_" + datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = ensure_dir(os.path.join(PROJECT_ROOT, "runs", RUN_ID))
CKPT_DIR = ensure_dir(os.path.join(RUN_DIR, "checkpoints"))
LOG_DIR = ensure_dir(os.path.join(RUN_DIR, "logs"))
TABLE_DIR = ensure_dir(os.path.join(RUN_DIR, "tables"))
PLOT_DIR = ensure_dir(os.path.join(RUN_DIR, "plots"))
CONFIG_DIR = ensure_dir(os.path.join(RUN_DIR, "config"))

# -------------------------
# Training config
# -------------------------
FEATURE_DIM = 1536
NUM_CLASSES = 9
ADAPTER_REDUCTION = 16

SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 5
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
TCGA_TARGET_N = 10000
TCGA_DATA_SEED = 42

SOURCE_VAL_SIZE = 0.10
SOURCE_SPLIT_SEED = 42

LR = 5e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
USE_AMP = False
RAMP_EPOCHS = 2

ALIGN_NUM_BATCHES_DURING_TRAIN = 25
ALIGN_NUM_BATCHES_FINAL = 50
PROTO_MAX_BATCHES = 200
KNN_K = 25
PAD_MAX_SAMPLES = 5000

# Final TriAlign-UDA configuration used in the manuscript:
#   CE + CORAL + MK-MMD + prototype-based semantic regularization + adversarial alignment.
# Prototype regularization starts from epoch 3 because early prototypes can be unstable.
TRIALIGN_UDA_CONFIG = dict(
    coral=0.05,
    mkmmd=0.05,
    proto=0.10,
    dann=0.05,
    proto_start=3
)

METHOD_CONFIGS = {
    # Ablation variants
    "B0_CE":                 dict(coral=0.00, mkmmd=0.00, proto=0.00, dann=0.00, proto_start=999),
    "B1_CE_CORAL":          dict(coral=0.05, mkmmd=0.00, proto=0.00, dann=0.00, proto_start=999),
    "B2_CE_CORAL_MKMMD":    dict(coral=0.05, mkmmd=0.05, proto=0.00, dann=0.00, proto_start=999),
    "B3_STAT_PROTO":        dict(coral=0.05, mkmmd=0.05, proto=0.10, dann=0.00, proto_start=3),
    "TriAlign_UDA":         TRIALIGN_UDA_CONFIG,

    # Baseline methods
    "SourceOnly":           dict(coral=0.00, mkmmd=0.00, proto=0.00, dann=0.00, proto_start=999),
    "DeepCORAL":            dict(coral=0.05, mkmmd=0.00, proto=0.00, dann=0.00, proto_start=999),
    "DAN":                  dict(coral=0.00, mkmmd=0.05, proto=0.00, dann=0.00, proto_start=999),
    "DANN":                 dict(coral=0.00, mkmmd=0.00, proto=0.00, dann=0.10, proto_start=999),
}

METHODS_TO_RUN = [
    "B0_CE", "B1_CE_CORAL", "B2_CE_CORAL_MKMMD", "B3_STAT_PROTO", "TriAlign_UDA",
    "SourceOnly", "DeepCORAL", "DAN", "DANN"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("RUN_ID:", RUN_ID)
print("RUN_DIR:", RUN_DIR)
print("Device:", device)
print("Seeds:", SEEDS)
print("Methods:", METHODS_TO_RUN)

CONFIG = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (str, int, float, bool, list, dict))}
save_json(CONFIG, os.path.join(CONFIG_DIR, "config.json"))


# %%
# Notebook cell 2
# ============================================================
# 2) Data loading
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class CachedFeatureDataset(Dataset):
    def __init__(self, X, y=None, indices=None):
        self.X = X
        self.y = y
        self.indices = np.asarray(indices) if indices is not None else None
    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.X)
    def __getitem__(self, i):
        j = int(self.indices[i]) if self.indices is not None else i
        x = torch.from_numpy(np.asarray(self.X[j], dtype=np.float32).copy())
        if self.y is None:
            return x
        return x, int(self.y[j])

def make_loaders():
    Xs = np.load(NCT_X_PATH, mmap_mode="r")
    ys = np.load(NCT_Y_PATH)

    Xv = np.load(CRC_X_PATH, mmap_mode="r")
    yv = np.load(CRC_Y_PATH)

    Xt = np.load(TCGA_X_PATH, mmap_mode="r")
    class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True).tolist()

    # ------------------------------------------------------------
    # Source-domain split for checkpoint selection
    # NCT-CRC-HE-100K is split into source train and source validation.
    # CRC-VAL-HE-7K is kept fully external.
    # ------------------------------------------------------------
    all_src_idx = np.arange(len(ys))

    src_train_idx, src_val_idx = train_test_split(
        all_src_idx,
        test_size=SOURCE_VAL_SIZE,
        random_state=SOURCE_SPLIT_SEED,
        stratify=ys
    )

    src_train_idx = np.asarray(src_train_idx)
    src_val_idx = np.asarray(src_val_idx)

    np.save(
        os.path.join(CONFIG_DIR, f"nct_source_train_indices_seed{SOURCE_SPLIT_SEED}.npy"),
        src_train_idx
    )
    np.save(
        os.path.join(CONFIG_DIR, f"nct_source_val_indices_seed{SOURCE_SPLIT_SEED}.npy"),
        src_val_idx
    )

    # ------------------------------------------------------------
    # Fixed unlabeled TCGA target subset
    # ------------------------------------------------------------
    rng = np.random.default_rng(TCGA_DATA_SEED)
    n_t = min(TCGA_TARGET_N, len(Xt))
    tcga_idx = np.sort(rng.choice(len(Xt), size=n_t, replace=False))

    np.save(
        os.path.join(CONFIG_DIR, f"tcga_indices_seed{TCGA_DATA_SEED}_n{n_t}.npy"),
        tcga_idx
    )

    src_train_ds = CachedFeatureDataset(Xs, ys, src_train_idx)
    src_val_ds = CachedFeatureDataset(Xs, ys, src_val_idx)
    crc_ext_ds = CachedFeatureDataset(Xv, yv)
    tgt_ds = CachedFeatureDataset(Xt, None, tcga_idx)

    src_loader = DataLoader(
        src_train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    src_val_loader = DataLoader(
        src_val_ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    crc_ext_loader = DataLoader(
        crc_ext_ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    tgt_loader = DataLoader(
        tgt_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    print("NCT total:", Xs.shape)
    print("NCT source train:", len(src_train_ds))
    print("NCT source val:", len(src_val_ds))
    print("CRC7K external:", Xv.shape)
    print("TCGA selected:", len(tgt_ds))
    print("Classes:", class_names)

    return src_loader, src_val_loader, crc_ext_loader, tgt_loader, class_names


src_loader, src_val_loader, crc_ext_loader, tgt_loader, CLASS_NAMES = make_loaders()


# %%
# Notebook cell 3
# ============================================================
# 3) Models and losses
# ============================================================
class BottleneckAdapter(nn.Module):
    def __init__(self, dim=1536, reduction=16):
        super().__init__()
        h = max(dim // reduction, 64)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, h), nn.GELU(),
            nn.Linear(h, dim),
        )
    def forward(self, x):
        return x + self.net(x)

class LinearHead(nn.Module):
    def __init__(self, dim=1536, num_classes=9):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)
    def forward(self, x):
        return self.fc(x)

class DomainDiscriminator(nn.Module):
    def __init__(self, dim=1536, hidden=512, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
    def forward(self, x):
        return self.net(x)

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grl(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

def build_model(use_dann=False):
    adapter = BottleneckAdapter(FEATURE_DIM, ADAPTER_REDUCTION).to(device)
    head = LinearHead(FEATURE_DIM, NUM_CLASSES).to(device)
    disc = DomainDiscriminator(FEATURE_DIM).to(device) if use_dann else None
    return adapter, head, disc

def linear_ramp(epoch, ramp_epochs=2):
    return 1.0 if ramp_epochs <= 0 else min(1.0, epoch / float(ramp_epochs))

def coral_loss(fs, ft):
    fs = fs - fs.mean(0, keepdim=True)
    ft = ft - ft.mean(0, keepdim=True)
    cs = fs.t().matmul(fs) / max(fs.size(0) - 1, 1)
    ct = ft.t().matmul(ft) / max(ft.size(0) - 1, 1)
    return ((cs - ct) ** 2).mean()

def _gaussian_kernel(x, y, sigma):
    x2 = (x ** 2).sum(1, keepdim=True)
    y2 = (y ** 2).sum(1, keepdim=True)
    d = x2 - 2 * x.matmul(y.t()) + y2.t()
    return torch.exp(-d / (2.0 * sigma * sigma))

def mkmmd_loss(fs, ft, sigmas=(1, 2, 4, 8, 16)):
    kxx = kyy = kxy = 0.0
    for s in sigmas:
        kxx = kxx + _gaussian_kernel(fs, fs, s)
        kyy = kyy + _gaussian_kernel(ft, ft, s)
        kxy = kxy + _gaussian_kernel(fs, ft, s)
    return kxx.mean() + kyy.mean() - 2.0 * kxy.mean()

@torch.no_grad()
def compute_source_prototypes(adapter, loader, max_batches=200):
    adapter.eval()
    sums = torch.zeros(NUM_CLASSES, FEATURE_DIM, device=device)
    counts = torch.zeros(NUM_CLASSES, device=device)
    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        f = adapter(x)
        for c in y.unique():
            m = (y == c)
            sums[int(c)] += f[m].sum(0)
            counts[int(c)] += m.sum()
    return sums / counts.unsqueeze(1).clamp_min(1.0)

def proto_loss(fs, ys, prototypes, temperature=0.07):
    logits = F.normalize(fs, dim=1).matmul(F.normalize(prototypes, dim=1).t()) / temperature
    return F.cross_entropy(logits, ys)

def domain_loss(domain_disc, fs, ft, lambd):
    f = torch.cat([grl(fs, lambd), grl(ft, lambd)], dim=0)
    y = torch.cat([
        torch.zeros(fs.size(0), dtype=torch.long, device=device),
        torch.ones(ft.size(0), dtype=torch.long, device=device),
    ])
    logits = domain_disc(f)
    loss = F.cross_entropy(logits, y)
    acc = (logits.argmax(1) == y).float().mean().item()
    return loss, acc


# %%
# Notebook cell 4
# ============================================================
# 4) Evaluation metrics
# ============================================================
@torch.no_grad()
def eval_classifier(adapter, head, loader, prefix):
    adapter.eval()
    head.eval()

    ys, ps = [], []

    for x, y in loader:
        x = x.to(device)
        pred = head(adapter(x)).argmax(1).cpu().numpy()

        ps.append(pred)
        ys.append(y.numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)

    return {
        f"{prefix}_macroF1": float(f1_score(y_true, y_pred, average="macro")),
        f"{prefix}_balAcc": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}_acc": float(accuracy_score(y_true, y_pred)),
    }

# ------------------------------------------------------------
# CRC-VAL-HE-7K class-wise diagnostics and confusion matrices
# This block does not affect training or checkpoint selection.
# It only saves final evaluation details after the best checkpoint.
# ------------------------------------------------------------
@torch.no_grad()
def collect_crc_predictions(adapter, head, loader):
    adapter.eval()
    head.eval()

    ys, ps = [], []

    for x, y in loader:
        x = x.to(device)

        logits = head(adapter(x))
        pred = logits.argmax(1).detach().cpu().numpy()

        ps.append(pred)
        ys.append(y.numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)

    return y_true, y_pred


def save_crc_classwise_and_cm(adapter, head, method, seed, class_names=None):
    if class_names is None:
        class_names = [f"class_{i}" for i in range(NUM_CLASSES)]
    else:
        class_names = [str(c) for c in class_names]

    y_true, y_pred = collect_crc_predictions(adapter, head, crc_ext_loader)

    labels = np.arange(NUM_CLASSES)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0
    )

    classwise_df = pd.DataFrame({
        "method": method,
        "seed": seed,
        "class_id": labels,
        "class_name": class_names,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in class_names],
        columns=[f"pred_{c}" for c in class_names]
    )

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0
    )

    cm_norm_df = pd.DataFrame(
        cm_norm,
        index=[f"true_{c}" for c in class_names],
        columns=[f"pred_{c}" for c in class_names]
    )

    cm_long_rows = []

    for i, true_name in enumerate(class_names):
        total = cm[i].sum()

        for j, pred_name in enumerate(class_names):
            cm_long_rows.append({
                "method": method,
                "seed": seed,
                "true_class_id": i,
                "true_class": true_name,
                "pred_class_id": j,
                "pred_class": pred_name,
                "count": int(cm[i, j]),
                "row_normalized": float(cm[i, j] / total) if total > 0 else 0.0,
            })

    cm_long_df = pd.DataFrame(cm_long_rows)

    pred_df = pd.DataFrame({
        "method": method,
        "seed": seed,
        "y_true": y_true,
        "y_pred": y_pred,
        "true_class": [class_names[i] for i in y_true],
        "pred_class": [class_names[i] for i in y_pred],
    })

    classwise_dir = ensure_dir(os.path.join(TABLE_DIR, "classwise"))
    cm_dir = ensure_dir(os.path.join(TABLE_DIR, "confusion_matrices"))
    pred_dir = ensure_dir(os.path.join(TABLE_DIR, "crc7k_predictions"))

    save_df(
        classwise_df,
        os.path.join(classwise_dir, f"classwise_{method}_seed{seed}.csv"),
        os.path.join(classwise_dir, f"classwise_{method}_seed{seed}.xlsx")
    )

    save_df(
        cm_df.reset_index().rename(columns={"index": "true_class"}),
        os.path.join(cm_dir, f"cm_counts_{method}_seed{seed}.csv"),
        os.path.join(cm_dir, f"cm_counts_{method}_seed{seed}.xlsx")
    )

    save_df(
        cm_norm_df.reset_index().rename(columns={"index": "true_class"}),
        os.path.join(cm_dir, f"cm_row_normalized_{method}_seed{seed}.csv"),
        os.path.join(cm_dir, f"cm_row_normalized_{method}_seed{seed}.xlsx")
    )

    save_df(
        cm_long_df,
        os.path.join(cm_dir, f"cm_long_{method}_seed{seed}.csv"),
        os.path.join(cm_dir, f"cm_long_{method}_seed{seed}.xlsx")
    )

    # Prediction-level file can be large but useful for later analysis.
    # CSV is enough here.
    pred_df.to_csv(
        os.path.join(pred_dir, f"crc7k_predictions_{method}_seed{seed}.csv"),
        index=False
    )

    return classwise_df, cm_df, cm_norm_df, cm_long_df

@torch.no_grad()
def collect_alignment(adapter, n_batches):
    adapter.eval()
    src_iter, tgt_iter = iter(src_loader), iter(tgt_loader)
    fs_bank, ys_bank, ft_bank = [], [], []
    coral_vals, mmd_vals, proto_vals = [], [], []
    prototypes = compute_source_prototypes(adapter, src_loader, PROTO_MAX_BATCHES)
    proto_norm = F.normalize(prototypes, dim=1)

    for _ in range(n_batches):
        try: xs, ys = next(src_iter)
        except StopIteration:
            src_iter = iter(src_loader); xs, ys = next(src_iter)
        try: xt = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader); xt = next(tgt_iter)

        xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)
        fs, ft = adapter(xs), adapter(xt)

        coral_vals.append(float(coral_loss(fs, ft).item()))
        mmd_vals.append(float(mkmmd_loss(fs, ft).item()))
        proto_vals.append(float((1.0 - F.normalize(ft, dim=1).matmul(proto_norm.t()).max(1).values).mean().item()))

        fs_bank.append(fs.detach().cpu())
        ys_bank.append(ys.detach().cpu())
        ft_bank.append(ft.detach().cpu())

    return {
        "TCGA_CORAL": float(np.mean(coral_vals)),
        "TCGA_MKMMD": float(np.mean(mmd_vals)),
        "TCGA_ProtoDist": float(np.mean(proto_vals)),
        "src_feats": torch.cat(fs_bank),
        "src_labels": torch.cat(ys_bank),
        "tgt_feats": torch.cat(ft_bank),
    }

def _finite_np(X, y=None):
    X = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
    X = X.astype(np.float32)
    m = np.isfinite(X).all(axis=1)
    if y is None:
        return X[m]
    y = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    return X[m], y[m]

def compute_pad(src_feats, tgt_feats, seed=42, max_samples=5000):
    Xs = _finite_np(src_feats); Xt = _finite_np(tgt_feats)
    if len(Xs) < 20 or len(Xt) < 20:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    ns, nt = min(len(Xs), max_samples), min(len(Xt), max_samples)
    Xs = Xs[rng.choice(len(Xs), ns, replace=False)] if len(Xs) > ns else Xs
    Xt = Xt[rng.choice(len(Xt), nt, replace=False)] if len(Xt) > nt else Xt
    X = np.vstack([Xs, Xt])
    y = np.r_[np.zeros(len(Xs), dtype=int), np.ones(len(Xt), dtype=int)]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    clf = LogisticRegression(max_iter=2000, n_jobs=-1, class_weight="balanced", random_state=seed)
    clf.fit(Xtr, ytr)
    acc = float(clf.score(Xte, yte))
    eps = 1.0 - acc
    pad = 2.0 * (1.0 - 2.0 * eps)
    return float(pad), float(eps), float(acc)

def shannon_entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)

@torch.no_grad()
def compute_knn_semantics(src_feats, src_labels, tgt_feats, k=25, chunk_size=1024):
    Xs, ys = _finite_np(src_feats, src_labels)
    Xt = _finite_np(tgt_feats)
    Xs = F.normalize(torch.tensor(Xs, dtype=torch.float32), dim=1)
    Xt = F.normalize(torch.tensor(Xt, dtype=torch.float32), dim=1)
    ys = torch.tensor(ys, dtype=torch.long)

    majorities, entropies = [], []
    for st in range(0, len(Xt), chunk_size):
        sim = Xt[st:st+chunk_size].matmul(Xs.t())
        idx = torch.topk(sim, k=k, dim=1).indices
        labs = ys[idx]
        counts = torch.stack([(labs == c).float().sum(1) for c in range(NUM_CLASSES)], dim=1)
        p = (counts / float(k)).numpy()
        majorities.append(p.max(axis=1))
        entropies.append(shannon_entropy(p) / np.log(NUM_CLASSES))
    return float(np.concatenate(majorities).mean()), float(np.concatenate(entropies).mean())

@torch.no_grad()
def evaluate(adapter, head, seed, n_batches):
    out = {}

    # Internal source validation for checkpoint selection
    out.update(eval_classifier(adapter, head, src_val_loader, "NCTval"))

    # External validation/evaluation on CRC-VAL-HE-7K
    out.update(eval_classifier(adapter, head, crc_ext_loader, "7K"))

    align = collect_alignment(adapter, n_batches)

    pad, eps, dacc = compute_pad(
        align["src_feats"],
        align["tgt_feats"],
        seed=seed,
        max_samples=PAD_MAX_SAMPLES
    )

    maj, ncent = compute_knn_semantics(
        align["src_feats"],
        align["src_labels"],
        align["tgt_feats"],
        k=KNN_K
    )

    out.update({
        "TCGA_CORAL": align["TCGA_CORAL"],
        "TCGA_MKMMD": align["TCGA_MKMMD"],
        "TCGA_ProtoDist": align["TCGA_ProtoDist"],
        "TCGA_PAD": pad,
        "TCGA_PAD_eps": eps,
        "TCGA_PAD_domain_acc": dacc,
        "TCGA_NCEntropy": ncent,
        "TCGA_Majority": maj,
    })

    return out


# %%
# Notebook cell 5
# ============================================================
# 5) Training one method/seed
# ============================================================
def method_ckpt_dir(method, seed):
    return ensure_dir(os.path.join(CKPT_DIR, f"seed{seed}", method))

def save_ckpt(path, adapter, head, disc, meta):
    payload = {
        "adapter": adapter.state_dict(),
        "head": head.state_dict(),
        "disc": disc.state_dict() if disc is not None else None,
        "meta": meta,
        "config": CONFIG,
    }
    torch.save(payload, path)

def load_ckpt(path, adapter, head, disc=None):
    payload = torch.load(path, map_location=device)
    adapter.load_state_dict(payload["adapter"])
    head.load_state_dict(payload["head"])
    if disc is not None and payload.get("disc") is not None:
        disc.load_state_dict(payload["disc"])
    return payload

def train_one(method, seed):
    cfg = METHOD_CONFIGS[method]
    use_dann = cfg["dann"] > 0
    set_seed(seed)
    adapter, head, disc = build_model(use_dann=use_dann)
    params = list(adapter.parameters()) + list(head.parameters()) + ([] if disc is None else list(disc.parameters()))
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and torch.cuda.is_available()))

    best_f1 = -1.0
    best_epoch = None
    best_path = os.path.join(method_ckpt_dir(method, seed), "best_checkpoint.pt")
    logs = []
    tgt_cycle = cycle(tgt_loader)
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        adapter.train(); head.train()
        if disc is not None:
            disc.train()

        r = linear_ramp(epoch, RAMP_EPOCHS)
        w_coral = cfg["coral"] * r
        w_mkmmd = cfg["mkmmd"] * r
        w_dann = cfg["dann"] * r
        w_proto = cfg["proto"] * r if epoch >= cfg.get("proto_start", 1) else 0.0
        prototypes = compute_source_prototypes(adapter, src_loader, PROTO_MAX_BATCHES).detach() if w_proto > 0 else None

        loss_list, domain_acc_list = [], []
        pbar = tqdm(src_loader, desc=f"{method} seed={seed} epoch={epoch}/{EPOCHS}", leave=False, dynamic_ncols=True)
        for xs, ys in pbar:
            xt = next(tgt_cycle)
            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(USE_AMP and torch.cuda.is_available())):
                fs, ft = adapter(xs), adapter(xt)
                loss = F.cross_entropy(head(fs), ys)
                if w_coral > 0:
                    loss = loss + w_coral * coral_loss(fs, ft)
                if w_mkmmd > 0:
                    loss = loss + w_mkmmd * mkmmd_loss(fs, ft)
                if w_proto > 0:
                    loss = loss + w_proto * proto_loss(fs, ys, prototypes)
                if w_dann > 0:
                    dl, da = domain_loss(disc, fs, ft, lambd=r)
                    loss = loss + w_dann * dl
                    domain_acc_list.append(da)

            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
            scaler.step(opt); scaler.update()
            loss_list.append(float(loss.detach().item()))

        metrics = evaluate(adapter, head, seed, ALIGN_NUM_BATCHES_DURING_TRAIN)
        row = dict(method=method, seed=seed, epoch=epoch, loss=float(np.mean(loss_list)), domain_acc=float(np.mean(domain_acc_list)) if domain_acc_list else np.nan)
        row.update(metrics)
        logs.append(row)

        # Checkpoint selection is performed only on the internal NCT source-validation split.
        # CRC-VAL-HE-7K is kept external and is not used for model selection.
        if metrics["NCTval_macroF1"] > best_f1:
            best_f1 = metrics["NCTval_macroF1"]
            best_epoch = epoch
        
            save_ckpt(
                best_path,
                adapter,
                head,
                disc,
                dict(
                    method=method,
                    seed=seed,
                    epoch=epoch,
                    best_NCTval_macroF1=best_f1,
                    metrics=metrics,
                    cfg=cfg
                )
            )

        print(
            f"[{method} seed={seed} epoch={epoch}] "
            f"NCTval_F1={metrics['NCTval_macroF1']:.4f} "
            f"7K_F1={metrics['7K_macroF1']:.4f} "
            f"MKMMD={metrics['TCGA_MKMMD']:.4f} "
            f"Proto={metrics['TCGA_ProtoDist']:.4f} "
            f"PAD={metrics['TCGA_PAD']:.4f} "
            f"NCEnt={metrics['TCGA_NCEntropy']:.4f} "
            f"Maj={metrics['TCGA_Majority']:.4f}"
        )
    load_ckpt(best_path, adapter, head, disc)
    final = evaluate(adapter, head, seed, ALIGN_NUM_BATCHES_FINAL)
    final.update(dict(
    method=method,
    seed=seed,
    best_epoch=best_epoch,
    best_NCTval_macroF1=best_f1,
    method_time_min=(time.time() - t0) / 60.0))
    save_crc_classwise_and_cm(adapter, head, method, seed, CLASS_NAMES)
    for k, v in cfg.items():
        final[f"lambda_{k}"] = v if k != "proto_start" else np.nan
    final["proto_start"] = cfg.get("proto_start", np.nan)

    log_df = pd.DataFrame(logs)
    save_df(log_df, os.path.join(LOG_DIR, f"epoch_logs_{method}_seed{seed}.csv"), os.path.join(LOG_DIR, f"epoch_logs_{method}_seed{seed}.xlsx"))
    save_df(pd.DataFrame([final]), os.path.join(TABLE_DIR, f"final_result_{method}_seed{seed}.csv"), os.path.join(TABLE_DIR, f"final_result_{method}_seed{seed}.xlsx"))
    print(
    "FINAL",
    method,
    seed,
    {
        k: round(final[k], 4)
        for k in [
            "NCTval_macroF1",
            "7K_macroF1",
            "TCGA_MKMMD",
            "TCGA_ProtoDist",
            "TCGA_PAD",
            "TCGA_NCEntropy",
            "TCGA_Majority"
        ]
    })    
    return final, log_df


# %%
# Notebook cell 6
# ============================================================
# 6) Run experiments
# ============================================================
def run_experiments():
    all_results, all_logs = [], []
    for seed in SEEDS:
        for method in METHODS_TO_RUN:
            final, logs = train_one(method, seed)
            all_results.append(final)
            all_logs.append(logs)
            save_df(pd.DataFrame(all_results), os.path.join(TABLE_DIR, "seed_results_all.csv"), os.path.join(TABLE_DIR, "seed_results_all.xlsx"))
            save_df(pd.concat(all_logs, ignore_index=True), os.path.join(LOG_DIR, "epoch_logs_all.csv"), os.path.join(LOG_DIR, "epoch_logs_all.xlsx"))
    return pd.DataFrame(all_results), pd.concat(all_logs, ignore_index=True)

seed_results_df, epoch_logs_df = run_experiments()
seed_results_df


# %%
# Notebook cell 7
# ============================================================
# 7) Summary tables, rankings, paired tests
# ============================================================
METRICS = [
    "7K_macroF1", "7K_balAcc", "7K_acc",
    "TCGA_CORAL", "TCGA_MKMMD", "TCGA_ProtoDist", "TCGA_PAD", "TCGA_NCEntropy", "TCGA_Majority",
]
DIRECTION = {
    "7K_macroF1": "max", "7K_balAcc": "max", "7K_acc": "max", "TCGA_Majority": "max",
    "TCGA_CORAL": "min", "TCGA_MKMMD": "min", "TCGA_ProtoDist": "min", "TCGA_PAD": "min", "TCGA_NCEntropy": "min",
}

def mean_std_table(df, methods=None):
    d = df.copy()
    if methods is not None:
        d = d[d["method"].isin(methods)]
    rows = []
    for method, g in d.groupby("method"):
        row = {"method": method, "n_seeds": g["seed"].nunique()}
        for m in METRICS:
            row[m] = f"{g[m].mean():.4f} ± {g[m].std(ddof=1):.4f}" if len(g) > 1 else f"{g[m].mean():.4f}"
            row[m + "_mean"] = g[m].mean()
            row[m + "_std"] = g[m].std(ddof=1) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)

def rank_table(df):
    s = df.groupby("method")[METRICS].mean().reset_index()
    for m in METRICS:
        s[m + "_rank"] = s[m].rank(ascending=(DIRECTION[m] == "min"), method="min")
    rank_cols = [m + "_rank" for m in METRICS]
    s["mean_rank"] = s[rank_cols].mean(axis=1)
    return s.sort_values("mean_rank")

summary = mean_std_table(seed_results_df)
ranking = rank_table(seed_results_df)
save_df(summary, os.path.join(TABLE_DIR, "summary_mean_std.csv"), os.path.join(TABLE_DIR, "summary_mean_std.xlsx"))
save_df(ranking, os.path.join(TABLE_DIR, "summary_ranking.csv"), os.path.join(TABLE_DIR, "summary_ranking.xlsx"))

ablation_methods = [m for m in ["B0_CE", "B1_CE_CORAL", "B2_CE_CORAL_MKMMD", "B3_STAT_PROTO", "TriAlign_UDA"] if m in seed_results_df.method.unique()]
baseline_methods = [m for m in ["SourceOnly", "DeepCORAL", "DAN", "DANN", "TriAlign_UDA"] if m in seed_results_df.method.unique()]

ablation_table = mean_std_table(seed_results_df, ablation_methods)
baseline_table = mean_std_table(seed_results_df, baseline_methods)
save_df(ablation_table, os.path.join(TABLE_DIR, "ablation_table_mean_std.csv"), os.path.join(TABLE_DIR, "ablation_table_mean_std.xlsx"))
save_df(baseline_table, os.path.join(TABLE_DIR, "baseline_table_mean_std.csv"), os.path.join(TABLE_DIR, "baseline_table_mean_std.xlsx"))

# Paired tests are meaningful only when all compared methods have the same seeds.
def paired_tests(df, target="TriAlign_UDA"):
    try:
        from scipy.stats import ttest_rel, wilcoxon
    except Exception:
        return pd.DataFrame()
    rows = []
    methods = [m for m in df.method.unique() if m != target]
    for other in methods:
        for metric in METRICS:
            a = df[df.method == target].set_index("seed")[metric]
            b = df[df.method == other].set_index("seed")[metric]
            common = sorted(set(a.index).intersection(b.index))
            if len(common) < 2:
                continue
            av, bv = a.loc[common].values, b.loc[common].values
            try:
                t_p = float(ttest_rel(av, bv).pvalue)
            except Exception:
                t_p = np.nan
            try:
                w_p = float(wilcoxon(av, bv).pvalue) if len(common) >= 3 else np.nan
            except Exception:
                w_p = np.nan
            rows.append(dict(target=target, other=other, metric=metric, n=len(common), target_mean=av.mean(), other_mean=bv.mean(), diff_mean=(av-bv).mean(), paired_t_p=t_p, wilcoxon_p=w_p))
    return pd.DataFrame(rows)

pt = paired_tests(seed_results_df)
if len(pt):
    save_df(pt, os.path.join(TABLE_DIR, "paired_tests_vs_TriAlign_UDA.csv"), os.path.join(TABLE_DIR, "paired_tests_vs_TriAlign_UDA.xlsx"))

print("Saved tables to:", TABLE_DIR)
display(summary)
display(ranking)


# %%
# Notebook cell 8
# ============================================================
# 8) Convergence diagnostics from existing epoch logs
# ============================================================
def convergence_diagnostics(epoch_df, result_df):
    conv_dir = ensure_dir(os.path.join(TABLE_DIR, "convergence_diagnostics"))

    best_epoch_counts = result_df.groupby("best_epoch").size().reset_index(name="n_selected")
    save_df(best_epoch_counts, os.path.join(conv_dir, "best_epoch_distribution.csv"), os.path.join(conv_dir, "best_epoch_distribution.xlsx"))

    e3 = epoch_df[epoch_df.epoch == 3][["method", "seed", "7K_macroF1", "TCGA_MKMMD", "TCGA_ProtoDist", "TCGA_NCEntropy", "TCGA_Majority"]]
    e5 = epoch_df[epoch_df.epoch == 5][["method", "seed", "7K_macroF1", "TCGA_MKMMD", "TCGA_ProtoDist", "TCGA_NCEntropy", "TCGA_Majority"]]
    e3 = e3.rename(columns={c: c + "_e3" for c in e3.columns if c not in ["method", "seed"]})
    e5 = e5.rename(columns={c: c + "_e5" for c in e5.columns if c not in ["method", "seed"]})
    d = e3.merge(e5, on=["method", "seed"], how="inner")
    for m in ["7K_macroF1", "TCGA_MKMMD", "TCGA_ProtoDist", "TCGA_NCEntropy", "TCGA_Majority"]:
        d[m + "_abs_delta_e5_vs_e3"] = (d[m + "_e5"] - d[m + "_e3"]).abs()
    save_df(d, os.path.join(conv_dir, "epoch3_vs_epoch5_delta_per_seed.csv"), os.path.join(conv_dir, "epoch3_vs_epoch5_delta_per_seed.xlsx"))

    curve = epoch_df.groupby(["method", "epoch"])[METRICS].agg(["mean", "std"]).reset_index()
    # Flatten columns
    curve.columns = ["_".join([str(x) for x in c if str(x)]) for c in curve.columns.values]
    save_df(curve, os.path.join(conv_dir, "learning_curve_summary.csv"), os.path.join(conv_dir, "learning_curve_summary.xlsx"))
    return best_epoch_counts, d, curve

best_epoch_counts, epoch_delta, curve_summary = convergence_diagnostics(epoch_logs_df, seed_results_df)
display(best_epoch_counts)
display(epoch_delta.head())


# %%
# Notebook cell 9
# ============================================================
# 9) Aggregate class-wise CRC-VAL diagnostics
# ============================================================
import glob

def aggregate_crc_classwise_outputs():
    classwise_dir = os.path.join(TABLE_DIR, "classwise")
    cm_dir = os.path.join(TABLE_DIR, "confusion_matrices")

    classwise_files = sorted(glob.glob(os.path.join(classwise_dir, "classwise_*_seed*.csv")))
    cm_long_files = sorted(glob.glob(os.path.join(cm_dir, "cm_long_*_seed*.csv")))

    if len(classwise_files) == 0:
        print("No class-wise files found.")
        return None, None, None, None

    classwise_all = pd.concat(
        [pd.read_csv(f) for f in classwise_files],
        ignore_index=True
    )

    classwise_summary = (
        classwise_all
        .groupby(["method", "class_id", "class_name"])[["precision", "recall", "f1", "support"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    classwise_summary.columns = [
        "_".join([str(x) for x in c if str(x)])
        for c in classwise_summary.columns.values
    ]

    save_df(
        classwise_all,
        os.path.join(TABLE_DIR, "crc7k_classwise_all_seeds.csv"),
        os.path.join(TABLE_DIR, "crc7k_classwise_all_seeds.xlsx")
    )

    save_df(
        classwise_summary,
        os.path.join(TABLE_DIR, "crc7k_classwise_summary_mean_std.csv"),
        os.path.join(TABLE_DIR, "crc7k_classwise_summary_mean_std.xlsx")
    )

    cm_all = None
    cm_summary = None

    if len(cm_long_files) > 0:
        cm_all = pd.concat(
            [pd.read_csv(f) for f in cm_long_files],
            ignore_index=True
        )

        cm_summary = (
            cm_all
            .groupby(["method", "true_class_id", "true_class", "pred_class_id", "pred_class"])[["count", "row_normalized"]]
            .agg(["mean", "std"])
            .reset_index()
        )

        cm_summary.columns = [
            "_".join([str(x) for x in c if str(x)])
            for c in cm_summary.columns.values
        ]

        save_df(
            cm_all,
            os.path.join(TABLE_DIR, "crc7k_confusion_matrix_long_all_seeds.csv"),
            os.path.join(TABLE_DIR, "crc7k_confusion_matrix_long_all_seeds.xlsx")
        )

        save_df(
            cm_summary,
            os.path.join(TABLE_DIR, "crc7k_confusion_matrix_long_summary_mean_std.csv"),
            os.path.join(TABLE_DIR, "crc7k_confusion_matrix_long_summary_mean_std.xlsx")
        )

    print("Class-wise and confusion-matrix summaries saved to:", TABLE_DIR)

    return classwise_all, classwise_summary, cm_all, cm_summary


classwise_all, classwise_summary, cm_all, cm_summary = aggregate_crc_classwise_outputs()

if classwise_summary is not None:
    display(classwise_summary.head(20))
else:
    print("Class-wise summary could not be displayed because no class-wise files were found.")


# %%
# Notebook cell 10
# ============================================================
# 10) Computational cost summary
# No retraining is performed in this cell.
# ============================================================

def count_params_module(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def compute_trainable_param_counts():
    adapter = BottleneckAdapter(FEATURE_DIM, ADAPTER_REDUCTION)
    head = LinearHead(FEATURE_DIM, NUM_CLASSES)
    disc = DomainDiscriminator(FEATURE_DIM)

    adapter_params = count_params_module(adapter)
    head_params = count_params_module(head)
    disc_params = count_params_module(disc)

    no_disc_total = adapter_params + head_params
    with_disc_total = adapter_params + head_params + disc_params

    rows = []

    for method in sorted(seed_results_df["method"].unique()):
        uses_disc = METHOD_CONFIGS[method].get("dann", 0) > 0

        rows.append({
            "method": method,
            "feature_backbone": "UNI2-H frozen",
            "image_level_finetuning": "No",
            "trainable_components": (
                "adapter + classifier head + domain discriminator"
                if uses_disc else
                "adapter + classifier head"
            ),
            "adapter_params": adapter_params,
            "head_params": head_params,
            "domain_discriminator_params": disc_params if uses_disc else 0,
            "total_trainable_params": with_disc_total if uses_disc else no_disc_total,
        })

    return pd.DataFrame(rows)

param_df = compute_trainable_param_counts()

time_df = (
    seed_results_df
    .groupby("method")["method_time_min"]
    .agg(["mean", "std"])
    .reset_index()
    .rename(columns={
        "mean": "training_time_min_mean",
        "std": "training_time_min_std"
    })
)

computation_table = param_df.merge(time_df, on="method", how="left")

save_df(
    computation_table,
    os.path.join(TABLE_DIR, "computational_cost_summary.csv"),
    os.path.join(TABLE_DIR, "computational_cost_summary.xlsx")
)

display(computation_table)
