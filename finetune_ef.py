#!/usr/bin/env python3
"""
Fine-tune PanEcho for ejection fraction (EF) regression on EchoNet-Dynamic.

Strategy (two-phase):
  Phase 1 — freeze the backbone, train only EF-related parameters (fast convergence)
  Phase 2 — unfreeze all, fine-tune end-to-end with a lower learning rate

Usage:
    python finetune_ef.py --data_dir ~/data/EchoNet-Dynamic
    python finetune_ef.py --data_dir ~/data/EchoNet-Dynamic --epochs 20 --device mps
"""

import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None, None]
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None, None]


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_video(path: Path, clip_len: int, size: int = 224) -> torch.Tensor | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()
    if not frames:
        return None
    if len(frames) >= clip_len:
        idx = np.linspace(0, len(frames) - 1, clip_len, dtype=int)
        frames = [frames[i] for i in idx]
    else:
        while len(frames) < clip_len:
            frames.append(frames[-1])
    video = np.stack(frames).transpose(3, 0, 1, 2).astype(np.float32) / 255.0
    video = (video - _MEAN) / _STD
    return torch.from_numpy(video)  # (C, T, H, W)


class EchoEFDataset(Dataset):
    def __init__(self, filelist: pd.DataFrame, video_dir: Path, clip_len: int):
        self.rows = filelist.dropna(subset=["EF"]).reset_index(drop=True)
        self.video_dir = video_dir
        self.clip_len = clip_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        fname = str(row["FileName"])
        avi = fname if fname.endswith(".avi") else fname + ".avi"
        tensor = load_video(self.video_dir / avi, self.clip_len)
        if tensor is None:
            tensor = torch.zeros(3, self.clip_len, 224, 224)
        ef = torch.tensor(float(row["EF"]), dtype=torch.float32)
        return tensor, ef


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    mae  = (preds - targets).abs().mean().item()
    rmse = ((preds - targets) ** 2).mean().sqrt().item()
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).item()
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ── Training ──────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    all_preds, all_targets = [], []
    total_loss = 0.0
    loss_fn = nn.HuberLoss()

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for videos, efs in tqdm(loader, desc="train" if train else "val", leave=False):
            videos = videos.to(device)
            efs    = efs.to(device)

            preds_dict = model(videos)
            ef_pred = preds_dict["EF"].squeeze()

            loss = loss_fn(ef_pred, efs)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(efs)
            all_preds.append(ef_pred.detach().cpu())
            all_targets.append(efs.cpu())

    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(all_targets)
    return metrics


def train(args):
    data_dir  = Path(args.data_dir)
    video_dir = data_dir / "Videos"
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {video_dir}")

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    # ── Load model ─────────────────────────────────────────────────────────
    print("Loading PanEcho …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = torch.hub.load(
            "CarDS-Yale/PanEcho", "PanEcho",
            force_reload=False, clip_len=args.clip_len,
        )
    model = model.to(device)

    # ── Data ───────────────────────────────────────────────────────────────
    filelist = pd.read_csv(data_dir / "FileList.csv")
    train_df = filelist[filelist["Split"].str.upper() == "TRAIN"]
    val_df   = filelist[filelist["Split"].str.upper() == "VAL"]

    train_ds = EchoEFDataset(train_df, video_dir, args.clip_len)
    val_ds   = EchoEFDataset(val_df,   video_dir, args.clip_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Phase 1: freeze backbone, train EF head only ───────────────────────
    print("\n── Phase 1: head-only training ──")
    for name, p in model.named_parameters():
        p.requires_grad = "ef" in name.lower() or "head" in name.lower()

    # If no EF-specific params found, unfreeze the last block
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        print("No EF-specific params found — unfreezing last block instead.")
        params = list(model.parameters())
        for p in params[-20:]:
            p.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_phase1, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.phase1_epochs)

    best_val_mae = float("inf")
    log_rows = []

    for epoch in range(1, args.phase1_epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, train=True)
        vl = run_epoch(model, val_loader,   optimizer, device, train=False)
        scheduler.step()

        log_rows.append({"phase": 1, "epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
                         **{f"val_{k}": v for k, v in vl.items()}})
        print(f"[P1 {epoch:02d}] train MAE={tr['MAE']:.2f}  val MAE={vl['MAE']:.2f}  R²={vl['R2']:.3f}")

        if vl["MAE"] < best_val_mae:
            best_val_mae = vl["MAE"]
            torch.save(model.state_dict(), out_dir / "best_ef_model.pt")

    # ── Phase 2: unfreeze all, fine-tune end-to-end ────────────────────────
    if args.phase2_epochs > 0:
        print("\n── Phase 2: full fine-tuning ──")
        for p in model.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_phase2, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.phase2_epochs)

        for epoch in range(1, args.phase2_epochs + 1):
            tr = run_epoch(model, train_loader, optimizer, device, train=True)
            vl = run_epoch(model, val_loader,   optimizer, device, train=False)
            scheduler.step()

            log_rows.append({"phase": 2, "epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
                             **{f"val_{k}": v for k, v in vl.items()}})
            print(f"[P2 {epoch:02d}] train MAE={tr['MAE']:.2f}  val MAE={vl['MAE']:.2f}  R²={vl['R2']:.3f}")

            if vl["MAE"] < best_val_mae:
                best_val_mae = vl["MAE"]
                torch.save(model.state_dict(), out_dir / "best_ef_model.pt")

    pd.DataFrame(log_rows).to_csv(out_dir / "training_log.csv", index=False)
    print(f"\nBest val MAE: {best_val_mae:.2f}")
    print(f"Model saved to: {out_dir / 'best_ef_model.pt'}")
    print(f"Training log:  {out_dir / 'training_log.csv'}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune PanEcho for EF regression.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",       required=True)
    p.add_argument("--output_dir",     default="ef_finetune")
    p.add_argument("--clip_len",       type=int,   default=16)
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--workers",        type=int,   default=2)
    p.add_argument("--phase1_epochs",  type=int,   default=5,
                   help="Epochs with backbone frozen")
    p.add_argument("--phase2_epochs",  type=int,   default=10,
                   help="Epochs with full model unfrozen (0 to skip)")
    p.add_argument("--lr_phase1",      type=float, default=1e-3)
    p.add_argument("--lr_phase2",      type=float, default=1e-4)
    p.add_argument("--device",         default=None, choices=["cuda", "mps", "cpu"])
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
