#!/usr/bin/env python3
"""
Run PanEcho inference on EchoNet-Dynamic videos.

Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU — auto-detected.
Results are written incrementally so a run can be safely interrupted and resumed.

Usage:
    python run_panecho.py --data_dir /path/to/EchoNet-Dynamic
    python run_panecho.py --data_dir /path/to/EchoNet-Dynamic --split test
    python run_panecho.py --data_dir /path/to/EchoNet-Dynamic --max_videos 100 --device cpu
"""

import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ImageNet stats used for normalisation (matches PanEcho training)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None, None]
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None, None]


# ── Device selection ─────────────────────────────────────────────────────────

def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Video loading ────────────────────────────────────────────────────────────

def load_video(path: Path, clip_len: int, size: int = 224) -> torch.Tensor | None:
    """
    Read an AVI, uniformly sample `clip_len` frames, resize to `size`x`size`,
    apply ImageNet normalisation, and return shape (1, 3, T, H, W).
    Returns None if the video cannot be read.
    """
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

    # Uniform temporal sampling
    if len(frames) >= clip_len:
        idx = np.linspace(0, len(frames) - 1, clip_len, dtype=int)
        frames = [frames[i] for i in idx]
    else:
        while len(frames) < clip_len:           # pad by repeating last frame
            frames.append(frames[-1])

    video = np.stack(frames).transpose(3, 0, 1, 2).astype(np.float32) / 255.0  # (C,T,H,W)
    video = (video - _MEAN) / _STD
    return torch.from_numpy(video).unsqueeze(0)   # (1,C,T,H,W)


# ── Prediction flattening ────────────────────────────────────────────────────

def flatten_preds(preds: dict) -> dict:
    """Convert the model output dict into a flat {column: scalar} dict."""
    flat = {}
    for task, val in preds.items():
        if isinstance(val, torch.Tensor):
            v = val.detach().cpu().float()
            if v.numel() == 1:
                flat[task] = v.item()
            else:
                for i, scalar in enumerate(v.flatten().tolist()):
                    flat[f"{task}_cls{i}"] = scalar
        else:
            flat[task] = val
    return flat


# ── Inference loop ───────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    data_dir   = Path(args.data_dir)
    video_dir  = data_dir / "Videos"
    filelist_p = data_dir / "FileList.csv"
    output_p   = Path(args.output)

    if not video_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {video_dir}")
    if not filelist_p.exists():
        raise FileNotFoundError(f"FileList.csv not found: {filelist_p}")

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device(args.device) if args.device else best_device()
    print(f"Device: {device}")

    # ── Load model ──────────────────────────────────────────────────────────
    print("Loading PanEcho model (downloads weights on first run) …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = torch.hub.load(
            "CarDS-Yale/PanEcho",
            "PanEcho",
            force_reload=False,
            clip_len=args.clip_len,
        )
    model.eval()

    try:
        model = model.to(device)
    except Exception as e:
        print(f"Cannot move model to {device} ({e}). Falling back to CPU.")
        device = torch.device("cpu")
        model = model.to(device)

    # ── File list ────────────────────────────────────────────────────────────
    filelist = pd.read_csv(filelist_p)
    if args.split:
        filelist = filelist[filelist["Split"].str.lower() == args.split.lower()]
    if args.max_videos:
        filelist = filelist.head(args.max_videos)

    # ── Resume support ───────────────────────────────────────────────────────
    already_done: set[str] = set()
    if output_p.exists() and not args.overwrite:
        done_df = pd.read_csv(output_p)
        already_done = set(done_df["FileName"].astype(str))
        print(f"Resuming — {len(already_done)} videos already processed.")
        filelist = filelist[~filelist["FileName"].astype(str).isin(already_done)]

    print(f"Videos to process: {len(filelist)}")

    gt_cols = [c for c in ["EF", "ESV", "EDV", "Split"] if c in filelist.columns]
    failed: list[str] = []
    write_header = not output_p.exists() or args.overwrite

    with open(output_p, "w" if args.overwrite else "a", buffering=1) as fout:
        for idx, row in tqdm(filelist.iterrows(), total=len(filelist), desc="PanEcho"):
            fname = str(row["FileName"])
            avi   = fname if fname.endswith(".avi") else fname + ".avi"
            vpath = video_dir / avi

            if not vpath.exists():
                failed.append(fname)
                continue

            tensor = load_video(vpath, clip_len=args.clip_len)
            if tensor is None:
                failed.append(fname)
                continue

            try:
                with torch.no_grad():
                    preds = model(tensor.to(device))
            except RuntimeError:
                # MPS may not support every op — retry on CPU
                if device.type == "mps":
                    with torch.no_grad():
                        preds = model.cpu()(tensor)
                    model = model.to(device)          # move back for next video
                else:
                    failed.append(fname)
                    continue

            result = {"FileName": fname}
            for col in gt_cols:
                result[f"GT_{col}"] = row[col]
            result.update(flatten_preds(preds))

            row_df = pd.DataFrame([result])
            row_df.to_csv(fout, index=False, header=write_header)
            write_header = False   # only write header once

    print(f"\nResults saved to: {output_p}")
    print(f"Processed: {len(filelist) - len(failed)} | Failed/skipped: {len(failed)}")

    if failed:
        fail_path = output_p.with_stem(output_p.stem + "_failed").with_suffix(".txt")
        fail_path.write_text("\n".join(failed))
        print(f"Failed video list: {fail_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run PanEcho on EchoNet-Dynamic videos (Mac-compatible).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",    required=True,
                   help="Root directory of EchoNet-Dynamic (contains Videos/ and FileList.csv)")
    p.add_argument("--output",      default="panecho_results.csv",
                   help="Output CSV path")
    p.add_argument("--clip_len",    type=int, default=16,
                   help="Number of frames sampled per video")
    p.add_argument("--split",       default=None,
                   choices=["train", "val", "test"],
                   help="Only process one dataset split")
    p.add_argument("--max_videos",  type=int, default=None,
                   help="Stop after N videos (useful for testing)")
    p.add_argument("--device",      default=None,
                   choices=["cuda", "mps", "cpu"],
                   help="Force a specific device (default: auto-detect)")
    p.add_argument("--overwrite",   action="store_true",
                   help="Overwrite existing output instead of resuming")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
