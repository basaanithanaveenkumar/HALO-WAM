"""
Render MoMa dataloader samples as an MP4 at **1 FPS**: input sequence vs GT vs DiT prediction.

For each sample, the video shows one composite frame per second:
  - **Input**: current timestep in the sampled frame window (context).
  - **GT**: same pixel observation (ground-truth video frame); on the **last** timestep
    this is the training target for the visual DiT head.
  - **Pred**: DiT RGB prediction from ``predict_visual_future`` (needs a checkpoint
    for meaningful output; otherwise a placeholder).

Usage:
    PYTHONPATH=src/Halo_VLA python scripts/visualize_airoa_moma.py \\
        --moma_data_root /path/to/airoa-moma \\
        --checkpoint checkpoints/halo_vla_epoch1.pt \\
        --output_dir vis_moma --num_samples 3

    tensorboard --logdir runs/   # optional, separate from this script

Requires: opencv-python, torch, transformers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "Halo_VLA"))

from config import HaloVLMConfig
from dataloader.airoa_moma_dataset import AiroaMomaConfig, AiroaMomaDataset
from models.halo_vla import HaloVLM

from loguru import logger


def _import_cv2():
    import cv2
    return cv2


def tensor_to_bgr(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> np.ndarray:
    """Denormalise ImageNet [3,H,W] → uint8 BGR."""
    cv2 = _import_cv2()
    img = tensor.detach().cpu().float()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def pred_rgb_to_bgr(pred: torch.Tensor) -> np.ndarray:
    cv2 = _import_cv2()
    x = pred.detach().cpu().float().clamp(-8, 8)
    if x.dim() == 4:
        x = x[0]
    cmin = x.amin(dim=(1, 2), keepdim=True)
    cmax = x.amax(dim=(1, 2), keepdim=True)
    x = (x - cmin) / (cmax - cmin + 1e-8)
    x = x.permute(1, 2, 0).numpy()
    x = (x * 255).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def resize_h(img: np.ndarray, h: int) -> np.ndarray:
    cv2 = _import_cv2()
    if img.shape[0] == h:
        return img
    w = int(round(img.shape[1] * (h / img.shape[0])))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def label_bar(img: np.ndarray, text: str) -> np.ndarray:
    cv2 = _import_cv2()
    h, w = img.shape[:2]
    bar = np.ones((32, w, 3), dtype=np.uint8) * 45
    cv2.putText(bar, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def hstack_three(inp: np.ndarray, gt: np.ndarray, pr: np.ndarray, height: int) -> np.ndarray:
    a = label_bar(resize_h(inp, height), "input (seq)")
    b = label_bar(resize_h(gt, height), "ground truth")
    c = label_bar(resize_h(pr, height), "predicted")
    w_max = max(a.shape[1], b.shape[1], c.shape[1])
    def pad(x):
        if x.shape[1] < w_max:
            padw = w_max - x.shape[1]
            return np.pad(x, ((0, 0), (0, padw), (0, 0)), constant_values=40)
        return x
    a, b, c = pad(a), pad(b), pad(c)
    return np.hstack([a, b, c])


def load_model_ckpt(path: str | None, device: torch.device):
    from dataclasses import fields

    if not path:
        return None, None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw = ckpt.get("config")
    cfg = HaloVLMConfig()
    if raw is not None:
        for f in fields(HaloVLMConfig):
            if hasattr(raw, f.name):
                setattr(cfg, f.name, getattr(raw, f.name))
    model = HaloVLM(config=cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, cfg


def parse_args():
    p = argparse.ArgumentParser(description="MoMa 1 FPS input/GT/pred video")
    p.add_argument("--moma_data_root", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output_dir", type=Path, default=Path("vis_airoa_moma"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--fps", type=float, default=1.0, help="Output video FPS (default 1)")
    p.add_argument("--row_height", type=int, default=240)
    p.add_argument("--split", default="train")
    p.add_argument("--max_dataset_samples", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cv2 = _import_cv2()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = AiroaMomaConfig(
        data_root=args.moma_data_root,
        max_samples=args.max_dataset_samples,
    )
    ds = AiroaMomaDataset(config=cfg, split=args.split)
    if len(ds) == 0:
        logger.error("No episodes in dataset — check data_root and videos")
        sys.exit(1)

    model, _ = load_model_ckpt(args.checkpoint, device)

    mean, std = cfg.img_mean, cfg.img_std
    n_out = min(args.num_samples, len(ds))
    rng = np.random.default_rng(0)
    indices = rng.choice(len(ds), size=n_out, replace=False)

    for si, idx in enumerate(indices):
        sample = ds[int(idx)]
        images = sample["images"]  # [T,3,H,W]
        T = images.size(0)
        past = images[:-1].unsqueeze(0).to(device) if T >= 2 else images.unsqueeze(0).to(device)
        pred_bgr = None
        if model is not None and T >= 2 and model.visual_predictor is not None:
            with torch.no_grad():
                enc = model.encode_visual_context_from_past_vit(past[0])
                preds = model.predict_visual_future(past, visual_context_emb=enc)
                if preds.get("rgb") is not None:
                    pred_bgr = pred_rgb_to_bgr(preds["rgb"])

        blank_pred = np.full(
            (args.row_height + 32, args.row_height + 32, 3), 40, dtype=np.uint8
        )
        cv2.putText(
            blank_pred,
            "no pred",
            (20, args.row_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )

        frames_out: List[np.ndarray] = []
        for t in range(T):
            inp_bgr = tensor_to_bgr(images[t], mean, std)
            gt_bgr = inp_bgr.copy()
            if t < T - 1:
                pr = blank_pred
            else:
                pr = pred_bgr if pred_bgr is not None else blank_pred
            panel = hstack_three(inp_bgr, gt_bgr, pr, args.row_height)
            frames_out.append(panel)

        out_path = args.output_dir / f"moma_sample_{si:03d}_idx{int(idx)}.mp4"
        h, w = frames_out[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(out_path), fourcc, float(args.fps), (w, h))
        if not vw.isOpened():
            logger.error("VideoWriter failed for {}", out_path)
            continue
        for fr in frames_out:
            vw.write(fr)
        vw.release()
        logger.info("Wrote {} ({} frames @ {} FPS)", out_path, len(frames_out), args.fps)

    logger.info("Done → {}", args.output_dir)


if __name__ == "__main__":
    main()
