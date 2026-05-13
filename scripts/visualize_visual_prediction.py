"""
Visualise HaloVLM DiT outputs: predicted future RGB, depth, and optical flow.

Builds a tiled PNG per sample: past frames, ground-truth future, predicted RGB,
depth (colour-mapped), flow (HSV colour wheel), and past frame warped by
predicted flow.

Usage:
    PYTHONPATH=src/Halo_VLA python scripts/visualize_visual_prediction.py \\
        --checkpoint checkpoints/halo_vla_epoch1.pt \\
        --output_dir vis_dit --num_samples 8

    # Without checkpoint (random DiT — sanity check layout only):
    python scripts/visualize_visual_prediction.py --output_dir vis_dit --no_checkpoint

Requires: opencv-python, numpy, torch
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "Halo_VLA"))

from config import HaloVLMConfig
from dataloader.eo_dataset import EODatasetConfig, build_eo_dataloader
from models.dit_frame_prediction import backward_warp
from models.halo_vla import HaloVLM

from loguru import logger


def _import_cv2():
    import cv2
    return cv2


def load_model(
    ckpt_path: str | None,
    device: torch.device,
    enable_visual_dit: bool,
):
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_cfg = ckpt.get("config")
        config = HaloVLMConfig()
        if raw_cfg is not None:
            for f in fields(HaloVLMConfig):
                if hasattr(raw_cfg, f.name):
                    setattr(config, f.name, getattr(raw_cfg, f.name))
        config.enable_visual_dit = enable_visual_dit
        model = HaloVLM(config=config).to(device)
        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        if missing:
            logger.warning(
                "Checkpoint missing {} keys (inits random): e.g. {}",
                len(missing),
                missing[:3],
            )
        if unexpected:
            logger.warning("Checkpoint had {} unexpected keys", len(unexpected))
        logger.info("Loaded weights from {}", ckpt_path)
    else:
        config = HaloVLMConfig()
        config.enable_visual_dit = enable_visual_dit
        model = HaloVLM(config=config).to(device)
        logger.warning("No checkpoint — random weights (layout demo only)")

    model.eval()
    return model, config


def unnormalise_image(tensor: torch.Tensor, mean: Tuple, std: Tuple) -> np.ndarray:
    """[3,H,W] normalised tensor → uint8 BGR."""
    cv2 = _import_cv2()
    img = tensor.clone().cpu().float()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def pred_rgb_to_bgr(pred: torch.Tensor) -> np.ndarray:
    """DiT RGB output — per-image min–max to uint8 BGR (not ImageNet-normalised)."""
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


def depth_to_bgr(depth: torch.Tensor) -> np.ndarray:
    """[1,H,W] → colour BGR uint8."""
    cv2 = _import_cv2()
    d = depth.detach().cpu().float()
    if d.dim() == 4:
        d = d[0, 0]
    else:
        d = d[0]
    d = d.numpy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    d_u8 = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)


def flow_to_bgr(flow: torch.Tensor, max_mag: float | None = None) -> np.ndarray:
    """
    [2,H,W] motion field → BGR (Middlebury-style hue = direction, value = speed).
    """
    cv2 = _import_cv2()
    f = flow.detach().cpu().float()
    if f.dim() == 4:
        f = f[0]
    u = f[0].numpy()
    v = f[1].numpy()
    mag = np.sqrt(u * u + v * v)
    if max_mag is None:
        max_mag = float(np.percentile(mag, 95) + 1e-6)
    ang = np.arctan2(v, u)
    h = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    s = np.full_like(h, 255, dtype=np.uint8)
    vm = np.clip(mag / max_mag, 0, 1)
    vv = (vm * 255).astype(np.uint8)
    hsv = np.stack([h, s, vv], axis=-1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    cv2 = _import_cv2()
    out = img.copy()
    cv2.putText(
        out,
        text,
        (6, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        text,
        (6, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return out


def resize_h(img: np.ndarray, height: int) -> np.ndarray:
    cv2 = _import_cv2()
    if img.shape[0] == height:
        return img
    w = int(round(img.shape[1] * (height / img.shape[0])))
    return cv2.resize(img, (w, height), interpolation=cv2.INTER_AREA)


def hstack_images(imgs: List[np.ndarray], height: int) -> np.ndarray:
    resized = [resize_h(im, height) for im in imgs]
    return np.hstack(resized)


def build_panel(
    past_bgr: List[np.ndarray],
    gt_future_bgr: np.ndarray,
    pred_rgb_bgr: np.ndarray,
    depth_bgr: np.ndarray,
    flow_bgr: np.ndarray,
    warped_bgr: np.ndarray,
    row_height: int = 240,
) -> np.ndarray:
    row1 = hstack_images(
        [add_label(x, f"past {i}") for i, x in enumerate(past_bgr)],
        row_height,
    )
    row2_list = [
        add_label(gt_future_bgr, "GT future"),
        add_label(pred_rgb_bgr, "pred RGB"),
        add_label(depth_bgr, "pred depth"),
        add_label(flow_bgr, "pred flow"),
        add_label(warped_bgr, "warp(prev)"),
    ]
    row2 = hstack_images(row2_list, row_height)
    w = max(row1.shape[1], row2.shape[1])
    pad1 = np.zeros((row1.shape[0], w - row1.shape[1], 3), dtype=np.uint8)
    pad2 = np.zeros((row2.shape[0], w - row2.shape[1], 3), dtype=np.uint8)
    row1 = np.hstack([row1, pad1])
    row2 = np.hstack([row2, pad2])
    gap = np.ones((8, w, 3), dtype=np.uint8) * 40
    return np.vstack([row1, gap, row2])


def parse_args():
    p = argparse.ArgumentParser(description="Visualise DiT RGB / depth / flow")
    p.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint")
    p.add_argument("--no_checkpoint", action="store_true", help="Random weights")
    p.add_argument("--output_dir", type=Path, default=Path("vis_visual_dit"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--subset", default="interleave-temporal")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=256)
    p.add_argument("--disable_visual_dit", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cv2 = _import_cv2()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = None if args.no_checkpoint else args.checkpoint
    if not args.no_checkpoint and not ckpt:
        logger.error("Pass --checkpoint PATH or --no_checkpoint")
        sys.exit(1)

    enable_v = not args.disable_visual_dit
    model, config = load_model(ckpt, device, enable_visual_dit=enable_v)

    if not enable_v or model.visual_predictor is None:
        logger.error("Visual DiT disabled — nothing to render")
        sys.exit(1)

    ds_mean = EODatasetConfig.img_mean
    ds_std = EODatasetConfig.img_std

    loader = build_eo_dataloader(
        subset=args.subset,
        split="train",
        batch_size=args.batch_size,
        num_workers=0,
        img_size=config.img_size,
        max_seq_len=128,
        action_dim=config.action_dim,
        state_dim=config.state_dim,
        shuffle=True,
        max_samples=args.max_samples,
    )

    written = 0
    for batch in loader:
        if written >= args.num_samples:
            break
        images = batch["images"].to(device)
        mask = batch.get("image_mask")
        B = images.size(0)
        for b in range(B):
            if written >= args.num_samples:
                break
            if mask is not None:
                n_valid = int(mask[b].sum().item())
                if n_valid < 2:
                    continue
                past = images[b : b + 1, : n_valid - 1]
                fut = images[b, n_valid - 1]
                n_past = n_valid - 1
            else:
                if images.size(1) < 2:
                    continue
                past = images[b : b + 1, :-1]
                fut = images[b, -1]
                n_past = images.size(1) - 1

            preds = model.predict_visual_future(past)
            if not preds:
                continue

            past_bgr = [
                unnormalise_image(past[0, i].cpu(), ds_mean, ds_std) for i in range(n_past)
            ]
            gt_bgr = unnormalise_image(fut.cpu(), ds_mean, ds_std)

            pr_rgb = pred_rgb_to_bgr(preds["rgb"])
            pr_depth = depth_to_bgr(preds["depth"])
            pr_flow = flow_to_bgr(preds["flow"])

            last = past
            warp_t = backward_warp(last[:, -1], preds["flow"])
            warped_bgr = pred_rgb_to_bgr(warp_t[0])

            panel = build_panel(
                past_bgr,
                gt_bgr,
                pr_rgb,
                pr_depth,
                pr_flow,
                warped_bgr,
            )
            out_path = args.output_dir / f"sample_{written:03d}.png"
            cv2.imwrite(str(out_path), panel)
            logger.info("Wrote {}", out_path)
            written += 1

    logger.info("Done, {} panel(s) in {}", written, args.output_dir)


if __name__ == "__main__":
    main()
