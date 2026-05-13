"""
Train DiT visual heads standalone (same module as ``HaloVLM.visual_predictor``).

For end-to-end VLA training (language + actions + visual DiT), prefer
``scripts/train.py`` — it calls ``compute_visual_prediction_loss`` automatically.

This script only optimises ``VisualDiTPredictor`` on consecutive frames.

Usage:
    PYTHONPATH=src/Halo_VLA python scripts/train_visual_prediction.py --device cuda --steps 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "Halo_VLA"))

from config import HaloVLMConfig
from dataloader.eo_dataset import build_eo_dataloader
from models.dit_frame_prediction import VisualDiTPredictor


def parse_args():
    p = argparse.ArgumentParser(description="Train DiT frame / depth / flow heads")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steps", type=int, default=50, help="Max optimisation steps (early stop if dataset ends)")
    p.add_argument("--subset", default="interleave-temporal")
    p.add_argument("--max_samples", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    cfg = HaloVLMConfig()
    model = VisualDiTPredictor(cfg).to(device)
    opt = AdamW(model.parameters(), lr=args.lr)

    loader = build_eo_dataloader(
        subset=args.subset,
        split="train",
        batch_size=args.batch_size,
        num_workers=0,
        img_size=cfg.img_size,
        max_seq_len=128,
        action_dim=cfg.action_dim,
        state_dim=cfg.state_dim,
        shuffle=True,
        max_samples=args.max_samples,
    )

    step = 0
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            images = batch["images"].to(device)
            if images.size(1) < 2:
                continue
            past = images[:, :-1]
            future = images[:, -1]
            losses = model.compute_losses(past, future)
            loss = losses["loss_visual_total"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 10 == 0:
                print(
                    f"step {step}  total={loss.item():.4f}  "
                    f"rgb_cfm={losses.get('loss_rgb_cfm', torch.tensor(0)).item():.4f}  "
                    f"depth_cfm={losses.get('loss_depth_cfm', torch.tensor(0)).item():.4f}  "
                    f"flow_cfm={losses.get('loss_flow_cfm', torch.tensor(0)).item():.4f}"
                )
            step += 1

    print("Done.")


if __name__ == "__main__":
    main()
