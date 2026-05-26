"""
Simple training script for Halo-VLA.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 10 --batch_size 4 --lr 1e-4
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---------------------------------------------------------------------------
# Make sure project root is on sys.path so imports work
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "Halo_VLA"))

from config import HaloVLMConfig
from models.halo_vla import HaloVLM
from dataloader.eo_dataset import build_eo_dataloader
from dataloader.airoa_moma_dataset import build_airoa_moma_dataloader
from utils import log_module_parameters
from scripts.visualize import (
    unnormalise_image,
    generate_text,
    get_font,
    action_comparison_image,
    sample_dit_with_intermediates,
    save_diffusion_gif,
    tensor_to_display_image,
)

from loguru import logger

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False

try:
    import imageio
    _IMAGEIO_AVAILABLE = True
except ImportError:
    imageio = None  # type: ignore[assignment]
    _IMAGEIO_AVAILABLE = False

import numpy as np


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
def compute_language_loss(logits, labels, num_prepended):
    """
    Cross-entropy loss on the text portion of the sequence.

    Args:
        logits:        [B, total_len, vocab_size]
        labels:        [B, seq_len]       (−100 for masked positions)
        num_prepended: int — number of image-patch tokens prepended
    """
    # The logits for text tokens start after the prepended image patches.
    # Shift by 1 for next-token prediction: predict token t+1 from position t.
    text_logits = logits[:, num_prepended:-1, :]  # [B, seq_len-1, V]
    target = labels[:, 1:]                         # [B, seq_len-1]

    # Truncate to the shorter of the two (in case of length mismatch)
    min_len = min(text_logits.size(1), target.size(1))
    text_logits = text_logits[:, :min_len, :]
    target = target[:, :min_len]

    loss = F.cross_entropy(
        text_logits.reshape(-1, text_logits.size(-1)),
        target.reshape(-1),
        ignore_index=-100,
    )
    return loss


def compute_action_loss(action_preds, action_targets, action_mask):
    """
    MSE loss between predicted and ground-truth actions.

    Args:
        action_preds:   [B, n_act, chunk_size, action_dim] or None
        action_targets: [B, max_T, action_dim]
        action_mask:    [B, max_T]  (1 for real, 0 for pad)
    """
    if action_preds is None:
        return torch.tensor(0.0, device=action_targets.device)

    # Flatten chunk dimension: [B, n_act * chunk_size, action_dim]
    B, n_act, chunk, dim = action_preds.shape
    preds_flat = action_preds.view(B, n_act * chunk, dim)

    # Align lengths (targets may be longer/shorter)
    T = min(preds_flat.size(1), action_targets.size(1))
    preds_flat = preds_flat[:, :T, :]
    targets = action_targets[:, :T, :]
    mask = action_mask[:, :T].unsqueeze(-1).float()  # [B, T, 1]

    if mask.sum() == 0:
        return torch.tensor(0.0, device=preds_flat.device)
    # l1 loss
    loss = ((preds_flat - targets).abs() * mask).sum() / mask.sum() / dim
    return loss

def draw_overlays(frame, question, gt_answer, pred_answer, gt_actions, pred_actions, frame_idx):
    if not _CV2_AVAILABLE:
        return frame
    cv2.putText(frame, f"Q: {question[:50]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"GT: {gt_answer[:30]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if pred_answer:
        cv2.putText(frame, f"Pred: {pred_answer[:30]}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return frame


# ---------------------------------------------------------------------------
# Unified training visualization
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_training_visualization(
    model,
    batch,
    tokenizer,
    config,
    device,
    output_dir: Path,
    global_step: int,
    epoch: int,
    max_videos: int = 4,
    min_frames: int = 2,
    panel_w: int = 480,
    panel_h: int = 320,
    fps_ctx: int = 6,
    fps_text: int = 8,
    fps_future: int = 10,
    frame_hold: int = 2,
    diffusion_steps: int = 30,
    diffusion_snapshots: int = 24,
    diffusion_fps: int = 12,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
):
    """
    Training-time visualisation GIF — 2x2 panel layout with PIL-rendered text.

        +------------------------------------------------------------------+
        |  Task / GT / Pred (word-by-word reveal)  + correctness badge     |  <- header
        +------------------------------+-----------------------------------+
        |  Context frame  i / N        |  Action trajectory (GT vs Pred)   |
        +------------------------------+-----------------------------------+
        |  GT future frame  f / P      |  DiT predicted frame  f / P       |
        +------------------------------+-----------------------------------+

    Phases:
      1. Context scroll    - top-left cycles through VLA input frames,
                             held slowly so each frame is easy to read.
      2. Autoregressive    - header Pred text revealed one word at a time.
      3. Future reveal     - bottom row shows GT vs DiT predicted futures,
                             action chart switches to GT + Pred overlay.

    A separate noise->image GIF is also written per sample showing the DiT
    diffusion process for the first predicted future frame.
    """
    if not _CV2_AVAILABLE or not _IMAGEIO_AVAILABLE:
        logger.warning("cv2 or imageio not available; skipping visualization")
        return

    from PIL import Image as PILImage, ImageDraw as PILDraw

    # ── layout ───────────────────────────────────────────────
    HDR_H   = 88
    PANEL_H = panel_h
    PANEL_W = panel_w
    COL_SEP = 4
    ROW_SEP = 26
    CW      = PANEL_W * 2 + COL_SEP
    CH      = HDR_H + PANEL_H + ROW_SEP + PANEL_H

    # RGB colour palette (PIL works in RGB; convert to BGR only at write time).
    C_BG       = (14,  16,  28)
    C_MID      = (22,  26,  44)
    C_SEP      = (60,  68,  104)
    C_TXT      = (220, 226, 240)
    C_DIM      = (90,  98,  120)
    C_WHT      = (255, 255, 255)
    C_GRN      = (130, 235, 170)   # GT
    C_CYN      = (255, 200, 110)   # prediction (amber)
    C_ORG      = (150, 220, 255)   # task / accent blue
    C_PINK     = (235, 130, 150)
    C_ACCENT   = (96,  200, 255)

    F_HEAD     = get_font(20, bold=True)
    F_LABEL    = get_font(14, bold=True)
    F_BODY     = get_font(15)
    F_PANEL    = get_font(13, bold=True)
    F_BAND     = get_font(13, bold=True)
    F_BADGE    = get_font(12, bold=True)

    def _new_canvas():
        """Fresh RGB canvas with column/row dividers and a band label."""
        canvas = PILImage.new("RGB", (CW, CH), C_BG)
        d = PILDraw.Draw(canvas, "RGBA")

        # Header strip with a soft horizontal gradient.
        for x in range(CW):
            t = x / max(CW - 1, 1)
            r = int((1 - t) * 18 + t * C_ACCENT[0] * 0.40)
            g = int((1 - t) * 24 + t * C_ACCENT[1] * 0.40)
            b = int((1 - t) * 40 + t * C_ACCENT[2] * 0.50)
            d.line([(x, 0), (x, HDR_H - 1)], fill=(r, g, b))
        d.rectangle((0, 0, 5, HDR_H), fill=C_ACCENT + (255,))
        d.line((0, HDR_H - 1, CW, HDR_H - 1), fill=C_SEP + (220,), width=1)

        # Vertical column divider.
        d.rectangle((PANEL_W, HDR_H, PANEL_W + COL_SEP, CH), fill=C_SEP)

        # Row band between the action chart and the future-frame row.
        hr = HDR_H + PANEL_H
        d.rectangle((0, hr, CW, hr + ROW_SEP), fill=C_MID)
        d.line((0, hr, CW, hr), fill=C_SEP, width=1)
        d.line((0, hr + ROW_SEP - 1, CW, hr + ROW_SEP - 1),
               fill=C_SEP, width=1)
        band_label = "Future Frame Prediction"
        bbox = d.textbbox((0, 0), band_label, font=F_BAND)
        tw = bbox[2] - bbox[0]
        d.text(((CW - tw) // 2, hr + (ROW_SEP - 16) // 2),
               band_label, font=F_BAND, fill=C_CYN)
        return canvas

    def panel_origin(row, col):
        x0 = col * (PANEL_W + COL_SEP)
        y0 = HDR_H + row * (PANEL_H + ROW_SEP)
        return x0, y0

    def _wrap(text: str, max_chars: int) -> str:
        if not text:
            return ""
        return text if len(text) <= max_chars else text[: max_chars - 1] + "…"

    def draw_header(canvas, task, gt, pred_partial, cursor=False,
                    is_correct=None):
        d = PILDraw.Draw(canvas, "RGBA")
        max_c = max(20, (CW - 30) // 9)
        tick = "|" if cursor else ""

        # Three labelled rows, each with a bold tag and a value.
        labels = [
            ("TASK", _wrap(task, max_c), C_ORG),
            ("GT",   _wrap(gt,   max_c), C_GRN),
            ("PRED", _wrap(pred_partial + tick, max_c), C_CYN),
        ]
        y = 10
        for tag, val, val_col in labels:
            d.text((18, y + 1), tag, font=F_LABEL, fill=(0, 0, 0, 200))
            d.text((17, y),     tag, font=F_LABEL, fill=C_WHT)
            d.text((78, y + 1), val, font=F_BODY,  fill=(0, 0, 0, 200))
            d.text((77, y),     val, font=F_BODY,  fill=val_col)
            y += 22

        # Correctness badge in the top-right.
        if is_correct is not None:
            label = "CORRECT" if is_correct else "WRONG"
            badge_col = (60, 200, 120) if is_correct else (235, 90, 110)
            bbox = d.textbbox((0, 0), label, font=F_BADGE)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pad_x, pad_y = 10, 4
            x1 = CW - 14
            x0 = x1 - tw - pad_x * 2
            y0 = 12
            y1 = y0 + th + pad_y * 2
            try:
                d.rounded_rectangle((x0, y0, x1, y1), radius=10,
                                    fill=badge_col + (235,))
            except AttributeError:
                d.rectangle((x0, y0, x1, y1), fill=badge_col + (235,))
            d.text((x0 + pad_x, y0 + pad_y - 1), label, font=F_BADGE,
                   fill=(255, 255, 255))

    def place_image(canvas, img_bgr, row, col, label,
                    color=C_WHT, border=None):
        x0, y0 = panel_origin(row, col)
        panel = cv2.resize(img_bgr, (PANEL_W, PANEL_H))
        panel_rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
        canvas.paste(PILImage.fromarray(panel_rgb), (x0, y0))

        d = PILDraw.Draw(canvas, "RGBA")
        if border:
            d.rectangle((x0, y0, x0 + PANEL_W - 1, y0 + PANEL_H - 1),
                        outline=border, width=2)
        # Top label ribbon.
        d.rectangle((x0, y0, x0 + PANEL_W, y0 + 24),
                    fill=(0, 0, 0, 160))
        d.text((x0 + 9, y0 + 5), label, font=F_PANEL, fill=color)

    def place_chart(canvas, chart_bgr, row, col, label="", color=C_TXT):
        x0, y0 = panel_origin(row, col)
        resized = cv2.resize(chart_bgr, (PANEL_W, PANEL_H))
        chart_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        canvas.paste(PILImage.fromarray(chart_rgb), (x0, y0))
        if label:
            d = PILDraw.Draw(canvas, "RGBA")
            d.rectangle((x0, y0 + PANEL_H - 22, x0 + PANEL_W, y0 + PANEL_H),
                        fill=(0, 0, 0, 160))
            d.text((x0 + 9, y0 + PANEL_H - 18), label, font=F_PANEL,
                   fill=color)

    def place_placeholder(canvas, row, col, label):
        x0, y0 = panel_origin(row, col)
        d = PILDraw.Draw(canvas, "RGBA")
        d.rectangle((x0, y0, x0 + PANEL_W, y0 + PANEL_H), fill=C_BG)
        # Hatched pattern.
        for offset in range(0, PANEL_W + PANEL_H, 14):
            d.line((x0 + offset, y0, x0, y0 + offset),
                   fill=(28, 32, 50), width=1)
        bbox = d.textbbox((0, 0), label, font=F_LABEL)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text(((x0 + (PANEL_W - tw) // 2), y0 + (PANEL_H - th) // 2),
               label, font=F_LABEL, fill=C_DIM)

    def render_action_chart(gt_arr, pred_arr, reveal_up_to=None):
        """Use the polished shared chart renderer from visualize.py."""
        if gt_arr is None and pred_arr is None:
            return np.full((PANEL_H, PANEL_W, 3), (28, 32, 50), dtype=np.uint8)
        if gt_arr is None:
            # action_comparison_image expects a GT array; substitute zeros.
            gt_arr = np.zeros_like(pred_arr)
        return action_comparison_image(
            pred=pred_arr, gt=gt_arr,
            width=PANEL_W, height=PANEL_H,
            reveal_up_to=reveal_up_to,
            phase_label="Action Trajectory",
        )

    def to_rgb_array(canvas):
        return np.asarray(canvas.convert("RGB"))

    # ── load batch ───────────────────────────────────────────
    model.eval()
    vis_dir = output_dir / f"vis_epoch{epoch}_step{global_step}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    images         = batch["images"].to(device)
    input_ids_full = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    states         = batch["states"].to(device)
    image_mask     = batch.get("image_mask")
    gt_future      = batch.get("future_frames")
    if gt_future is not None:
        gt_future = gt_future.to(device)

    has_predictor   = hasattr(model, "visual_predictor") and model.visual_predictor is not None
    num_predict_cap = getattr(model.visual_predictor, "num_predict_frames", 1) if has_predictor else 0

    B       = images.size(0)
    written = 0

    for b in range(B):
        if written >= max_videos:
            break
        n_imgs = int(image_mask[b].sum().item()) if image_mask is not None else images.size(1)
        if n_imgs < min_frames:
            continue

        # ── decode GT text ───────────────────────────────────
        full_seq = tokenizer.decode(input_ids_full[b].cpu(), skip_special_tokens=False)
        task_text, gt_answer = "", ""
        if "<|im_start|>user" in full_seq:
            blk = full_seq.split("<|im_start|>user")[-1]
            task_text = blk.split("<|im_end|>")[0].strip() if "<|im_end|>" in blk else blk.strip()
        if "<|im_start|>assistant" in full_seq:
            blk = full_seq.split("<|im_start|>assistant")[-1]
            gt_answer = blk.split("<|im_end|>")[0].strip() if "<|im_end|>" in blk else blk.strip()
        for tok in ("<image>", "<state>", "<halo_action>", "<halo_world_video>"):
            task_text = task_text.replace(tok, "").strip()
            gt_answer = gt_answer.replace(tok, "").strip()
        if "Task:" in task_text:
            task_text = task_text.split("Task:")[-1].split("\n")[0].strip()
        task_text = task_text[:90]
        gt_answer = gt_answer[:90]

        # ── build prompt-only ids (strip completed assistant turn) ──
        asst_prefix = "<|im_start|>assistant\n"
        cut = full_seq.rfind(asst_prefix)
        if cut != -1:
            enc     = tokenizer(full_seq[: cut + len(asst_prefix)],
                                return_tensors="pt", add_special_tokens=False)
            ids_in  = enc["input_ids"].to(device)
            mask_in = enc["attention_mask"].to(device)
        else:
            ids_in  = input_ids_full[b : b + 1]
            mask_in = attention_mask[b : b + 1]

        imgs_in     = images[b : b + 1]
        states_in   = states[b : b + 1]
        img_mask_in = image_mask[b : b + 1] if image_mask is not None else None

        # ── DiT visual future prediction ─────────────────────
        pred_rgb = None
        if has_predictor:
            try:
                _, _, vce, wv = model(
                    images=imgs_in, input_ids=ids_in,
                    attention_mask=mask_in, states=states_in,
                    image_mask=img_mask_in,
                )
                n_frames = min(wv.size(1), num_predict_cap) if wv is not None else num_predict_cap
                preds = model.predict_visual_future(
                    past_rgb=imgs_in, visual_context_emb=vce,
                    num_frames=n_frames, world_video_query_hiddens=wv,
                )
                if preds and "rgb" in preds:
                    pred_rgb = preds["rgb"]
            except Exception as e:
                logger.warning("Visual pred failed sample {}: {}", b, e)

        # ── text + action generation ─────────────────────────
        pred_answer = ""
        act_preds   = None
        try:
            pred_text, act_preds = generate_text(
                model, tokenizer, imgs_in, ids_in, mask_in, states_in,
                max_new_tokens=96, device=device,
            )
            pred_answer = pred_text.strip()
            for tok in ("<halo_action>", "<halo_world_video>"):
                pred_answer = pred_answer.replace(tok, "").strip()
            if not pred_answer:
                pred_answer = "(no output)"
            pred_answer = pred_answer[:90]
        except Exception as e:
            logger.warning("Text gen failed sample {}: {}", b, e)
            pred_answer = "(error)"

        # ── action arrays: full GT traj + predicted chunk ────
        gt_actions_np   = None
        pred_actions_np = None
        if batch.get("actions") is not None:
            try:
                gt_actions_np = batch["actions"][b].cpu().float().numpy()  # [T, D]
            except Exception:
                pass
        if act_preds is not None:
            try:
                # Use only the last chunk (conditioned on the last action token)
                # so the pred length matches chunk_size, not n_act*chunk_size.
                pred_actions_np = act_preds[0, -1].cpu().float().numpy()  # [chunk_size, D]
            except Exception:
                pass

        # ── correctness heuristic for badge ──────────────────
        def _word_overlap(g: str, p: str) -> float:
            gw = set(g.lower().split())
            pw = set(p.lower().split())
            if not gw:
                return 1.0
            return len(gw & pw) / max(len(gw), 1)
        is_correct = (_word_overlap(gt_answer, pred_answer) >= 0.5)

        # ── pre-render action charts once per sample ─────────
        chart_gt   = render_action_chart(gt_actions_np, None)
        chart_both = render_action_chart(gt_actions_np, pred_actions_np)

        last_ctx_bgr = unnormalise_image(images[b, n_imgs - 1].cpu(), img_mean, img_std)
        n_pred       = pred_rgb.size(1)  if pred_rgb  is not None else 0
        n_gt_fut     = gt_future.size(1) if gt_future is not None else 0
        pred_words   = pred_answer.split()

        gif_frames    = []
        gif_durations = []          # ms per GIF frame

        # Timing per GIF frame (ms).
        ms_ctx    = max(150, 1000 // max(fps_ctx, 1))
        ms_word   = max( 80, 1000 // max(fps_text, 1))
        ms_future = max( 80, 1000 // max(fps_future, 1))

        # ── Phase 1 : context scroll (slow) ──────────────────
        for i in range(n_imgs):
            c = _new_canvas()
            draw_header(c, task_text, gt_answer, "", is_correct=None)
            ctx_bgr = unnormalise_image(images[b, i].cpu(), img_mean, img_std)
            place_image(c, ctx_bgr, 0, 0, f"Context {i+1}/{n_imgs}")
            place_chart(c, chart_gt, 0, 1, "Action  GT only", C_GRN)
            place_placeholder(c, 1, 0, "GT Future")
            place_placeholder(c, 1, 1, "DiT Predicted")
            for _ in range(frame_hold):
                gif_frames.append(to_rgb_array(c))
                gif_durations.append(ms_ctx)

        # ── Phase 2 : autoregressive text reveal ─────────────
        for wi in range(max(len(pred_words), 1)):
            partial = " ".join(pred_words[: wi + 1]) if pred_words else pred_answer
            is_last = wi == len(pred_words) - 1
            c = _new_canvas()
            draw_header(c, task_text, gt_answer, partial,
                        cursor=not is_last, is_correct=is_correct)
            place_image(c, last_ctx_bgr, 0, 0, f"Context {n_imgs}/{n_imgs}")
            place_chart(c, chart_gt, 0, 1, "Action  GT only", C_GRN)
            place_placeholder(c, 1, 0, "GT Future")
            place_placeholder(c, 1, 1, "DiT Predicted")
            gif_frames.append(to_rgb_array(c))
            gif_durations.append(ms_word)

        # hold on completed prediction for a beat
        for _ in range(2):
            c = _new_canvas()
            draw_header(c, task_text, gt_answer, pred_answer, is_correct=is_correct)
            place_image(c, last_ctx_bgr, 0, 0, f"Context {n_imgs}/{n_imgs}")
            place_chart(c, chart_gt, 0, 1, "Action  GT only", C_GRN)
            place_placeholder(c, 1, 0, "GT Future")
            place_placeholder(c, 1, 1, "DiT Predicted")
            gif_frames.append(to_rgb_array(c))
            gif_durations.append(ms_ctx)

        # ── Phase 2.5 : action trajectory animated reveal ────
        if pred_actions_np is not None:
            n_act_steps = pred_actions_np.shape[0]
            for step_i in range(1, n_act_steps + 1):
                c = _new_canvas()
                draw_header(c, task_text, gt_answer, pred_answer,
                            is_correct=is_correct)
                place_image(c, last_ctx_bgr, 0, 0,
                            f"Context {n_imgs}/{n_imgs}")
                act_chart = render_action_chart(gt_actions_np, pred_actions_np,
                                               reveal_up_to=step_i)
                place_chart(c, act_chart, 0, 1,
                            f"Action  step {step_i}/{n_act_steps}", C_CYN)
                place_placeholder(c, 1, 0, "GT Future")
                place_placeholder(c, 1, 1, "DiT Predicted")
                gif_frames.append(to_rgb_array(c))
                gif_durations.append(ms_future)

        # ── Phase 3 : future frame reveal ────────────────────
        if pred_actions_np is not None:
            act_lbl = "Action  GT (solid)  /  Pred (dashed)"
            act_col = C_CYN
        else:
            act_lbl = "Action  GT only"
            act_col = C_GRN
        n_future_steps = max(n_pred, n_gt_fut, 1)
        for f in range(n_future_steps):
            c = _new_canvas()
            draw_header(c, task_text, gt_answer, pred_answer, is_correct=is_correct)
            place_image(c, last_ctx_bgr, 0, 0, f"Context {n_imgs}/{n_imgs}")
            place_chart(c, chart_both, 0, 1, act_lbl, act_col)

            if gt_future is not None and f < n_gt_fut:
                gt_bgr = unnormalise_image(gt_future[b, f].cpu(), img_mean, img_std)
                place_image(c, gt_bgr, 1, 0,
                            f"GT Future {f+1}/{n_gt_fut}",
                            color=C_GRN, border=C_GRN)
            else:
                place_placeholder(c, 1, 0, f"GT Future {f+1}")

            if pred_rgb is not None and f < n_pred:
                pf     = pred_rgb[0, f].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                pf_bgr = cv2.cvtColor((pf * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                place_image(c, pf_bgr, 1, 1,
                            f"DiT Predicted {f+1}/{n_pred}",
                            color=C_CYN, border=C_CYN)
            else:
                place_placeholder(c, 1, 1, f"DiT Predicted {f+1}")

            for _ in range(frame_hold):
                gif_frames.append(to_rgb_array(c))
                gif_durations.append(ms_future)

        if not gif_frames:
            continue

        gif_path = vis_dir / f"sample_{b}_step{global_step}.gif"
        imageio.v2.mimsave(str(gif_path), gif_frames, duration=gif_durations, loop=0)

        # ── Separate noise -> image GIF for the first predicted frame ──
        if has_predictor:
            try:
                _, _, vce_diff, wv_diff = model(
                    images=imgs_in, input_ids=ids_in,
                    attention_mask=mask_in, states=states_in,
                    image_mask=img_mask_in,
                )
                if vce_diff is not None:
                    n_frames_d = 1
                    if wv_diff is not None:
                        n_frames_d = max(1, min(wv_diff.size(1), num_predict_cap))
                    h_in, w_in = imgs_in.shape[-2], imgs_in.shape[-1]
                    dit_steps = getattr(config, "dit_num_sample_steps", diffusion_steps)
                    snapshots, step_times = sample_dit_with_intermediates(
                        model.visual_predictor,
                        text_emb=vce_diff,
                        height=h_in, width=w_in,
                        num_frames=n_frames_d,
                        num_steps=dit_steps,
                        num_snapshots=dit_steps,
                        world_video_query_hiddens=wv_diff,
                    )
                    if snapshots:
                        diff_path = vis_dir / f"sample_{b}_step{global_step}_diffusion.gif"
                        save_diffusion_gif(
                            snapshots, step_times,
                            output_path=str(diff_path),
                            frame_idx=0,
                            sample_idx=0,
                            canvas_w=PANEL_W,
                            canvas_h=PANEL_H,
                            fps=diffusion_fps,
                        )
            except Exception as e:
                logger.warning("Diffusion GIF failed sample {}: {}", b, e)

        written += 1

    model.train()
    logger.info("Saved {} visualization GIFs -> {}", written, vis_dir)
# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device(args.device)
    logger.info("Device: {}", device)

    # ---- Config ----
    config = HaloVLMConfig(
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        action_chunk_size=args.action_chunk_size,
        num_visual_predict_frames=args.num_predict_frames,
    )
    if args.visual_loss_weight is not None:
        config.visual_loss_weight = args.visual_loss_weight
    visual_loss_weight = config.visual_loss_weight

    # ---- Model ----
    model = HaloVLM(config=config).to(device)
    log_module_parameters(model, model_name="HaloVLM", logger_fn=logger)

    # ---- Dataloader ----
    if args.dataset == "eo":
        train_loader = build_eo_dataloader(
            subset=args.subset,
            split="train",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=config.img_size,
            max_seq_len=args.max_seq_len,
            action_dim=args.action_dim,
            state_dim=args.state_dim,
            shuffle=True,
            max_samples=args.max_samples,
        )
    else:
        if not args.moma_data_root:
            raise ValueError("--moma_data_root is required when --dataset moma")
        moma_workers = args.moma_num_workers
        train_loader = build_airoa_moma_dataloader(
            data_root=args.moma_data_root,
            batch_size=args.batch_size,
            num_workers=moma_workers,
            img_size=config.img_size,
            max_seq_len=args.max_seq_len,
            max_action_len=args.moma_max_action_len,
            action_dim=args.action_dim,
            state_dim=args.state_dim,
            num_sample_frames=args.moma_num_frames,
            num_predict_frames=args.num_predict_frames,
            frame_stride=args.moma_frame_stride,
            camera=args.moma_camera,
            shuffle=True,
            max_samples=args.max_samples,
        )
    logger.info("Dataset size: {}  |  Batches: {}", len(train_loader.dataset), len(train_loader))

    # ---- Grab tokenizer from dataset for debug decoding ----
    tokenizer = train_loader.dataset.tokenizer

    # ---- Optimizer & scheduler ----
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * len(train_loader)))
    scaler = GradScaler("cuda")

    # ---- Checkpoint dir ----
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- TensorBoard ----
    tb_writer = None
    if args.tensorboard_dir:
        if SummaryWriter is None:
            logger.warning(
                "tensorboard package missing; install with: pip install tensorboard"
            )
        else:
            tb_root = Path(args.tensorboard_dir)
            if args.tb_run_name:
                tb_root = tb_root / args.tb_run_name
            else:
                tb_root = tb_root / datetime.now().strftime("%Y%m%d_%H%M%S")
            tb_writer = SummaryWriter(log_dir=str(tb_root))
            logger.info("TensorBoard log dir: {}  (tensorboard --logdir {})", tb_root, tb_root.parent)

    # ---- Training ----
    model.train()
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_lang_loss = 0.0
        epoch_act_loss = 0.0
        epoch_vis_loss = 0.0
        epoch_total_loss = 0.0
        n_steps = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            images = batch["images"].to(device)              # [B, N, 3, H, W]
            input_ids = batch["input_ids"].to(device)        # [B, seq_len]
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)              # [B, seq_len]
            actions = batch["actions"].to(device)            # [B, T, action_dim]
            action_mask = batch["action_mask"].to(device)    # [B, T]
            states = batch["states"].to(device)              # [B, S, state_dim]

            # Debug: decode and print input_ids for the first sample in the batch
            if step == 1 and epoch == 1:
                ids_cpu = input_ids[0].cpu()
                mask_cpu = attention_mask[0].cpu()
                total_tokens = ids_cpu.size(0)
                real_tokens = mask_cpu.sum().item()
                pad_tokens = total_tokens - real_tokens
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                pad_id_count = (ids_cpu == pad_id).sum().item()
                decoded_text = tokenizer.decode(ids_cpu, skip_special_tokens=False)
                logger.info(
                    "=== Debug (sample 0) ===\n"
                    "  Total tokens : {}\n"
                    "  Real tokens  : {}  (attention_mask=1)\n"
                    "  Pad tokens   : {}  (attention_mask=0)\n"
                    "  pad_token_id={} count: {}\n"
                    "  Decoded text:\n{}",
                    total_tokens, real_tokens, pad_tokens,
                    pad_id, pad_id_count, decoded_text,
                )

            # Forward  — returns (logits, action_hiddens)
            # action_hiddens: [B, n_act, emb_dim] conditioning for flow decoder
            # logits, action_hiddens, visual_context_emb = model(
            #     images=images,
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     states=states,
            #     image_mask=batch.get("image_mask"),
            # )

            # # Number of prepended image-patch tokens (for loss alignment)
            # num_images = (input_ids == config.image_token_id).sum(dim=1).max().item()
            # num_patches = (config.img_size // config.patch_size) ** 2
            # num_prepended = num_images * num_patches

            # # Losses
            # lang_loss = compute_language_loss(logits, labels, num_prepended)
            # # Flow matching loss — replaces MLP action MSE loss
            # act_loss = model.compute_flow_loss(action_hiddens, actions, action_mask)
            # vis_loss = model.compute_visual_prediction_loss(
            #     images, batch.get("image_mask"), visual_context_emb,
            # )
            # total_loss = (
            #     lang_loss
            #     + args.action_loss_weight * act_loss
            #     + visual_loss_weight * vis_loss
            # )

            # # Backward
            # optimizer.zero_grad()
            # total_loss.backward()
            # if args.grad_clip > 0:
            #     nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # optimizer.step()
            # scheduler.step()
            # --- Mixed precision forward pass ---
            with autocast("cuda"):
                logits, action_hiddens, visual_context_emb, world_video_query_hiddens = model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    states=states,
                    image_mask=batch.get("image_mask"),
                )

                num_images = (input_ids == config.image_token_id).sum(dim=1).max().item()
                num_patches = (config.img_size // config.patch_size) ** 2
                num_prepended = num_images * num_patches

                lang_loss = compute_language_loss(logits, labels, num_prepended)
                act_loss = model.compute_flow_loss(action_hiddens, actions, action_mask)
                future_frames = batch.get("future_frames")
                if future_frames is not None:
                    future_frames = future_frames.to(device)
                vis_loss = model.compute_visual_prediction_loss(
                    images, batch.get("image_mask"), visual_context_emb,
                    world_video_query_hiddens=world_video_query_hiddens,
                    future_frames=future_frames,
                )
                total_loss = (
                    lang_loss
                    + args.action_loss_weight * act_loss
                    + visual_loss_weight * vis_loss
                )

            # --- Backward with scaler ---
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()

            # Gradient clipping (must unscale first)
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()          # updates the scaler for next iteration
            scheduler.step()         # step scheduler after optimizer step

            global_step += 1
            n_steps += 1
            epoch_lang_loss += lang_loss.item()
            epoch_act_loss += act_loss.item()
            epoch_vis_loss += vis_loss.item()
            epoch_total_loss += total_loss.item()

            # Logging
            if step % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch {} | Step {}/{} | lang={:.4f}  act={:.4f}  vis={:.4f}  total={:.4f} | lr={:.2e}",
                    epoch, step, len(train_loader),
                    lang_loss.item(), act_loss.item(), vis_loss.item(), total_loss.item(), lr,
                )

            if tb_writer is not None:
                lr = optimizer.param_groups[0]["lr"]
                tb_writer.add_scalar("loss/lang", lang_loss.item(), global_step)
                tb_writer.add_scalar("loss/action_flow", act_loss.item(), global_step)
                tb_writer.add_scalar("loss/visual_dit", vis_loss.item(), global_step)
                tb_writer.add_scalar("loss/total", total_loss.item(), global_step)
                tb_writer.add_scalar("optim/lr", lr, global_step)
                tb_writer.add_scalar("epoch", epoch, global_step)

                if args.tb_image_every > 0 and global_step % args.tb_image_every == 0:
                    # First sample in batch: strip of frames [N,3,H,W] in [0,1] for TB
                    grid = images[0, : min(8, images.size(1))].detach().cpu()
                    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
                    tb_writer.add_images(
                        "train/input_frames", grid, global_step, dataformats="NCHW"
                    )

            # Visualisation GIFs
            if args.vis_every > 0 and global_step % args.vis_every == 0:
                generate_training_visualization(
                    model=model, batch=batch, tokenizer=tokenizer, config=config,
                    device=device, output_dir=ckpt_dir, global_step=global_step,
                    epoch=epoch, max_videos=args.vis_max_videos, min_frames=args.vis_min_frames,
                )
                


        # End of epoch summary
        n = max(n_steps, 1)   # guard against empty loader (e.g. max_samples=1 + drop_last)
        elapsed = time.time() - t0
        logger.info(
            "=== Epoch {} done in {:.1f}s  steps={}  | avg lang={:.4f}  act={:.4f}  vis={:.4f}  total={:.4f} ===",
            epoch, elapsed, n_steps,
            epoch_lang_loss / n, epoch_act_loss / n, epoch_vis_loss / n, epoch_total_loss / n,
        )

        if tb_writer is not None:
            tb_writer.add_scalar("epoch_avg/lang", epoch_lang_loss / n, epoch)
            tb_writer.add_scalar("epoch_avg/action_flow", epoch_act_loss / n, epoch)
            tb_writer.add_scalar("epoch_avg/visual_dit", epoch_vis_loss / n, epoch)
            tb_writer.add_scalar("epoch_avg/total", epoch_total_loss / n, epoch)

        # Save checkpoint — atomic (write to .tmp then rename) so a disk-full
        # or crash never leaves a corrupted file.  Prune old checkpoints after.
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = ckpt_dir / f"halo_vla_epoch{epoch}.pt"
            tmp_path  = ckpt_dir / f".tmp_epoch{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": config,
                },
                tmp_path,
            )
            tmp_path.replace(ckpt_path)   # atomic rename on POSIX
            logger.info("Saved checkpoint → {}", ckpt_path)

            # Prune: keep only the latest `--keep_ckpts` checkpoints
            if args.keep_ckpts > 0:
                old_ckpts = sorted(
                    ckpt_dir.glob("halo_vla_epoch*.pt"),
                    key=lambda p: int(p.stem.replace("halo_vla_epoch", "")),
                )
                for old in old_ckpts[: -args.keep_ckpts]:
                    old.unlink()
                    logger.info("Removed old checkpoint {}", old)

    logger.info("Training complete.")
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
        logger.info("TensorBoard writer closed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train Halo-VLA")

    # Data
    p.add_argument(
        "--dataset",
        choices=("eo", "moma"),
        default="eo",
        help="eo: EO-Data1.5M; moma: local AIRoA MoMa clone (see dataloader/airoa_moma_dataset.py)",
    )
    p.add_argument(
        "--moma_data_root",
        default=None,
        help="Path to local airoa-moma clone (episodes.jsonl + videos/). Required for --dataset moma.",
    )
    p.add_argument("--moma_camera", default="head", choices=("hand", "head"))
    p.add_argument(
        "--moma_num_frames",
        type=int,
        default=5,
        help="Context frames per sample fed to the model as input",
    )
    p.add_argument(
        "--num_predict_frames",
        type=int,
        default=5,
        help="Future frames to predict beyond the context window (sets num_visual_predict_frames)",
    )
    p.add_argument("--moma_frame_stride", type=int, default=25)
    p.add_argument("--moma_max_action_len", type=int, default=256)
    p.add_argument(
        "--moma_num_workers",
        type=int,
        default=0,
        help="Video decoding is safest with 0 workers; increase if you use a thread-safe backend",
    )
    p.add_argument("--subset", default="interleave-temporal")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--action_dim", type=int, default=32)
    p.add_argument("--state_dim", type=int, default=32)
    p.add_argument("--action_chunk_size", type=int, default=16)
    p.add_argument("--max_samples", type=int, default=2000,
                    help="Limit dataset to N samples for fast loading (None = all)")

    # Training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--action_loss_weight", type=float, default=1.0)
    p.add_argument(
        "--visual_loss_weight",
        type=float,
        default=None,
        help="Weight for DiT frame/depth/flow loss (default: config.visual_loss_weight)",
    )

    # Device
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Logging & checkpoints
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--keep_ckpts", type=int, default=3,
                    help="Keep only the last N checkpoints (0 = keep all)")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument(
        "--tensorboard_dir",
        type=str,
        default="runs",
        help="TensorBoard root; logs go under runs/<timestamp>/ unless --tb_run_name is set. Empty string disables.",
    )
    p.add_argument(
        "--tb_run_name",
        type=str,
        default="",
        help="Subfolder under tensorboard_dir (default: timestamp)",
    )
    p.add_argument(
        "--tb_image_every",
        type=int,
        default=0,
        help="Log input frame strip every N steps (0 = off; increases log size)",
    )

    # Visualisation during training
    p.add_argument("--vis_every", type=int, default=10,
                    help="Generate vis videos every N steps (0 = disabled)")
    p.add_argument("--vis_max_videos", type=int, default=4,
                    help="Max videos per visualisation round")
    p.add_argument("--vis_min_frames", type=int, default=3,
                    help="Only visualise samples with >= N frames")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
