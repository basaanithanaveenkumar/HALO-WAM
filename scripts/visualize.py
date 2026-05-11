"""
Visualization script for Halo-VLA.

Creates MP4 videos from dataset samples that have >= 3 frames.
Each video shows:
    1. Frames played as a slideshow
    2. Semi-transparent overlay with:
       - Question (user prompt) in white
       - Ground-truth answer in cyan
       - Generated answer in green (correct) or red (incorrect)
    3. Action comparison panel: decoded vs ground-truth action dimensions

Usage:
    # From dataset only (no model, shows GT answer only)
    python scripts/visualize.py --output_dir vis_out --num_samples 20

    # With model checkpoint (generates answers + actions, colour-codes)
    python scripts/visualize.py --checkpoint checkpoints/halo_vla_epoch5.pt \
        --output_dir vis_out --num_samples 20

    # Filter for a minimum frame count
    python scripts/visualize.py --min_frames 5 --num_samples 10
"""

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "Halo_VLA"))

from config import HaloVLMConfig
from dataloader.eo_dataset import EODataset, EODatasetConfig

from loguru import logger


# ---------------------------------------------------------------------------
# Lazy imports — avoid hard crash if libs aren't installed
# ---------------------------------------------------------------------------
def _import_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        raise ImportError(
            "OpenCV is required: pip install opencv-python"
        )


def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# Checkpoint / model helpers (optional — used when --checkpoint is given)
# ---------------------------------------------------------------------------
def load_model(ckpt_path: str, device: torch.device):
    from models.halo_vla import HaloVLM

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", HaloVLMConfig())
    model = HaloVLM(config=config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(
        "Loaded checkpoint {}  (epoch {})", ckpt_path, ckpt.get("epoch", -1)
    )
    return model, config


@torch.no_grad()
def generate_text(model, tokenizer, images, input_ids, attention_mask, states,
                  max_new_tokens=128, device="cpu"):
    """Greedy auto-regressive generation — returns decoded string + action preds."""
    eos_id = tokenizer.eos_token_id
    action_preds = None
    original_len = input_ids.size(1)  # save to decode only new tokens

    for _ in range(max_new_tokens):
        logits, action_hiddens = model(
            images=images, input_ids=input_ids,
            attention_mask=attention_mask, states=states,
        )
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token_id = next_token.item()
        if token_id == eos_id:
            break
        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones(1, 1, dtype=torch.long, device=device)],
            dim=1,
        )
        # Sample actions from flow decoder at inference time
        if action_hiddens is not None:
            act = model.sample_actions(action_hiddens)
        else:
            act = None
        if act is not None:
            action_preds = act

    # Decode only the newly generated tokens
    gen_ids = input_ids[0, original_len:].cpu().tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, action_preds


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------
def extract_question_answer(sample: Dict) -> Tuple[str, str]:
    """Pull the first user question and assistant answer from raw conversation."""
    conv = sample.get("conversation", [])
    question, answer = "", ""
    for i, turn in enumerate(conv):
        role = turn.get("from", turn.get("role", "")).lower()
        value = turn.get("value", turn.get("content", ""))
        if role in ("human", "user") and not question:
            question = value.strip()
        elif role not in ("human", "user", "system") and question and not answer:
            answer = value.strip()
            break
    return question, answer


def simple_match(gt: str, pred: str, threshold: float = 0.5) -> bool:
    """
    Rough correctness heuristic: word-level overlap ratio.
    Returns True if overlap >= threshold.
    """
    gt_words = set(gt.lower().split())
    pred_words = set(pred.lower().split())
    if not gt_words:
        return True
    overlap = len(gt_words & pred_words) / len(gt_words)
    return overlap >= threshold


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def wrap_text(text: str, max_chars: int = 60) -> List[str]:
    """Word-wrap text into lines of at most max_chars."""
    return textwrap.wrap(text, width=max_chars) or [""]


def draw_text_overlay(frame: np.ndarray, lines: List[Tuple[str, Tuple[int, int, int]]],
                      alpha: float = 0.55, font_scale: float = 0.5,
                      thickness: int = 1, line_gap: int = 22,
                      y_start: int | None = None):
    """
    Draw semi-transparent text block on a frame (in-place).

    Args:
        lines: list of (text, BGR_colour) tuples
        alpha: overlay transparency (0 = invisible, 1 = opaque)
    """
    cv2 = _import_cv2()
    H, W = frame.shape[:2]
    if y_start is None:
        y_start = H - len(lines) * line_gap - 10

    # Compute bounding box for the overlay region
    y0 = max(y_start - 8, 0)
    y1 = min(y_start + len(lines) * line_gap + 4, H)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (W, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    y = y_start
    for text, colour in lines:
        cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_DUPLEX,
                    font_scale, colour, thickness, cv2.LINE_AA)
        y += line_gap


def action_comparison_image(pred: Optional[np.ndarray],
                            gt: np.ndarray,
                            width: int = 640,
                            height: int = 200,
                            reveal_up_to: Optional[int] = None) -> np.ndarray:
    """
    Render predicted vs GT action trajectories on a dark-themed plot.

    Each action dimension (e.g. x, y, z, roll, pitch, yaw, gripper) is
    drawn in its own unique colour picked from a perceptual colour-map:
      • GT  dimensions use the 'cool' spectrum  (blues → magentas)
      • Pred dimensions use the 'autumn' spectrum (reds → yellows)
    This makes it easy to visually match and compare individual DoFs
    across the two trajectories.

    Args:
        pred:  predicted actions [T_p, D] or [D] (may be None).
        gt:    ground-truth actions [T_g, D] or [D].
        reveal_up_to: number of predicted **timesteps** to reveal
                      (None = show all).  GT is always fully shown.

    Returns an HxW BGR numpy image.
    """
    plt = _import_matplotlib()
    import matplotlib.cm as cm

    # Ensure 2-D [T, D]
    if gt.ndim == 1:
        gt = gt[np.newaxis, :]
    if pred is not None and pred.ndim == 1:
        pred = pred[np.newaxis, :]

    n_gt_steps, n_dims = gt.shape
    gt_t = np.arange(n_gt_steps)

    # ── Colour palettes ──────────────────────────────────────────────
    # Pick evenly-spaced hues from two distinct colour-maps so that:
    #   • Each dimension d gets its own recognisable colour.
    #   • GT and prediction palettes are visually distinct families.
    # 'cool' gives blues→pinks for GT;  'plasma' gives vivid purple→yellow for pred.
    gt_cmap  = cm.get_cmap("cool",   n_dims)   # one colour per dimension
    pred_cmap = cm.get_cmap("plasma", n_dims)   # vibrant purple→amber spectrum
    # Pre-compute RGBA tuples for each dimension index
    gt_colours   = [gt_cmap(d)   for d in range(n_dims)]
    pred_colours = [pred_cmap(d) for d in range(n_dims)]

    # --- Dark theme ---
    with plt.rc_context({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#444466",
        "axes.labelcolor": "#cccccc",
        "xtick.color": "#999999",
        "ytick.color": "#999999",
        "text.color": "#dddddd",
        "grid.color": "#2a2a4a",
        "grid.alpha": 0.6,
    }):
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

        # ── GT: each dimension in its own colour from the 'cool' spectrum ──
        for d in range(n_dims):
            colour = gt_colours[d]
            # Bold line for each GT dimension so every DoF is visible
            ax.plot(gt_t, gt[:, d], color=colour, linewidth=1.4,
                    alpha=0.85,
                    label=f"GT d{d}" if d < 8 else "_nolegend_")
            # Light filled band per dimension to hint at the trajectory
            if n_gt_steps > 1:
                ax.fill_between(gt_t,
                                gt[:, d] - 0.02 * (gt[:, d].max() - gt[:, d].min() + 1e-6),
                                gt[:, d] + 0.02 * (gt[:, d].max() - gt[:, d].min() + 1e-6),
                                color=colour, alpha=0.10)

        # ── Predicted: each dimension in its own colour from 'autumn' ──
        if pred is not None:
            n_pred_steps = pred.shape[0]
            pred_t = np.arange(n_pred_steps)
            k = min(reveal_up_to, n_pred_steps) if reveal_up_to is not None else n_pred_steps

            if k > 0:
                pred_slice = pred[:k, :]
                pred_t_slice = pred_t[:k]

                for d in range(min(n_dims, pred_slice.shape[1])):
                    colour = pred_colours[d]
                    # Dashed line distinguishes prediction from GT at a glance
                    ax.plot(pred_t_slice, pred_slice[:, d],
                            color=colour, linewidth=1.4, alpha=0.90,
                            linestyle="--",
                            label=f"Pred d{d}" if d < 8 else "_nolegend_")

                    # Animated cursor dot at the latest revealed timestep
                    if k < n_pred_steps:
                        ax.plot(pred_t_slice[-1], pred_slice[-1, d], "o",
                                color=colour, markersize=4, zorder=5,
                                markeredgecolor="#fff", markeredgewidth=0.6)

        # --- Styling ---
        # Put the legend outside the axes so it doesn't occlude data
        ax.legend(fontsize=5, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0),
                  facecolor="#1a1a2e", edgecolor="#444466",
                  labelcolor="#cccccc", framealpha=0.85,
                  ncol=1, borderaxespad=0.2)
        ax.set_xlabel("Timestep", fontsize=7)
        ax.set_ylabel("Value", fontsize=7)
        ax.set_title("Action Trajectory — GT (solid) vs Predicted (dashed)",
                      fontsize=8, color="#eeeeee", pad=4)
        ax.tick_params(labelsize=6)
        ax.grid(True, linewidth=0.5)
        fig.tight_layout(pad=0.4)

        # Render to numpy array
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()
        buf = buf.reshape((h, w, 4))[:, :, :3]  # drop alpha → RGB
        plt.close(fig)

    cv2 = _import_cv2()
    buf = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    buf = cv2.resize(buf, (width, height))
    return buf


# ---------------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------------
def create_video(
    frames: List[np.ndarray],
    question: str,
    gt_answer: str,
    pred_answer: Optional[str],
    gt_actions: Optional[np.ndarray],
    pred_actions: Optional[np.ndarray],
    output_path: str,
    fps: int = 3,
    frame_hold: int = 1,
    save_gif: bool = True,
    token_reveal_fps: int = 5,
    action_reveal_fps: int = 8,
):
    """
    Write a visualisation for a single sample with three animated phases.

    Phase 1 — Frame slideshow:
        Each image frame is displayed for `frame_hold` seconds with the
        question (white) and ground-truth answer (cyan) overlaid.

    Phase 2 — Token-by-token prediction reveal:
        On the last image frame, the predicted answer appears one word at a
        time.  Colour-coded green (correct) or red (wrong).

    Phase 3 — Action bar-chart animation:
        The GT action bars are shown fully, while the predicted action bars
        grow in one dimension at a time.

    Saves both MP4 and GIF.
    """
    cv2 = _import_cv2()

    if not frames:
        logger.warning("No frames to write for {}", output_path)
        return

    H, W = frames[0].shape[:2]

    # ---- Action trajectories for EO1-style per-dim subplots ----
    has_actions = gt_actions is not None and gt_actions.size > 0
    gt_act_traj, pred_act_traj = None, None
    if has_actions:
        # Keep full [T, D] trajectory (or promote [D] → [1, D])
        gt_act_traj = gt_actions if gt_actions.ndim >= 2 else gt_actions[np.newaxis, :]
        if pred_actions is not None and pred_actions.size > 0:
            pred_act_traj = pred_actions if pred_actions.ndim >= 2 else pred_actions[np.newaxis, :]
            min_d = min(gt_act_traj.shape[-1], pred_act_traj.shape[-1])
            gt_act_traj = gt_act_traj[:, :min_d]
            pred_act_traj = pred_act_traj[:, :min_d]

    # Total canvas height (includes action chart area for phases 2-3)
    chart_h = 300 if has_actions else 0
    total_H = H + chart_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, total_H))
    if not writer.isOpened():
        logger.error("Cannot open video writer for {}", output_path)
        return

    is_correct = None
    if pred_answer is not None:
        is_correct = simple_match(gt_answer, pred_answer)

    composed_frames: List[np.ndarray] = []

    def _write(frame_bgr: np.ndarray, copies: int = 1):
        """Write frame to MP4 and collect for GIF."""
        for _ in range(max(copies, 1)):
            writer.write(frame_bgr)
        composed_frames.append(frame_bgr.copy())

    # ==================================================================
    # PHASE 1 — Frame slideshow with Q + GT overlay
    # ==================================================================
    for idx, raw_frame in enumerate(frames):
        canvas = raw_frame.copy()

        lines: List[Tuple[str, Tuple[int, int, int]]] = []
        lines.append((f"Frame {idx + 1}/{len(frames)}", (180, 180, 180)))
        for wl in wrap_text(f"Q: {question}", max_chars=70):
            lines.append((wl, (255, 255, 255)))
        for wl in wrap_text(f"GT: {gt_answer}", max_chars=70):
            lines.append((wl, (255, 204, 0)))  # BGR cyan-ish

        draw_text_overlay(canvas, lines, alpha=0.7, y_start=20,
                          line_gap=26, font_scale=0.55, thickness=1)

        # Show GT-only action trajectory during phase 1
        if chart_h > 0 and gt_act_traj is not None:
            chart = action_comparison_image(
                None, gt_act_traj, width=W, height=chart_h,
            )
            combined = np.vstack([canvas, chart])
        elif chart_h > 0:
            blank_chart = np.zeros((chart_h, W, 3), dtype=np.uint8)
            combined = np.vstack([canvas, blank_chart])
        else:
            combined = canvas

        _write(combined, copies=max(frame_hold * fps, 1))

    # ==================================================================
    # PHASE 2 — Token-by-token prediction reveal
    # ==================================================================
    if pred_answer is not None:
        pred_words = pred_answer.split()
        colour = (0, 255, 128) if is_correct else (128, 0, 255)  # BGR bright green / magenta
        tag = "CORRECT" if is_correct else "WRONG"
        last_frame = frames[-1].copy()  # use last image as background

        for n_words in range(1, len(pred_words) + 1):
            canvas = last_frame.copy()
            partial_text = " ".join(pred_words[:n_words])
            cursor_char = "|" if n_words < len(pred_words) else ""

            lines: List[Tuple[str, Tuple[int, int, int]]] = []
            lines.append((f"Generating...", (180, 180, 180)))
            for wl in wrap_text(f"Q: {question}", max_chars=70):
                lines.append((wl, (255, 255, 255)))
            for wl in wrap_text(f"GT: {gt_answer}", max_chars=70):
                lines.append((wl, (255, 204, 0)))

            # Show partial prediction with typing cursor
            label = f"Pred [{tag}]: {partial_text}{cursor_char}"
            for wl in wrap_text(label, max_chars=70):
                lines.append((wl, colour))

            draw_text_overlay(canvas, lines, alpha=0.7, y_start=20,
                              line_gap=26, font_scale=0.55, thickness=1)

            # Show GT-only action trajectory during phase 2
            if chart_h > 0 and gt_act_traj is not None:
                chart = action_comparison_image(
                    None, gt_act_traj, width=W, height=chart_h,
                )
                combined = np.vstack([canvas, chart])
            elif chart_h > 0:
                blank_chart = np.zeros((chart_h, W, 3), dtype=np.uint8)
                combined = np.vstack([canvas, blank_chart])
            else:
                combined = canvas

            # Each token frame is 1 sub-frame (fast reveal)
            _write(combined, copies=1)

        # Hold the final prediction for a moment
        if chart_h > 0 and gt_act_traj is not None:
            chart = action_comparison_image(
                None, gt_act_traj, width=W, height=chart_h,
            )
            combined = np.vstack([canvas, chart])
        _write(combined, copies=max(fps, 1))

    # ==================================================================
    # PHASE 3 — Animated action trajectory (predicted line drawn step-by-step)
    # ==================================================================
    if has_actions and gt_act_traj is not None:
        n_pred_steps = pred_act_traj.shape[0] if pred_act_traj is not None else 0
        # Build a static image canvas (last frame + full text)
        bg_canvas = frames[-1].copy()
        lines: List[Tuple[str, Tuple[int, int, int]]] = []
        lines.append(("Action Trajectory", (180, 180, 180)))
        for wl in wrap_text(f"Q: {question}", max_chars=70):
            lines.append((wl, (255, 255, 255)))
        for wl in wrap_text(f"GT: {gt_answer}", max_chars=70):
            lines.append((wl, (255, 204, 0)))
        if pred_answer is not None:
            colour = (0, 255, 128) if is_correct else (128, 0, 255)
            tag = "CORRECT" if is_correct else "WRONG"
            for wl in wrap_text(f"Pred [{tag}]: {pred_answer}", max_chars=70):
                lines.append((wl, colour))
        draw_text_overlay(bg_canvas, lines, alpha=0.7, y_start=20,
                          line_gap=26, font_scale=0.55, thickness=1)

        # Animate: draw predicted trajectory one timestep at a time
        n_anim = max(n_pred_steps, 1)
        for reveal in range(0, n_anim + 1):
            chart = action_comparison_image(
                pred_act_traj, gt_act_traj,
                width=W, height=chart_h,
                reveal_up_to=reveal if pred_act_traj is not None else None,
            )
            combined = np.vstack([bg_canvas, chart])
            _write(combined, copies=1)

        # Hold the final chart
        final_chart = action_comparison_image(
            pred_act_traj, gt_act_traj, width=W, height=chart_h,
        )
        final_combined = np.vstack([bg_canvas, final_chart])
        _write(final_combined, copies=max(fps * 2, 1))

    writer.release()

    # --- Save GIF ---
    if save_gif:
        from PIL import Image as PILImage

        gif_path = output_path.rsplit(".", 1)[0] + ".gif"
        # Adaptive durations: frame slideshow is slower, animation is faster
        base_dur = int(frame_hold * 1000)   # ms per slideshow frame
        anim_dur = int(1000 / fps)         # ms per animation frame

        pil_frames = []
        durations = []

        n_slideshow = len(frames)
        for i, f in enumerate(composed_frames):
            rgb = f[:, :, ::-1].copy()
            pil_frames.append(PILImage.fromarray(rgb))
            # First n_slideshow entries are slideshow (held longer), rest are animation
            if i < n_slideshow:
                durations.append(base_dur)
            else:
                durations.append(anim_dur)

        if pil_frames:
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=durations,
                loop=0,
            )
            logger.info("Saved GIF  → {}", gif_path)

    logger.info("Saved video → {}  ({} phases, {}x{})", output_path, 3, W, total_H)


# ---------------------------------------------------------------------------
# Frame extraction from dataset sample
# ---------------------------------------------------------------------------
def unnormalise_image(tensor: torch.Tensor, mean: Tuple, std: Tuple) -> np.ndarray:
    """Convert a normalised [3, H, W] tensor back to uint8 BGR for OpenCV."""
    cv2 = _import_cv2()
    img = tensor.clone().cpu().float()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()  # [H, W, 3] RGB float
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------
def main(args):
    cv2 = _import_cv2()
    device = torch.device(args.device)

    # --- Output directory ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataset (raw HF, not expanded) ---
    ds_cfg = EODatasetConfig(
        subset=args.subset,
        split="train",
        img_size=args.img_size,
        max_seq_len=args.max_seq_len,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
    )
    dataset = EODataset(config=ds_cfg)
    tokenizer = dataset.tokenizer

    # --- Optional model ---
    model, config = None, None
    if args.checkpoint:
        model, config = load_model(args.checkpoint, device)

    # --- Iterate raw HF samples, filter by frame count ---
    hf_ds = dataset.dataset
    n_total = len(hf_ds)
    written = 0

    logger.info(
        "Scanning {} samples for >= {} frames (max {} videos) …",
        n_total, args.min_frames, args.num_samples,
    )

    for sample_idx in range(n_total):
        if written >= args.num_samples:
            break

        sample = hf_ds[sample_idx]

        # --- Check frame count ---
        raw_imgs = sample.get("image")
        if raw_imgs is None:
            continue
        if not isinstance(raw_imgs, (list, tuple)):
            raw_imgs = [raw_imgs]
        n_frames = len(raw_imgs)
        if n_frames < args.min_frames:
            continue

        # --- Extract Q / A ---
        question, gt_answer = extract_question_answer(sample)
        if not question and not gt_answer:
            continue

        # --- Process images via dataset transforms ---
        images_tensor = dataset._process_images(sample)  # [N, 3, H, W]
        N_img = images_tensor.size(0)

        # --- Convert to BGR numpy for video ---
        bgr_frames = [
            unnormalise_image(images_tensor[i], ds_cfg.img_mean, ds_cfg.img_std)
            for i in range(N_img)
        ]

        # Resize to a common canvas size for the video
        canvas_h, canvas_w = args.canvas_h, args.canvas_w
        bgr_frames = [cv2.resize(f, (canvas_w, canvas_h)) for f in bgr_frames]

        # --- Process actions (GT) ---
        gt_actions_t, action_mask_t = dataset._process_actions(sample)
        gt_actions = gt_actions_t.numpy() if gt_actions_t.numel() > 0 else None

        # --- Model prediction (if checkpoint provided) ---
        pred_answer = None
        pred_actions = None
        if model is not None:
            # Prepare model inputs
            input_ids, attn_mask, _ = dataset._process_conversation(sample, pair_idx=0)
            states_t, _ = dataset._process_states(sample)

            imgs_in = images_tensor.unsqueeze(0).to(device)   # [1, N, 3, H, W]
            ids_in = input_ids.unsqueeze(0).to(device)         # [1, seq_len]
            mask_in = attn_mask.unsqueeze(0).to(device)
            if states_t.numel() > 0:
                states_in = states_t.unsqueeze(0).to(device)   # [1, S, state_dim]
            else:
                states_in = torch.zeros(1, 1, config.state_dim, device=device)

            pred_text, act_preds = generate_text(
                model, tokenizer, imgs_in, ids_in, mask_in, states_in,
                max_new_tokens=args.max_new_tokens, device=device,
            )
            # Strip system / user prompt from the full decoded text — keep only
            # the generated portion after "assistant\n"
            if "assistant" in pred_text:
                pred_answer = pred_text.split("assistant")[-1].strip()
            else:
                pred_answer = pred_text.strip()

            if act_preds is not None:
                # [1, n_act, chunk, dim] → [n_act * chunk, dim]
                pred_actions = act_preds[0].view(-1, act_preds.size(-1)).cpu().numpy()

        # --- Create video ---
        video_name = f"sample_{sample_idx:05d}_{n_frames}frames.mp4"
        video_path = str(out_dir / video_name)
        create_video(
            frames=bgr_frames,
            question=question,
            gt_answer=gt_answer,
            pred_answer=pred_answer,
            gt_actions=gt_actions,
            pred_actions=pred_actions,
            output_path=video_path,
            fps=args.fps,
            frame_hold=args.frame_hold,
        )
        written += 1

    logger.info("Done — {} videos written to {}", written, out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Halo-VLA Visualisation")

    # Data
    p.add_argument("--subset", default="interleave-temporal")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--action_dim", type=int, default=32)
    p.add_argument("--state_dim", type=int, default=32)

    # Filtering
    p.add_argument("--min_frames", type=int, default=3,
                    help="Only visualise samples with >= this many frames")
    p.add_argument("--num_samples", type=int, default=20,
                    help="Max number of videos to produce")

    # Model (optional)
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Path to .pt checkpoint for generated answers")
    p.add_argument("--max_new_tokens", type=int, default=128)

    # Video
    p.add_argument("--output_dir", default="vis_out")
    p.add_argument("--fps", type=int, default=3, help="Video FPS")
    p.add_argument("--frame_hold", type=int, default=2,
                    help="Seconds to hold each frame")
    p.add_argument("--canvas_w", type=int, default=728,
                    help="Video canvas width (2x of 364)")
    p.add_argument("--canvas_h", type=int, default=504,
                    help="Video canvas height (2x of 252)")

    # Device
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
