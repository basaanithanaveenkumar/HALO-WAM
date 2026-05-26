"""
Visualization script for Halo-VLA.

Creates polished MP4 + GIF visualisations from dataset samples (>= 3 frames).
Each video has three phases:
    1. CONTEXT  — input frames played slowly with the question/GT overlay.
    2. ANSWER   — token-by-token reveal of the model's predicted answer.
    3. ACTION   — animated GT vs predicted action trajectory plot.

When a model checkpoint is supplied and the DiT visual predictor is available,
a *separate* GIF is written per sample showing the diffusion noise -> image
conversion for one predicted future frame.

Usage:
    # Dataset only (no model, GT-only video)
    python scripts/visualize.py --output_dir vis_out --num_samples 20

    # With model checkpoint (text + action + diffusion GIF)
    python scripts/visualize.py --checkpoint checkpoints/halo_vla_epoch5.pt \
        --output_dir vis_out --num_samples 20

    # Minimum frame filter
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
        raise ImportError("OpenCV is required: pip install opencv-python")


def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _import_pil():
    from PIL import Image, ImageDraw, ImageFont
    return Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Font loading — DejaVu Sans is on every standard Linux distribution and
# supports Latin + extended characters, so curly-quotes, em-dashes, arrows
# etc. render properly instead of as ??? boxes.
# ---------------------------------------------------------------------------
_FONT_CACHE: Dict[Tuple[str, int], "object"] = {}
_FONT_CANDIDATES_REGULAR = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
]
_FONT_CANDIDATES_BOLD = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:\\Windows\\Fonts\\arialbd.ttf",
]


def get_font(size: int, bold: bool = False):
    """Load a TTF font with a sensible fallback. Cached per (style, size)."""
    _, _, ImageFont = _import_pil()
    key = ("bold" if bold else "regular", size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    candidates = _FONT_CANDIDATES_BOLD if bold else _FONT_CANDIDATES_REGULAR
    for path in candidates:
        if Path(path).is_file():
            try:
                font = ImageFont.truetype(path, size)
                _FONT_CACHE[key] = font
                return font
            except Exception:
                continue
    # Last resort: PIL's built-in bitmap font (won't be pretty but won't crash).
    font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


# ---------------------------------------------------------------------------
# Checkpoint / model helpers (optional — used when --checkpoint is given)
# ---------------------------------------------------------------------------
def load_model(ckpt_path: str, device: torch.device):
    from models.halo_vla import HaloVLM

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", HaloVLMConfig())
    model = HaloVLM(config=config).to(device)
    miss, _ = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if miss:
        logger.warning(
            "{} missing keys after load (random init), e.g. {}",
            len(miss),
            miss[:5],
        )
    model.eval()
    logger.info(
        "Loaded checkpoint {}  (epoch {})", ckpt_path, ckpt.get("epoch", -1)
    )
    return model, config


@torch.no_grad()
def generate_text(model, tokenizer, images, input_ids, attention_mask, states,
                  max_new_tokens=128, device="cpu"):
    """Greedy auto-regressive generation. Returns decoded string + action preds."""
    eos_id = tokenizer.eos_token_id
    action_preds = None
    original_len = input_ids.size(1)

    for _ in range(max_new_tokens):
        logits, action_hiddens, _, _ = model(
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
        if action_hiddens is not None:
            act = model.sample_actions(action_hiddens)
        else:
            act = None
        if act is not None:
            action_preds = act

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
    for turn in conv:
        role = turn.get("from", turn.get("role", "")).lower()
        value = turn.get("value", turn.get("content", ""))
        if role in ("human", "user") and not question:
            question = value.strip()
        elif role not in ("human", "user", "system") and question and not answer:
            answer = value.strip()
            break
    return question, answer


def simple_match(gt: str, pred: str, threshold: float = 0.5) -> bool:
    """Rough word-overlap correctness heuristic."""
    gt_words = set(gt.lower().split())
    pred_words = set(pred.lower().split())
    if not gt_words:
        return True
    overlap = len(gt_words & pred_words) / len(gt_words)
    return overlap >= threshold


def wrap_text(text: str, max_chars: int = 60) -> List[str]:
    return textwrap.wrap(text, width=max_chars) or [""]


# ---------------------------------------------------------------------------
# PIL drawing helpers — these replace the OpenCV Hershey-font text path,
# which can only render ASCII (anything else, including curly quotes and
# em-dashes that often appear in dataset prompts, shows as "???").
# ---------------------------------------------------------------------------
def _bgr_to_pil(frame_bgr: np.ndarray):
    """Convert an OpenCV BGR uint8 frame to a PIL Image (RGB)."""
    Image, _, _ = _import_pil()
    return Image.fromarray(frame_bgr[:, :, ::-1].copy())


def _pil_to_bgr(img) -> np.ndarray:
    """Convert a PIL Image (RGB) back to a BGR uint8 numpy frame."""
    arr = np.asarray(img.convert("RGB"))
    return arr[:, :, ::-1].copy()


def draw_header(img, phase: str, sub: str = "", accent: Tuple[int, int, int] = (96, 200, 255)):
    """Draw a slim gradient header strip with phase label and an optional sub-label."""
    Image, ImageDraw, _ = _import_pil()
    W, H = img.size
    bar_h = 38

    # Gradient strip (dark teal -> accent)
    gradient = Image.new("RGB", (W, bar_h), (0, 0, 0))
    for x in range(W):
        t = x / max(W - 1, 1)
        r = int((1 - t) * 18 + t * accent[0] * 0.35)
        g = int((1 - t) * 22 + t * accent[1] * 0.35)
        b = int((1 - t) * 36 + t * accent[2] * 0.45)
        for y in range(bar_h):
            gradient.putpixel((x, y), (r, g, b))
    img.paste(gradient, (0, 0))

    draw = ImageDraw.Draw(img, "RGBA")
    # Accent left ribbon
    draw.rectangle((0, 0, 6, bar_h), fill=accent + (255,))
    # Bottom hairline divider
    draw.line((0, bar_h - 1, W, bar_h - 1), fill=(255, 255, 255, 70), width=1)

    title_font = get_font(20, bold=True)
    sub_font = get_font(14)
    draw.text((18, 8), phase, font=title_font, fill=(255, 255, 255))
    if sub:
        # Right-aligned sub label
        right_pad = 16
        bbox = draw.textbbox((0, 0), sub, font=sub_font)
        tw = bbox[2] - bbox[0]
        draw.text((W - tw - right_pad, 11), sub, font=sub_font,
                  fill=(220, 230, 245))
    return bar_h


def draw_panel(img, lines: List[Tuple[str, Tuple[int, int, int], bool]],
               anchor: str = "bottom", padding: int = 14, line_gap: int = 6,
               font_size: int = 17, label_font_size: int = 15,
               max_chars: int = 70):
    """
    Draw a translucent rounded panel containing labelled text lines.

    ``lines`` is a list of (text, (r,g,b), is_label) tuples. Label lines render
    in bold and slightly smaller; body lines wrap to ``max_chars``.
    """
    Image, ImageDraw, _ = _import_pil()
    W, H = img.size

    body_font = get_font(font_size)
    label_font = get_font(label_font_size, bold=True)

    # Resolve wrapped text rows up-front so we can size the panel.
    rendered_rows: List[Tuple[str, Tuple[int, int, int], object, int]] = []
    for text, colour, is_label in lines:
        font = label_font if is_label else body_font
        for chunk in wrap_text(text, max_chars=max_chars):
            bbox = body_font.getbbox(chunk if chunk else " ")
            row_h = max(bbox[3] - bbox[1], font_size) + line_gap
            rendered_rows.append((chunk, colour, font, row_h))

    panel_h = sum(r[3] for r in rendered_rows) + padding * 2

    if anchor == "bottom":
        y0 = H - panel_h - 10
    else:
        y0 = 10
    y0 = max(y0, 44)  # leave space for header

    # Translucent rounded background
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    radius = 14
    margin = 12
    rect = (margin, y0, W - margin, y0 + panel_h)
    try:
        od.rounded_rectangle(rect, radius=radius, fill=(8, 10, 22, 200))
    except AttributeError:
        od.rectangle(rect, fill=(8, 10, 22, 200))
    # Subtle accent stripe on the left side of the panel
    od.rectangle((margin, y0, margin + 4, y0 + panel_h), fill=(96, 200, 255, 230))

    img.paste(overlay, (0, 0), overlay)

    draw = ImageDraw.Draw(img, "RGBA")
    y = y0 + padding
    for text, colour, font, row_h in rendered_rows:
        # 1px shadow for readability against busy backgrounds.
        draw.text((margin + 16, y + 1), text, font=font, fill=(0, 0, 0, 200))
        draw.text((margin + 15, y), text, font=font, fill=colour + (255,))
        y += row_h


def draw_correctness_badge(img, is_correct: bool, top: int):
    """Small pill badge in the top-right area indicating CORRECT/WRONG."""
    Image, ImageDraw, _ = _import_pil()
    W, _ = img.size
    label = "CORRECT" if is_correct else "WRONG"
    colour = (60, 200, 120) if is_correct else (235, 90, 110)
    font = get_font(13, bold=True)

    draw = ImageDraw.Draw(img, "RGBA")
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 10, 4
    x1 = W - 16
    x0 = x1 - tw - pad_x * 2
    y0 = top + 2
    y1 = y0 + th + pad_y * 2
    try:
        draw.rounded_rectangle((x0, y0, x1, y1), radius=10,
                               fill=colour + (220,))
    except AttributeError:
        draw.rectangle((x0, y0, x1, y1), fill=colour + (220,))
    draw.text((x0 + pad_x, y0 + pad_y - 1), label, font=font,
              fill=(255, 255, 255, 255))


def compose_frame(
    bgr_frame: np.ndarray,
    phase_label: str,
    sub_label: str,
    panel_lines: List[Tuple[str, Tuple[int, int, int], bool]],
    is_correct: Optional[bool] = None,
    accent: Tuple[int, int, int] = (96, 200, 255),
) -> np.ndarray:
    """Render header + bottom info panel onto an image frame. Returns BGR uint8."""
    img = _bgr_to_pil(bgr_frame)
    header_h = draw_header(img, phase_label, sub_label, accent=accent)
    if is_correct is not None:
        draw_correctness_badge(img, is_correct, top=header_h + 6)
    draw_panel(img, panel_lines, anchor="bottom")
    return _pil_to_bgr(img)


# ---------------------------------------------------------------------------
# Action comparison plot — polished dark theme with distinct per-DoF colours,
# legend in two columns underneath the axes (so it never occludes data), a
# vertical cursor at the latest revealed timestep, and a clear ASCII title.
# ---------------------------------------------------------------------------
def action_comparison_image(pred: Optional[np.ndarray],
                            gt: np.ndarray,
                            width: int = 640,
                            height: int = 240,
                            reveal_up_to: Optional[int] = None,
                            phase_label: str = "Action Trajectory") -> np.ndarray:
    """
    Render predicted vs GT action trajectories.

    Solid colour-coded line per DoF for GT, dashed line of the same colour
    family for the prediction, plus a vertical cursor at the current step.
    Each dimension uses a distinct hue from the ``tab20`` palette.
    """
    plt = _import_matplotlib()
    import matplotlib.cm as cm

    if gt.ndim == 1:
        gt = gt[np.newaxis, :]
    if pred is not None and pred.ndim == 1:
        pred = pred[np.newaxis, :]

    n_gt_steps, n_dims = gt.shape
    gt_t = np.arange(n_gt_steps)

    # Palette: tab20 gives 20 distinct hues. We use even indices for GT and
    # the matching odd index for the predicted version of the same DoF, so
    # GT-d0 and Pred-d0 share a colour family but are visually separable.
    base = cm.get_cmap("tab20", 20)
    gt_colours = [base(min((d * 2) % 20, 19)) for d in range(n_dims)]
    pred_colours = [base(min((d * 2 + 1) % 20, 19)) for d in range(n_dims)]

    with plt.rc_context({
        "figure.facecolor": "#0d1226",
        "axes.facecolor": "#141a30",
        "axes.edgecolor": "#3a4060",
        "axes.labelcolor": "#dde2f0",
        "xtick.color": "#a0a8c0",
        "ytick.color": "#a0a8c0",
        "text.color": "#e8ecf7",
        "grid.color": "#242a45",
        "grid.linestyle": "--",
        "grid.alpha": 0.55,
        "axes.titleweight": "bold",
    }):
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

        # GT lines — solid, medium weight.
        for d in range(n_dims):
            colour = gt_colours[d]
            ax.plot(gt_t, gt[:, d], color=colour, linewidth=1.6,
                    alpha=0.95,
                    label=f"GT d{d}" if d < 8 else "_nolegend_")

        # Predicted lines — dashed, with a cursor marker at the latest
        # revealed step so the user can see the animation progressing.
        if pred is not None:
            n_pred_steps = pred.shape[0]
            pred_t = np.arange(n_pred_steps)
            k = min(reveal_up_to, n_pred_steps) if reveal_up_to is not None else n_pred_steps

            if k > 0:
                pred_slice = pred[:k, :]
                pred_t_slice = pred_t[:k]

                for d in range(min(n_dims, pred_slice.shape[1])):
                    colour = pred_colours[d]
                    ax.plot(pred_t_slice, pred_slice[:, d],
                            color=colour, linewidth=1.7, alpha=0.95,
                            linestyle=(0, (4, 2)),
                            label=f"Pred d{d}" if d < 8 else "_nolegend_")

                    if k < n_pred_steps:
                        ax.plot(pred_t_slice[-1], pred_slice[-1, d], "o",
                                color=colour, markersize=5, zorder=6,
                                markeredgecolor="#ffffff", markeredgewidth=0.8)

                # Vertical cursor at the currently revealed timestep.
                if k < n_pred_steps and k >= 1:
                    ax.axvline(x=pred_t_slice[-1], color="#ffffff",
                               linewidth=0.6, alpha=0.25, linestyle=":")

        # Subtle horizontal zero line for reference.
        ax.axhline(y=0, color="#5a6488", linewidth=0.6, alpha=0.5,
                   linestyle="-")

        # Legend below the plot, two rows, so it never overlaps the data.
        ax.legend(
            fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.42),
            facecolor="#0d1226", edgecolor="#3a4060",
            labelcolor="#dde2f0", framealpha=0.95,
            ncol=8, borderaxespad=0.2, handlelength=2.2, columnspacing=1.0,
        )

        ax.set_xlabel("Timestep", fontsize=8.5)
        ax.set_ylabel("Value", fontsize=8.5)
        ax.set_title(f"{phase_label}   GT (solid)  /  Pred (dashed)",
                     fontsize=9.5, color="#ffffff", pad=6)
        ax.tick_params(labelsize=7.5, length=3, width=0.7)
        ax.grid(True, linewidth=0.5)

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

        fig.tight_layout(pad=0.6, rect=(0, 0.08, 1, 1))

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()
        buf = buf.reshape((h, w, 4))[:, :, :3]
        plt.close(fig)

    cv2 = _import_cv2()
    buf = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    buf = cv2.resize(buf, (width, height))
    return buf


def add_divider(top_img: np.ndarray, bottom_img: np.ndarray,
                divider_height: int = 6) -> np.ndarray:
    """Stack two BGR images with a thin gradient divider in between."""
    h_top, w, _ = top_img.shape
    h_bot = bottom_img.shape[0]
    divider = np.zeros((divider_height, w, 3), dtype=np.uint8)
    # Subtle teal accent strip.
    for y in range(divider_height):
        t = abs(y - divider_height / 2) / max(divider_height / 2, 1)
        intensity = int(180 * (1 - t))
        divider[y, :, 0] = max(intensity - 40, 12)
        divider[y, :, 1] = max(intensity - 20, 18)
        divider[y, :, 2] = max(intensity, 36)
    return np.vstack([top_img, divider, bottom_img])


# ---------------------------------------------------------------------------
# Image normalisation helpers
# ---------------------------------------------------------------------------
def unnormalise_image(tensor: torch.Tensor, mean: Tuple, std: Tuple) -> np.ndarray:
    """Convert a normalised [3, H, W] tensor to uint8 BGR for OpenCV."""
    cv2 = _import_cv2()
    img = tensor.clone().cpu().float()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def tensor_to_display_image(t: torch.Tensor) -> np.ndarray:
    """
    Convert an arbitrary [3, H, W] float tensor (possibly raw DiT latent) to a
    displayable BGR uint8 image by per-channel min-max normalisation.

    Used for visualising noisy intermediate diffusion steps where the values
    don't fit the ImageNet (mean, std) re-normalisation range.
    """
    cv2 = _import_cv2()
    t = t.detach().cpu().float()
    t_min = t.amin(dim=(1, 2), keepdim=True)
    t_max = t.amax(dim=(1, 2), keepdim=True)
    t = (t - t_min) / (t_max - t_min + 1e-6)
    img = t.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


# ---------------------------------------------------------------------------
# DiT diffusion intermediates: re-run the Euler integration but capture the
# intermediate ``x`` after each step so we can render a noise -> image GIF.
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_dit_with_intermediates(
    predictor,
    text_emb: torch.Tensor,
    height: int,
    width: int,
    num_frames: int = 1,
    num_steps: int = 30,
    num_snapshots: int = 24,
    world_video_query_hiddens: Optional[torch.Tensor] = None,
):
    """
    Mirror of ``RGBFramePredictor.predict_future_frames`` that records
    intermediate ``x`` values during Euler integration.

    Returns:
        snapshots: list of [B, num_frames, C, H, W] tensors (CPU), one per
                   captured step (always includes the initial noise and the
                   final sample).
        step_times: list[float] giving the t in [0, 1] each snapshot represents.
    """
    num_frames = min(num_frames, predictor.num_predict_frames)
    B = text_emb.size(0)
    rgb_channels = getattr(predictor.config, "rgb_channels", 3)

    cond_all = predictor._create_all_frame_contexts(
        text_emb, num_frames, world_video_query_hiddens,
    )  # [B*F, D]

    n = cond_all.size(0)
    device = cond_all.device
    x = torch.randn(n, rgb_channels, height, width, device=device)
    dt = 1.0 / num_steps

    # Choose which Euler steps to snapshot. Always include the start (noise)
    # and the final step, and spread the rest evenly across [0, num_steps].
    if num_snapshots >= num_steps + 1:
        keep = set(range(num_steps + 1))
    else:
        idxs = np.linspace(0, num_steps, num_snapshots).round().astype(int)
        keep = set(int(i) for i in idxs)

    snapshots: List[torch.Tensor] = []
    step_times: List[float] = []

    def _record(step_idx: int, x_now: torch.Tensor):
        snap = x_now.view(B, num_frames, rgb_channels, height, width).cpu()
        snapshots.append(snap)
        step_times.append(step_idx * dt)

    if 0 in keep:
        _record(0, x)

    for i in range(num_steps):
        t_val = i * dt
        t = torch.full((n,), t_val, device=device)
        v = predictor.dit_rgb(x, t, cond_all)
        x = x + v * dt
        if (i + 1) in keep:
            _record(i + 1, x)

    return snapshots, step_times


def save_diffusion_gif(
    snapshots: List[torch.Tensor],
    step_times: List[float],
    output_path: str,
    frame_idx: int = 0,
    sample_idx: int = 0,
    canvas_w: int = 384,
    canvas_h: int = 384,
    fps: int = 12,
    hold_final_seconds: float = 1.5,
):
    """
    Render the noise -> image diffusion trajectory for a single predicted
    frame as an animated GIF. Each GIF frame is annotated with the step
    progress and the diffusion time t.
    """
    cv2 = _import_cv2()
    from PIL import Image as PILImage

    if not snapshots:
        return

    n_steps = len(snapshots)
    pil_frames = []
    durations = []
    base_dur = int(1000 / fps)

    for i, (snap, t_val) in enumerate(zip(snapshots, step_times)):
        # snap: [B, F, C, H, W]
        img_t = snap[sample_idx, frame_idx]  # [C, H, W]
        img = tensor_to_display_image(img_t)
        img = cv2.resize(img, (canvas_w, canvas_h), interpolation=cv2.INTER_CUBIC)

        progress = (i + 1) / n_steps
        sub = f"step {i + 1}/{n_steps}    t = {t_val:.2f}"
        panel_lines = [
            ("DiT velocity-field denoising", (200, 215, 255), False),
            (f"Progress: {int(progress * 100):3d}%", (255, 255, 255), True),
        ]
        # Header colour shifts from cool (noise) to warm (data) so the user
        # can feel the progress at a glance.
        accent = (
            int(96 + (255 - 96) * progress),
            int(200 - (200 - 170) * progress),
            int(255 - (255 - 100) * progress),
        )
        composed = compose_frame(
            img, phase_label="NOISE -> IMAGE",
            sub_label=sub, panel_lines=panel_lines,
            accent=accent,
        )

        rgb = composed[:, :, ::-1].copy()
        pil_frames.append(PILImage.fromarray(rgb))
        durations.append(base_dur)

    # Hold the final frame for an extra moment so the result is readable.
    if pil_frames:
        hold = int(max(hold_final_seconds, 0) * 1000)
        if hold > 0:
            durations[-1] = durations[-1] + hold

        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=durations,
            loop=0,
        )
        logger.info("Saved diffusion GIF -> {}", output_path)


# ---------------------------------------------------------------------------
# Main video writer
# ---------------------------------------------------------------------------
def create_video(
    frames: List[np.ndarray],
    question: str,
    gt_answer: str,
    pred_answer: Optional[str],
    gt_actions: Optional[np.ndarray],
    pred_actions: Optional[np.ndarray],
    output_path: str,
    fps: int = 12,
    frame_hold: float = 3.0,
    save_gif: bool = True,
):
    """
    Write a polished visualisation for a single sample.

    PHASE 1 (CONTEXT) — each input frame is shown for ``frame_hold`` seconds
        so it's easy to read. Header shows phase + frame counter, bottom
        panel shows the question and GT answer.

    PHASE 2 (ANSWER) — model prediction streams in word-by-word with a typing
        cursor. A correctness badge appears in the top-right.

    PHASE 3 (ACTION) — animated GT vs predicted action trajectory chart, with
        a moving cursor at the currently revealed step.
    """
    cv2 = _import_cv2()

    if not frames:
        logger.warning("No frames to write for {}", output_path)
        return

    H, W = frames[0].shape[:2]

    has_actions = gt_actions is not None and gt_actions.size > 0
    gt_act_traj, pred_act_traj = None, None
    if has_actions:
        gt_act_traj = gt_actions if gt_actions.ndim >= 2 else gt_actions[np.newaxis, :]
        if pred_actions is not None and pred_actions.size > 0:
            pred_act_traj = pred_actions if pred_actions.ndim >= 2 else pred_actions[np.newaxis, :]
            min_d = min(gt_act_traj.shape[-1], pred_act_traj.shape[-1])
            gt_act_traj = gt_act_traj[:, :min_d]
            pred_act_traj = pred_act_traj[:, :min_d]

    chart_h = 260 if has_actions else 0
    divider_h = 6 if has_actions else 0
    total_H = H + divider_h + chart_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, total_H))
    if not writer.isOpened():
        logger.error("Cannot open video writer for {}", output_path)
        return

    is_correct = None
    if pred_answer is not None:
        is_correct = simple_match(gt_answer, pred_answer)

    composed_frames: List[np.ndarray] = []
    frame_durations_ms: List[int] = []

    def _stack(canvas: np.ndarray, reveal_up_to: Optional[int] = None,
               phase_label: str = "Action Trajectory") -> np.ndarray:
        """Append the action chart underneath ``canvas`` if actions are present."""
        if chart_h <= 0:
            return canvas
        if gt_act_traj is None:
            blank = np.zeros((chart_h, W, 3), dtype=np.uint8)
            return add_divider(canvas, blank, divider_h)
        chart = action_comparison_image(
            pred_act_traj, gt_act_traj,
            width=W, height=chart_h,
            reveal_up_to=reveal_up_to,
            phase_label=phase_label,
        )
        return add_divider(canvas, chart, divider_h)

    def _write(frame_bgr: np.ndarray, copies: int = 1, per_frame_ms: int = 80):
        """Write frame to MP4 (with hold copies) and collect for GIF."""
        for _ in range(max(copies, 1)):
            writer.write(frame_bgr)
        composed_frames.append(frame_bgr.copy())
        frame_durations_ms.append(per_frame_ms)

    # ==================================================================
    # PHASE 1 — Context-frame slideshow (slow, easy to read)
    # ==================================================================
    n_frames = len(frames)
    hold_copies = max(int(frame_hold * fps), 1)
    hold_ms = int(frame_hold * 1000)

    for idx, raw_frame in enumerate(frames):
        sub = f"frame {idx + 1} / {n_frames}"
        panel_lines: List[Tuple[str, Tuple[int, int, int], bool]] = []
        panel_lines.append(("Question", (160, 200, 255), True))
        panel_lines.append((question, (245, 245, 250), False))
        panel_lines.append(("Ground Truth", (255, 210, 120), True))
        panel_lines.append((gt_answer, (250, 230, 180), False))

        canvas = compose_frame(
            raw_frame, phase_label="CONTEXT", sub_label=sub,
            panel_lines=panel_lines, accent=(96, 200, 255),
        )
        combined = _stack(canvas, reveal_up_to=0,
                          phase_label="Action (GT preview)")
        _write(combined, copies=hold_copies, per_frame_ms=hold_ms)

    # ==================================================================
    # PHASE 2 — Token-by-token prediction reveal
    # ==================================================================
    last_frame = frames[-1].copy()
    if pred_answer is not None:
        pred_words = pred_answer.split() or [""]
        pred_colour = (130, 235, 170) if is_correct else (235, 130, 150)
        token_ms = int(1000 / max(fps, 1)) * 2  # readable typing speed

        for n_words in range(1, len(pred_words) + 1):
            partial = " ".join(pred_words[:n_words])
            cursor = "|" if n_words < len(pred_words) else ""

            panel_lines = []
            panel_lines.append(("Question", (160, 200, 255), True))
            panel_lines.append((question, (245, 245, 250), False))
            panel_lines.append(("Ground Truth", (255, 210, 120), True))
            panel_lines.append((gt_answer, (250, 230, 180), False))
            panel_lines.append(("Prediction", pred_colour, True))
            panel_lines.append((f"{partial}{cursor}", pred_colour, False))

            sub = f"generating ({n_words}/{len(pred_words)})"
            canvas = compose_frame(
                last_frame, phase_label="ANSWER", sub_label=sub,
                panel_lines=panel_lines, is_correct=is_correct,
                accent=(150, 220, 255),
            )
            combined = _stack(canvas, reveal_up_to=0,
                              phase_label="Action (GT preview)")
            _write(combined, copies=max(token_ms * fps // 1000, 1),
                   per_frame_ms=token_ms)

        # Hold the completed prediction briefly so the reader can absorb it.
        _write(combined, copies=max(fps, 1), per_frame_ms=900)

    # ==================================================================
    # PHASE 3 — Animated action trajectory (predicted line drawn step-by-step)
    # ==================================================================
    if has_actions and gt_act_traj is not None:
        n_pred_steps = pred_act_traj.shape[0] if pred_act_traj is not None else 0

        panel_lines = []
        panel_lines.append(("Question", (160, 200, 255), True))
        panel_lines.append((question, (245, 245, 250), False))
        panel_lines.append(("Ground Truth", (255, 210, 120), True))
        panel_lines.append((gt_answer, (250, 230, 180), False))
        if pred_answer is not None:
            pred_colour = (130, 235, 170) if is_correct else (235, 130, 150)
            panel_lines.append(("Prediction", pred_colour, True))
            panel_lines.append((pred_answer, pred_colour, False))

        n_anim = max(n_pred_steps, 1)
        anim_ms = int(1000 / fps)
        for reveal in range(0, n_anim + 1):
            canvas = compose_frame(
                last_frame, phase_label="ACTION",
                sub_label=f"step {reveal} / {n_anim}",
                panel_lines=panel_lines, is_correct=is_correct,
                accent=(255, 180, 110),
            )
            combined = _stack(
                canvas,
                reveal_up_to=reveal if pred_act_traj is not None else None,
                phase_label="Action Trajectory",
            )
            _write(combined, copies=1, per_frame_ms=anim_ms)

        # Hold the final chart.
        canvas = compose_frame(
            last_frame, phase_label="ACTION",
            sub_label=f"complete ({n_anim}/{n_anim})",
            panel_lines=panel_lines, is_correct=is_correct,
            accent=(255, 180, 110),
        )
        final_combined = _stack(canvas, reveal_up_to=n_anim,
                                phase_label="Action Trajectory")
        _write(final_combined, copies=max(fps * 2, 1), per_frame_ms=1200)

    writer.release()

    if save_gif:
        from PIL import Image as PILImage

        gif_path = output_path.rsplit(".", 1)[0] + ".gif"
        pil_frames = []
        for f in composed_frames:
            rgb = f[:, :, ::-1].copy()
            pil_frames.append(PILImage.fromarray(rgb))

        if pil_frames:
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=frame_durations_ms,
                loop=0,
            )
            logger.info("Saved GIF      -> {}", gif_path)

    logger.info("Saved video    -> {}  ({}x{})", output_path, W, total_H)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------
def main(args):
    cv2 = _import_cv2()
    device = torch.device(args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    model, config = None, None
    if args.checkpoint:
        model, config = load_model(args.checkpoint, device)

    hf_ds = dataset.dataset
    n_total = len(hf_ds)
    written = 0

    logger.info(
        "Scanning {} samples for >= {} frames (max {} videos) ...",
        n_total, args.min_frames, args.num_samples,
    )

    for sample_idx in range(n_total):
        if written >= args.num_samples:
            break

        sample = hf_ds[sample_idx]

        raw_imgs = sample.get("image")
        if raw_imgs is None:
            continue
        if not isinstance(raw_imgs, (list, tuple)):
            raw_imgs = [raw_imgs]
        n_frames = len(raw_imgs)
        if n_frames < args.min_frames:
            continue

        question, gt_answer = extract_question_answer(sample)
        if not question and not gt_answer:
            continue

        images_tensor = dataset._process_images(sample)
        N_img = images_tensor.size(0)

        bgr_frames = [
            unnormalise_image(images_tensor[i], ds_cfg.img_mean, ds_cfg.img_std)
            for i in range(N_img)
        ]

        canvas_h, canvas_w = args.canvas_h, args.canvas_w
        bgr_frames = [cv2.resize(f, (canvas_w, canvas_h)) for f in bgr_frames]

        gt_actions_t, action_mask_t = dataset._process_actions(sample)
        gt_actions = gt_actions_t.numpy() if gt_actions_t.numel() > 0 else None

        pred_answer = None
        pred_actions = None
        diffusion_snapshots = None
        diffusion_times = None

        if model is not None:
            input_ids, attn_mask, _ = dataset._process_conversation(sample, pair_idx=0)
            states_t, _ = dataset._process_states(sample)

            imgs_in = images_tensor.unsqueeze(0).to(device)
            ids_in = input_ids.unsqueeze(0).to(device)
            mask_in = attn_mask.unsqueeze(0).to(device)
            if states_t.numel() > 0:
                states_in = states_t.unsqueeze(0).to(device)
            else:
                states_in = torch.zeros(1, 1, config.state_dim, device=device)

            asst_prefix = "<|im_start|>assistant\n"
            full_text = tokenizer.decode(ids_in[0].cpu(), skip_special_tokens=False)
            cut = full_text.rfind(asst_prefix)
            if cut != -1:
                prompt_text = full_text[: cut + len(asst_prefix)]
                enc = tokenizer(prompt_text, return_tensors="pt",
                                add_special_tokens=False)
                ids_in = enc["input_ids"].to(device)
                mask_in = enc["attention_mask"].to(device)

            pred_text, act_preds = generate_text(
                model, tokenizer, imgs_in, ids_in, mask_in, states_in,
                max_new_tokens=args.max_new_tokens, device=device,
            )
            pred_answer = pred_text.strip()

            if act_preds is not None:
                pred_actions = act_preds[0].view(-1, act_preds.size(-1)).cpu().numpy()

            # --- Diffusion intermediates for the noise -> image GIF ---
            visual_predictor = getattr(model, "visual_predictor", None)
            if visual_predictor is not None:
                try:
                    _, _, vce, wv = model(
                        images=imgs_in, input_ids=ids_in,
                        attention_mask=mask_in, states=states_in,
                    )
                    if vce is not None:
                        n_predict = 1
                        if wv is not None:
                            n_predict = min(wv.size(1), visual_predictor.num_predict_frames)
                            n_predict = max(n_predict, 1)
                        # DiT operates at the same H/W as input frames.
                        h_in, w_in = imgs_in.shape[-2], imgs_in.shape[-1]
                        diffusion_snapshots, diffusion_times = sample_dit_with_intermediates(
                            visual_predictor,
                            text_emb=vce,
                            height=h_in, width=w_in,
                            num_frames=n_predict,
                            num_steps=args.diffusion_steps,
                            num_snapshots=args.diffusion_snapshots,
                            world_video_query_hiddens=wv,
                        )
                except Exception as e:
                    logger.warning("Diffusion intermediates failed: {}", e)

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

        if diffusion_snapshots is not None and len(diffusion_snapshots) > 0:
            gif_path = str(out_dir / f"sample_{sample_idx:05d}_diffusion.gif")
            save_diffusion_gif(
                diffusion_snapshots, diffusion_times,
                output_path=gif_path,
                frame_idx=0,
                sample_idx=0,
                canvas_w=args.canvas_w,
                canvas_h=args.canvas_h,
                fps=args.diffusion_fps,
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
    p.add_argument("--fps", type=int, default=12, help="Video FPS")
    p.add_argument("--frame_hold", type=float, default=3.0,
                   help="Seconds each context frame is held on screen")
    p.add_argument("--canvas_w", type=int, default=728,
                   help="Video canvas width")
    p.add_argument("--canvas_h", type=int, default=504,
                   help="Video canvas height")

    # Diffusion GIF (noise -> image)
    p.add_argument("--diffusion_steps", type=int, default=30,
                   help="Number of DiT Euler integration steps")
    p.add_argument("--diffusion_snapshots", type=int, default=24,
                   help="Frames captured for the noise->image GIF")
    p.add_argument("--diffusion_fps", type=int, default=12,
                   help="FPS for the noise->image GIF")

    # Device
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
