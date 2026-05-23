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
from torch.cuda.amp import autocast, GradScaler
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
    create_video,
    unnormalise_image,
    generate_text,
)

from loguru import logger

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]


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
    # Add question as title
    cv2.putText(frame, f"Q: {question[:50]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"GT: {gt_answer[:30]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    if pred_answer:
        cv2.putText(frame, f"Pred: {pred_answer[:30]}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    # Optionally plot action curves (requires drawing on a small region)
    return frame


# ---------------------------------------------------------------------------
# Visualization helper (called during training every N steps)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_training_videos_gif(
    model,
    batch,
    tokenizer,
    config,
    device,
    output_dir: Path,
    global_step: int,
    epoch: int,
    max_videos: int = 4,
    min_frames: int = 3,
    canvas_w: int = 640,
    canvas_h: int = 480,
    fps: int = 2,
    frame_hold: int = 2,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
):
    """
    Generate visualisation videos for batch samples that have >= min_frames.

    For each qualifying sample in the batch:
      - Runs greedy generation to get predicted text + actions
      - Creates an MP4 with frame slideshow, Q/A overlay, action chart
    """
    import cv2

    model.eval()
    vis_dir = output_dir / f"vis_epoch{epoch}_step{global_step}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    images = batch["images"].to(device)               # [B, N, 3, H, W]
    input_ids = batch["input_ids"].to(device)           # [B, seq_len]
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    actions = batch["actions"]                          # [B, T, act_dim]
    action_mask_b = batch["action_mask"]                # [B, T]
    states = batch["states"].to(device)                 # [B, S, state_dim]
    image_mask = batch.get("image_mask")                # [B, N] or None

    B = images.size(0)
    written = 0

    for b in range(B):
        if written >= max_videos:
            break

        # Count real images for this sample
        if image_mask is not None:
            n_imgs = int(image_mask[b].sum().item())
        else:
            n_imgs = images.size(1)

        if n_imgs < min_frames:
            continue

        # --- Decode GT question / answer from input_ids + labels ---
        ids_b = input_ids[b].cpu()
        full_decoded = tokenizer.decode(ids_b, skip_special_tokens=False)

        # Extract question (user turn) and GT answer (assistant turn)
        question, gt_answer = "", ""
        if "<|im_start|>user" in full_decoded:
            user_block = full_decoded.split("<|im_start|>user")[-1]
            if "<|im_end|>" in user_block:
                question = user_block.split("<|im_end|>")[0].strip()
        if "<|im_start|>assistant" in full_decoded:
            asst_block = full_decoded.split("<|im_start|>assistant")[-1]
            if "<|im_end|>" in asst_block:
                gt_answer = asst_block.split("<|im_end|>")[0].strip()
            else:
                gt_answer = asst_block.strip()

        # Clean up special tokens from question text
        for tok in ["<image>", "<state>", "<halo_action>"]:
            question = question.replace(tok, "").strip()

        # --- Convert images to BGR frames ---
        bgr_frames = []
        for i in range(n_imgs):
            frame = unnormalise_image(images[b, i].cpu(), img_mean, img_std)
            frame = cv2.resize(frame, (canvas_w, canvas_h))
            bgr_frames.append(frame)

        # --- GT actions ---
        gt_act = actions[b].cpu().numpy()
        amask = action_mask_b[b].cpu().numpy()
        valid_t = int(amask.sum())
        gt_act_valid = gt_act[:valid_t] if valid_t > 0 else None

        # --- Model prediction ---
        pred_answer = None
        pred_actions_np = None
        try:
            imgs_in = images[b:b+1]                      # [1, N, 3, H, W]
            ids_in = input_ids[b:b+1]                    # [1, seq_len]
            mask_in = attention_mask[b:b+1]
            states_in = states[b:b+1]                    # [1, S, state_dim]

            pred_text, act_preds = generate_text(
                model, tokenizer, imgs_in, ids_in, mask_in, states_in,
                max_new_tokens=128, device=device,
            )
            # Extract generated portion
            if "assistant" in pred_text:
                pred_answer = pred_text.split("assistant")[-1].strip()
            else:
                pred_answer = pred_text.strip()

            if act_preds is not None:
                pred_actions_np = (
                    act_preds[0, 0].cpu().numpy()
                )
        except Exception as e:
            logger.warning("Vis generation failed for sample {}: {}", b, e)
        # ---- Save as GIF instead of MP4 ----
        import imageio

        gif_path = vis_dir / f"sample_{b}_vla_step{global_step}.gif"
        # Convert BGR frames to RGB and repeat each frame according to frame_hold
        rgb_frames = []
        for frame in bgr_frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for _ in range(frame_hold):
                rgb_frames.append(rgb)
        # Write GIF (loop forever)
        imageio.mimsave(str(gif_path), rgb_frames, fps=fps, loop=0)

        written += 1
        # --- Create video ---
        video_path = str(vis_dir / f"sample_{b}_vla_step{global_step}.mp4")
        create_video(
            frames=bgr_frames,
            question=question,
            gt_answer=gt_answer,
            pred_answer=pred_answer,
            gt_actions=gt_act_valid,
            pred_actions=pred_actions_np,
            output_path=video_path,
            fps=fps,
            frame_hold=frame_hold,
        )

    model.train()
    logger.info(
        "Generated {} visualisation gif at step {} → {}",
        written, global_step, vis_dir,
    )

# -----------------------
@torch.no_grad()
def generate_training_videos_with_pred(
    model, batch, tokenizer, config, device, output_dir, global_step, epoch,
    max_videos=4, min_frames=3, canvas_w=640, canvas_h=480,
    fps=2, frame_hold=2, img_mean=(0.485,0.456,0.406), img_std=(0.229,0.224,0.225),
):
    import cv2
    import numpy as np

    model.eval()
    vis_dir = output_dir / f"vis_epoch{epoch}_step{global_step}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    images = batch["images"].to(device)          # [B, N, 3, H, W]
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    states = batch["states"].to(device)
    image_mask = batch.get("image_mask")
    img_h, img_w = images.shape[-2:]

    B = images.size(0)
    written = 0

    for b in range(B):
        if written >= max_videos:
            break
        if image_mask is not None and int(image_mask[b].sum().item()) < min_frames:
            continue

        # ---- Run model to get conditioning ----
        imgs_in = images[b:b+1]
        ids_in = input_ids[b:b+1]
        mask_in = attention_mask[b:b+1]
        states_in = states[b:b+1]
        img_mask_in = image_mask[b:b+1] if image_mask is not None else None

        _, _, visual_context_emb = model(
            images=imgs_in,
            input_ids=ids_in,
            attention_mask=mask_in,
            states=states_in,
            image_mask=img_mask_in,
        )   # visual_context_emb: [1, L, D]

        # ---- Generate predicted future RGB frame ----
        if hasattr(model, 'visual_predictor'):
            # The predictor expects conditioning of shape [B, emb_dim]
            # We pool over the sequence length
            cond = visual_context_emb   # [1, 512] – already pooled
            pred_frame = model.visual_predictor.predict_future_rgb(
                past_rgb=imgs_in,
                height=img_h,
                width=img_w,
                context_emb=cond,
            )  # shape: [1, 3, H, W]
        else:
            logger.warning("model has no visual_predictor; skipping generation")
            continue

        mean_t = torch.tensor(img_mean, device=pred_frame.device).view(1, 3, 1, 1)
        std_t = torch.tensor(img_std, device=pred_frame.device).view(1, 3, 1, 1)
        pred_denorm = pred_frame * std_t + mean_t
        pred_denorm = torch.clamp(pred_denorm, 0, 1)

        # Convert to BGR for OpenCV
        pred_np = pred_denorm[0].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
        pred_np = (pred_np * 255).clip(0, 255).astype(np.uint8)
        pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)
        pred_bgr = cv2.resize(pred_bgr, (canvas_w, canvas_h))

        import imageio

        gif_path = vis_dir / f"sample_{b}_pred_step{global_step}.gif"
        # Convert BGR frame to RGB for GIF
        frame_rgb = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB)
        # Create a list repeating the same frame to form an animation
        frames_rgb = [frame_rgb] * (frame_hold * 10)
        imageio.mimsave(str(gif_path), frames_rgb, fps=fps, loop=0)

        written += 1

        # # ---- Write a short video (loop the single frame for visibility) ----
        # pred_video_path = str(vis_dir / f"sample_{b}_pred_step{global_step}.mp4")
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out_vid = cv2.VideoWriter(pred_video_path, fourcc, fps, (canvas_w, canvas_h))
        # for _ in range(frame_hold * 10):   # repeat frame to make a short clip
        #     out_vid.write(pred_bgr)
        # out_vid.release()

        # written += 1

    model.train()
    logger.info("Saved predicted videos for {} samples in {}", written, vis_dir)
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
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    scaler = GradScaler() 

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
            with autocast():
                logits, action_hiddens, visual_context_emb = model(
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
                vis_loss = model.compute_visual_prediction_loss(
                    images, batch.get("image_mask"), visual_context_emb
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

            # Visualisation videos
            if args.vis_every > 0 and global_step % args.vis_every == 0:
                generate_training_videos_gif(
                    model=model, batch=batch, tokenizer=tokenizer, config=config,
                    device=device, output_dir=Path(args.ckpt_dir), global_step=global_step,
                    epoch=epoch, max_videos=args.vis_max_videos, min_frames=args.vis_min_frames,
                )
                generate_training_videos_with_pred(
                    model=model,
                    batch=batch,
                    tokenizer=tokenizer,
                    config=config,
                    device=device,
                    output_dir=ckpt_dir,
                    global_step=global_step,
                    epoch=epoch,
                    max_videos=args.vis_max_videos,
                    min_frames=args.vis_min_frames,
                )
                


        # End of epoch summary
        n = len(train_loader)
        elapsed = time.time() - t0
        logger.info(
            "=== Epoch {} done in {:.1f}s | avg lang={:.4f}  act={:.4f}  vis={:.4f}  total={:.4f} ===",
            epoch, elapsed,
            epoch_lang_loss / n, epoch_act_loss / n, epoch_vis_loss / n, epoch_total_loss / n,
        )

        if tb_writer is not None:
            tb_writer.add_scalar("epoch_avg/lang", epoch_lang_loss / n, epoch)
            tb_writer.add_scalar("epoch_avg/action_flow", epoch_act_loss / n, epoch)
            tb_writer.add_scalar("epoch_avg/visual_dit", epoch_vis_loss / n, epoch)
            tb_writer.add_scalar("epoch_avg/total", epoch_total_loss / n, epoch)

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = ckpt_dir / f"halo_vla_epoch{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": config,
                },
                ckpt_path,
            )
            logger.info("Saved checkpoint → {}", ckpt_path)

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
    p.add_argument("--moma_camera", default="hand", choices=("hand", "head"))
    p.add_argument(
        "--moma_num_frames",
        type=int,
        default=3,
        help="Temporal frames per sample (>=2 enables future-frame DiT loss)",
    )
    p.add_argument("--moma_frame_stride", type=int, default=2)
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
    p.add_argument("--save_every", type=int, default=1)
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
