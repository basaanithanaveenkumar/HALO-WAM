"""
PyTorch dataset for `airoa-org/airoa-moma` (HSR teleoperation, RGB video + metadata).

Designed for Halo-VLA training: **language + flow actions + multi-frame visuals**
(including future-frame / DiT losses when ``num_sample_frames >= 2``).

The public Hugging Face repo is gated and video-heavy; this loader targets a **local
clone** after ``git lfs pull`` (see the dataset card). Expected layout::

    <data_root>/
      episodes.jsonl
      videos/chunk-XXX/observation.image.hand/episode_YYYYYY.mp4
      videos/chunk-XXX/observation.image.head/episode_YYYYYY.mp4

Optional proprio (if you export arrays next to the clone)::

    <data_root>/states/episode_YYYYYY.npy   # float32 [T, state_dim_raw]

If ``states/*.npy`` is missing, states/actions are padded zeros with masks so the
language + vision branches still train; add exports to enable full action supervision.

Dataset card: https://huggingface.co/datasets/airoa-org/airoa-moma
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from loguru import logger

from config.tokens import get_all_custom_tokens
from dataloader.eo_dataset import build_image_transform, eo_collate_fn


@dataclass
class AiroaMomaConfig:
    """Hyper-parameters for MoMa → Halo-VLA batches."""

    data_root: str  # directory containing episodes.jsonl and videos/

    tokenizer_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    img_size: int = 224
    img_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    img_std: Tuple[float, ...] = (0.229, 0.224, 0.225)

    max_seq_len: int = 512
    max_action_len: int = 256  # subsample / truncate trajectory steps
    action_dim: int = 32
    state_dim: int = 32

    # Video sampling (temporal context + “future” = last frame in the stack)
    camera: str = "hand"  # "hand" | "head"
    num_sample_frames: int = 8
    frame_stride: int = 4  # consecutive windows: t, t+stride, ...
    min_episode_length: int = 32  # skip shorter episodes (metadata or probe)

    # Path templates relative to data_root
    episodes_jsonl: str = "episodes.jsonl"
    # chunk index from episode_index (dataset uses chunk-000, chunk-001, …)
    episodes_per_chunk: int = 1000
    video_rel_template: str = (
        "videos/observation.image.{camera}/chunk-{chunk:03d}/file-{file_num:03d}.mp4"
    )
    state_npy_subdir: str = "states"  # optional episode_XXXXXX.npy

    # Conversation template (ChatML-style, matches EO dataloader)
    system_prefix: str = "<|im_start|>system\n"
    human_prefix: str = "<|im_start|>user\n"
    assistant_prefix: str = "<|im_start|>assistant\n"
    turn_end: str = "<|im_end|>\n"

    max_samples: Optional[int] = None
    seed: int = 42


class AiroaMomaDataset(Dataset):
    """
    One training item = one **window** of frames from one episode + task text + state/action slice.

    ``images`` are ordered in time; the **last** image is the “future” target for the
    visual DiT head when ``num_sample_frames >= 2`` (matches ``compute_visual_prediction_loss``).
    """

    def __init__(
        self,
        config: Optional[AiroaMomaConfig] = None,
        tokenizer=None,
        split: str = "train",
        **kwargs,
    ):
        super().__init__()
        if config is None:
            config = AiroaMomaConfig(**kwargs)
        self.cfg = config
        self.split = split
        self.root = Path(config.data_root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"data_root is not a directory: {self.root}")

        self._setup_tokenizer(tokenizer)
        self.img_transform = build_image_transform(
            self.cfg.img_size, self.cfg.img_mean, self.cfg.img_std
        )

        jsonl_path = self.root / self.cfg.episodes_jsonl
        if not jsonl_path.is_file():
            raise FileNotFoundError(
                f"Missing {jsonl_path}. Clone the dataset and ensure episodes.jsonl exists.\n"
                "See https://huggingface.co/datasets/airoa-org/airoa-moma"
            )
        raw = self._load_episodes_jsonl(jsonl_path)
        self.episodes = self._filter_episodes(raw)
        if self.cfg.max_samples is not None:
            self.episodes = self.episodes[: self.cfg.max_samples]
        self._rng = random.Random(self.cfg.seed)

        logger.info(
            "AiroaMomaDataset: {} episodes under {} (camera={}, frames={}, stride={})",
            len(self.episodes),
            self.root,
            self.cfg.camera,
            self.cfg.num_sample_frames,
            self.cfg.frame_stride,
        )

    def _load_episodes_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        rows.sort(key=lambda r: int(r.get("episode_index", 0)))
        return rows

    def _filter_episodes(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Remove the video file and length checks; keep all episodes.
        ok = []
        for m in rows:
            ep = int(m.get("episode_index", -1))
            if ep < 0: continue
            vid = self._video_path(ep)
            if vid.is_file():
                ok.append(m)
        if not ok:
            logger.warning("No episodes found in jsonl")
        else:
            logger.info(f"Using all {len(ok)} episodes (video checks disabled)")
        return ok

    def _setup_tokenizer(self, tokenizer):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.tokenizer_name,
                trust_remote_code=True,
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        additional_special: List[str] = []
        for tok in get_all_custom_tokens():
            if tok not in self.tokenizer.get_vocab():
                additional_special.append(tok)
        if additional_special:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": additional_special}
            )

        from config.tokens import get_token

        self._image_tok = get_token("image_token")
        self._action_tok = get_token("action_token")
        self._state_tok = get_token("state_token")
        self.pad_token_id = self.tokenizer.pad_token_id

    def _video_path(self, episode_index: int) -> Path:
        chunk = int(episode_index) // self.cfg.episodes_per_chunk
        file_num = int(episode_index) % self.cfg.episodes_per_chunk   # 0‑999 per chunk
        rel = self.cfg.video_rel_template.format(
            chunk=chunk,
            camera=self.cfg.camera,
            ep=int(episode_index),
            file_num=file_num,          # add this line
        )
        return self.root / rel

    # def _probe_num_frames(self, video_path: Path) -> int:
    #     cv2 = _import_cv2()
    #     cap = cv2.VideoCapture(str(video_path))
    #     if not cap.isOpened():
    #         return 0
    #     n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     cap.release()
    #     return max(n, 0)
    def _get_video_frame_count(self, video_path: Path) -> int:
        """Return the exact number of frames in the video using PyAV."""
        import av
        try:
            container = av.open(str(video_path))
            stream = container.streams.video[0]
            # Some AV1 files may not have stream.frames set; fallback to counting
            count = stream.frames if stream.frames else 0
            if count == 0:
                # Count manually
                for packet in container.demux(stream):
                    for _ in packet.decode():
                        count += 1
            container.close()
            return count
        except Exception as e:
            logger.warning(f"Failed to get frame count for {video_path}: {e}")
            return 0
    # def _read_frames_at_indices(
    #     self, video_path: Path, indices: Sequence[int]
    # ) -> List[torch.Tensor]:
    #     """Return list of [3,H,W] float tensors (ImageNet-normalised), order = ``indices``."""
    #     from PIL import Image

    #     cv2 = _import_cv2()
    #     cap = cv2.VideoCapture(str(video_path))
    #     if not cap.isOpened():
    #         raise FileNotFoundError(f"Cannot open video: {video_path}")

    #     want = sorted(set(int(i) for i in indices))
    #     frame_map: Dict[int, torch.Tensor] = {}
    #     pos = 0
    #     wi = 0
    #     while wi < len(want):
    #         target = want[wi]
    #         while pos < target:
    #             if not cap.read()[0]:
    #                 break
    #             pos += 1
    #         if pos != target:
    #             break
    #         ret, bgr = cap.read()
    #         if not ret:
    #             break
    #         rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    #         pil = Image.fromarray(rgb)
    #         frame_map[target] = self.img_transform(pil)
    #         pos += 1
    #         wi += 1
    #     cap.release()

    #     blank = torch.zeros(3, self.cfg.img_size, self.cfg.img_size)
    #     out: List[torch.Tensor] = []
    #     for i in indices:
    #         ii = int(i)
    #         if ii in frame_map:
    #             out.append(frame_map[ii])
    #         else:
    #             logger.warning(
    #                 "Frame index {} missing in {}; using blank", ii, video_path.name
    #             )
    #             out.append(blank.clone())
    #     return out
    # Add at top of file

    # Replace _read_frames_at_indices with:
    # def _read_frames_at_indices(self, video_path, indices):
    #     from decord import VideoReader, cpu
    #     vr = VideoReader(str(video_path), ctx=cpu(0))
    #     # decord indices are 0-based; ensure they're within range
    #     valid_idx = [i for i in indices if i < len(vr)]
    #     frames = vr.get_batch(valid_idx).asnumpy()  # [T, H, W, C] (RGB)
    #     # Convert to torch tensor and apply transform
    #     transformed = []
    #     j = 0
    #     for i in indices:
    #         if i in valid_idx:
    #             pil = Image.fromarray(frames[j])
    #             transformed.append(self.img_transform(pil))
    #             j += 1
    #         else:
    #             transformed.append(torch.zeros(3, self.cfg.img_size, self.cfg.img_size))
    #     return transformed
    # def _read_frames_at_indices(self, video_path: Path, indices: Sequence[int]) -> List[torch.Tensor]:
    #     """Read specific frame indices from a video using PyAV."""
    #     import av
    #     from PIL import Image

    #     container = av.open(str(video_path))
    #     # Assume first video stream
    #     stream = container.streams.video[0]

    #     # Build a set of requested indices for fast lookup
    #     required = set(int(i) for i in indices)
    #     frame_map = {}

    #     # Decode frames sequentially (fast enough for short videos / small frame requests)
    #     for packet in container.demux(stream):
    #         for frame in packet.decode():
    #             pts = int(frame.pts)  # presentation timestamp (frame index)
    #             if pts in required:
    #                 # Convert frame to PIL RGB image
    #                 img = frame.to_image().convert('RGB')
    #                 frame_map[pts] = self.img_transform(img)
    #                 required.remove(pts)
    #                 if not required:
    #                     break
    #         if not required:
    #             break

    #     container.close()

    #     # Build output list: if a requested index wasn't found, insert a blank
    #     blank = torch.zeros(3, self.cfg.img_size, self.cfg.img_size)
    #     out = []
    #     for i in indices:
    #         ii = int(i)
    #         if ii in frame_map:
    #             out.append(frame_map[ii])
    #         else:
    #             logger.warning(f"Frame index {ii} missing in {video_path.name}; using blank")
    #             out.append(blank.clone())
    #     return out
    def _read_frames_at_indices(self, video_path: Path, indices: Sequence[int]) -> List[torch.Tensor]:
        """Read specific frame indices using PyAV with a simple counter."""
        import av
        from PIL import Image

        container = av.open(str(video_path))
        stream = container.streams.video[0]
        required = set(int(i) for i in indices)
        frame_map = {}
        frame_counter = 0

        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame_counter in required:
                    img = frame.to_image().convert('RGB')
                    frame_map[frame_counter] = self.img_transform(img)
                    required.remove(frame_counter)
                    if not required:
                        break
                frame_counter += 1
            if not required:
                break

        container.close()

        blank = torch.zeros(3, self.cfg.img_size, self.cfg.img_size)
        out = []
        for i in indices:
            ii = int(i)
            if ii in frame_map:
                out.append(frame_map[ii])
            else:
                # This warning should now be rare because indices are clamped
                logger.warning(f"Frame index {ii} missing in {video_path.name}; using blank")
                out.append(blank.clone())
        return out
    def _load_state_trajectory(self, episode_index: int) -> Optional[np.ndarray]:
        sub = self.root / self.cfg.state_npy_subdir / f"episode_{int(episode_index):06d}.npy"
        if not sub.is_file():
            return None
        arr = np.load(sub)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.astype(np.float32)

    def _build_text(self, meta: Dict[str, Any], num_images: int) -> Tuple[str, str]:
        """User string (with placeholders) and assistant string."""
        tasks = meta.get("tasks") or []
        task_txt = tasks[0] if tasks else meta.get("short_horizon_task", "manipulation")
        if not isinstance(task_txt, str):
            task_txt = str(task_txt)

        img_tags = " ".join([self._image_tok] * num_images)
        user_body = (
            f"{img_tags} {self._state_tok} Task: {task_txt}\n"
            "Predict the robot's motion and describe the next step."
        )
        assistant_body = (
            f"I will execute the task: {task_txt}\n{self._action_tok}"
        )
        return user_body, assistant_body

    def _tokenise_turns(self, user_body: str, assistant_body: str):
        from config import HaloVLMConfig

        model_cfg = HaloVLMConfig()
        system_msg = (
            f"{self.cfg.system_prefix}{model_cfg.system_prompt}{self.cfg.turn_end}"
        )
        human_msg = f"{self.cfg.human_prefix}{user_body}{self.cfg.turn_end}"
        asst_msg = f"{self.cfg.assistant_prefix}{assistant_body}{self.cfg.turn_end}"
        full_text = system_msg + human_msg + asst_msg

        encoding = self.tokenizer(
            full_text,
            max_length=self.cfg.max_seq_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # Mask system + user spans (same heuristic as EODataset)
        text_parts = [system_msg, human_msg, asst_msg]
        role_is_assistant = [False, False, True]
        labels = self._mask_non_assistant(full_text, text_parts, role_is_assistant, labels)
        labels[attention_mask == 0] = -100
        return input_ids, attention_mask, labels

    def _mask_non_assistant(
        self,
        full_text: str,
        text_parts: List[str],
        role_is_assistant: List[bool],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        cursor = 0
        for part, is_assistant in zip(text_parts, role_is_assistant):
            start_char = full_text.find(part, cursor)
            if start_char == -1:
                continue
            end_char = start_char + len(part)
            if not is_assistant:
                prefix = full_text[:end_char]
                prefix_ids = self.tokenizer(
                    prefix,
                    max_length=self.cfg.max_seq_len,
                    truncation=True,
                    add_special_tokens=False,
                )["input_ids"]
                start_prefix = full_text[:start_char]
                start_ids = self.tokenizer(
                    start_prefix,
                    max_length=self.cfg.max_seq_len,
                    truncation=True,
                    add_special_tokens=False,
                )["input_ids"]
                tok_start = len(start_ids)
                tok_end = len(prefix_ids)
                if tok_end <= labels.numel():
                    labels[tok_start:tok_end] = -100
            cursor = end_char
        return labels
    def _generate_dummy_frames(self, num_frames: int) -> torch.Tensor:
        """Return [num_frames, 3, H, W] black images."""
        shape = (num_frames, 3, self.cfg.img_size, self.cfg.img_size)
        return torch.zeros(shape, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.episodes[idx]
        ep = int(meta.get("episode_index", idx))
        vid_path = self._video_path(ep)

        T_meta = int(meta.get("length", 0) or 0)
        if vid_path.is_file():
            T_vid = self._get_video_frame_count(vid_path)
        else:
            T_vid = 0
        T_meta = int(meta.get("length", 0) or 0)
        T = T_vid if T_vid > 0 else T_meta 

        n = self.cfg.num_sample_frames
        stride = self.cfg.frame_stride
        span = (n - 1) * stride + 1
        if T < span:
            stride = max(1, (T - 1) // max(n - 1, 1))
            span = (n - 1) * stride + 1
        t0_max = max(0, T - span)
        t0 = self._rng.randint(0, t0_max) if self.split == "train" else 0
        frame_indices = [t0 + j * stride for j in range(n)]
        # Clamp indices to valid range (0 .. T-1) if video exists
        if vid_path.is_file() and T_vid > 0:
            frame_indices = [min(i, T_vid - 1) for i in frame_indices]
        if not vid_path.is_file():
            logger.warning(f"Video missing: {vid_path} – using dummy frames")
            images = self._generate_dummy_frames(n)
            frame_indices = list(range(n))   # dummy indices
        else:
            frame_tensors = self._read_frames_at_indices(vid_path, frame_indices)
            images = torch.stack(frame_tensors, dim=0)
        #frame_tensors = self._read_frames_at_indices(vid_path, frame_indices)
        #images = torch.stack(frame_tensors, dim=0)  # [N, 3, H, W]

        user_body, assistant_body = self._build_text(meta, num_images=n)
        input_ids, attention_mask, labels = self._tokenise_turns(user_body, assistant_body)

        state_traj = self._load_state_trajectory(ep)
        actions, action_mask, states_per_frame, _ = self._align_actions_states(
            state_traj, frame_indices
        )
        # One <state> token in the prompt → single pooled vector [1, state_dim]
        if states_per_frame.numel() > 0:
            states = states_per_frame.mean(dim=0, keepdim=True)
            state_mask = torch.ones(1, dtype=torch.long)
        else:
            states = torch.ones(1, self.cfg.state_dim)
            state_mask = torch.zeros(1, dtype=torch.long)

        return {
            "images": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "actions": actions,
            "action_mask": action_mask,
            "states": states,
            "state_mask": state_mask,
            "num_images": images.size(0),
            "pad_token_id": self.pad_token_id,
        }

    def _align_actions_states(
        self,
        state_traj: Optional[np.ndarray],
        frame_indices: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Align proprio to frame indices. Actions = consecutive differences in state.
        If no ``states/*.npy``, return empty tensors (masked).
        """
        n = len(frame_indices)
        D_a = self.cfg.action_dim
        D_s = self.cfg.state_dim
        if state_traj is None:
            dummy_states = torch.randn(n, D_s)
            dummy_actions = torch.randn(max(0, n-1), D_a)
            # masks = 1 means they will be used in loss
            state_mask = torch.ones(n, dtype=torch.long)
            action_mask = torch.ones(dummy_actions.shape[0], dtype=torch.long)
            # truncation
            if dummy_actions.shape[0] > self.cfg.max_action_len:
                dummy_actions = dummy_actions[:self.cfg.max_action_len]
                action_mask = action_mask[:self.cfg.max_action_len]
            return dummy_actions, action_mask, dummy_states, state_mask

        T = state_traj.shape[0]
        idx = [min(max(0, i), T - 1) for i in frame_indices]
        sel = state_traj[idx]  # [n, d_raw]
        d_raw = sel.shape[1]

        # States: one row per sampled frame, padded/truncated to state_dim
        states = np.zeros((sel.shape[0], D_s), dtype=np.float32)
        d_use = min(d_raw, D_s)
        states[:, :d_use] = sel[:, :d_use]
        state_mask = torch.ones(states.shape[0], dtype=torch.long)

        # Actions: finite differences along time within the window, length n-1
        diff = np.diff(sel, axis=0)  # [n-1, d_raw]
        if diff.size == 0:
            return (
                torch.zeros(0, D_a),
                torch.zeros(0, dtype=torch.long),
                torch.from_numpy(states),
                state_mask,
            )
        actions = np.zeros((diff.shape[0], D_a), dtype=np.float32)
        d_act = min(d_raw, D_a)
        actions[:, :d_act] = diff[:, :d_act]
        action_mask = torch.ones(actions.shape[0], dtype=torch.long)

        # Truncate to max_action_len
        max_t = self.cfg.max_action_len
        if actions.shape[0] > max_t:
            actions = actions[:max_t]
            action_mask = action_mask[:max_t]

        return (
            torch.from_numpy(actions),
            action_mask,
            torch.from_numpy(states),
            state_mask,
        )


def _import_cv2():
    try:
        import cv2
        return cv2
    except ImportError as e:
        raise ImportError("OpenCV required for MoMa video: pip install opencv-python") from e


def build_airoa_moma_dataloader(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    tokenizer=None,
    img_size: int = 224,
    max_seq_len: int = 512,
    max_action_len: int = 256,
    action_dim: int = 32,
    state_dim: int = 32,
    num_sample_frames: int = 8,
    frame_stride: int = 4,
    camera: str = "head",
    split: str = "train",
    shuffle: bool = True,
    max_samples: Optional[int] = None,
    pin_memory: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Build a DataLoader compatible with ``scripts/train.py`` (same keys as EO loader).

    Args:
        data_root: Local clone root (``episodes.jsonl`` + ``videos/``).
    """
    cfg = AiroaMomaConfig(
        data_root=data_root,
        img_size=img_size,
        max_seq_len=max_seq_len,
        max_action_len=max_action_len,
        action_dim=action_dim,
        state_dim=state_dim,
        num_sample_frames=num_sample_frames,
        frame_stride=frame_stride,
        camera=camera,
        max_samples=max_samples,
        **{k: v for k, v in kwargs.items() if k in AiroaMomaConfig.__dataclass_fields__},
    )
    ds = AiroaMomaDataset(config=cfg, tokenizer=tokenizer, split=split)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=eo_collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
    )
