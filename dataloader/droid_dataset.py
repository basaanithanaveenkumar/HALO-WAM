"""
PyTorch dataset for ``lerobot/droid_100`` (and the full DROID corpus) stored in
the **LeRobot v2 parquet + video format**.

DROID is a large-scale robot manipulation dataset collected on a Franka arm
with three cameras (two exterior, one wrist). The lerobot release stores
tabular data (actions, states, timestamps) in per-episode parquet files and
video frames in MP4 clips.

Expected local layout after ``huggingface-cli download lerobot/droid_100``::

    <data_root>/
      meta/
        info.json          # dataset-level metadata (fps, features, camera keys)
        episodes.parquet   # episode-level rows: episode_index, length, task_index
        tasks.parquet      # task_index → task string
      data/
        chunk-000/
          episode_000000.parquet   # per-frame rows: action, observation.state, …
          episode_000001.parquet
          …
      videos/
        chunk-000/
          observation.images.exterior_image_1_left/
            episode_000000.mp4
          observation.images.exterior_image_2_left/
            episode_000000.mp4
          observation.images.wrist_image_left/
            episode_000000.mp4
          …

Each training item is one **sliding window** of frames from one episode:
  - ``num_sample_frames`` context frames  → ``images`` [n, 3, H, W]
  - ``num_predict_frames`` future targets → ``future_frames`` [p, 3, H, W]
  - Per-frame actions from the parquet    → ``actions`` [T, action_dim]
  - Proprioceptive state                  → ``states``  [T, state_dim]

The output dict is fully compatible with ``eo_collate_fn`` and ``scripts/train.py``.

Reference: https://huggingface.co/datasets/lerobot/droid_100
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from loguru import logger

from config.tokens import get_all_custom_tokens, get_token
from dataloader.eo_dataset import build_image_transform, eo_collate_fn


# ---------------------------------------------------------------------------
# Known DROID camera view keys (lerobot v2 naming)
# ---------------------------------------------------------------------------
DROID_CAMERA_KEYS = [
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_left",
    "observation.images.wrist_image_left",
]

# DROID action space: 7-DoF delta end-effector control + gripper
DROID_ACTION_DIM = 7
# DROID state: 7 joint positions + 7 Cartesian EE pose (xyz + rpy + gripper)
DROID_STATE_DIM = 14


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DROIDConfig:
    """Hyper-parameters for DROID → Halo-VLA batches."""

    data_root: str  # local clone root

    tokenizer_name: str = "HuggingFaceTB/cosmo2-tokenizer"
    img_size: int = 224
    img_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    img_std: Tuple[float, ...] = (0.229, 0.224, 0.225)

    max_seq_len: int = 512
    max_action_len: int = 256
    action_dim: int = DROID_ACTION_DIM
    state_dim: int = DROID_STATE_DIM

    # Which camera view to use for the main image stream.
    # Must be a key present under videos/<camera_key>/
    camera: str = "observation.images.exterior_image_1_left"

    # Window sampling
    num_sample_frames: int = 5    # context frames fed as model input
    num_predict_frames: int = 5   # future frames returned as DiT targets
    frame_stride: int = 6         # temporal stride between sampled frames
    min_episode_length: int = 12  # skip very short episodes

    # Episodes per chunk directory (default matches lerobot droid_100)
    episodes_per_chunk: int = 1000

    # Conversation template (ChatML, matches other loaders)
    system_prefix: str = "<|im_start|>system\n"
    human_prefix: str = "<|im_start|>user\n"
    assistant_prefix: str = "<|im_start|>assistant\n"
    turn_end: str = "<|im_end|>\n"

    max_samples: Optional[int] = None
    seed: int = 42

    # Internal — populated from meta/info.json
    fps: float = 15.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DROIDDataset(Dataset):
    """
    One item = one temporal window from one DROID episode.

    ``images``        [n, 3, H, W]  — ``num_sample_frames`` context frames
    ``future_frames`` [p, 3, H, W]  — ``num_predict_frames`` ground-truth targets
    ``actions``       [T, action_dim]
    ``states``        [T, state_dim]
    """

    def __init__(
        self,
        config: Optional[DROIDConfig] = None,
        tokenizer=None,
        split: str = "train",
        **kwargs,
    ):
        super().__init__()
        if config is None:
            config = DROIDConfig(**kwargs)
        self.cfg = config
        self.split = split
        self.root = Path(config.data_root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"data_root not found: {self.root}")

        self._setup_tokenizer(tokenizer)
        self.img_transform = build_image_transform(
            self.cfg.img_size, self.cfg.img_mean, self.cfg.img_std
        )

        self._load_meta()
        self.episodes = self._build_episode_list()

        if self.cfg.max_samples is not None:
            self.episodes = self.episodes[: self.cfg.max_samples]

        logger.info(
            "DROIDDataset: {} episodes  camera={}  frames={}/+{}  stride={}",
            len(self.episodes),
            self.cfg.camera,
            self.cfg.num_sample_frames,
            self.cfg.num_predict_frames,
            self.cfg.frame_stride,
        )

    # ------------------------------------------------------------------
    # Metadata loading
    # ------------------------------------------------------------------

    def _load_meta(self) -> None:
        """Read meta/info.json, meta/episodes.parquet, meta/tasks.parquet."""
        meta_dir = self.root / "meta"

        # info.json — fps + feature list
        info_path = meta_dir / "info.json"
        if info_path.is_file():
            with open(info_path) as f:
                info = json.load(f)
            self.cfg.fps = float(info.get("fps", self.cfg.fps))
            # Discover camera keys from info if config key is not explicit
            cam_keys = info.get("camera_keys", []) or info.get("video_keys", [])
            if cam_keys and self.cfg.camera not in cam_keys:
                # Prefer the configured key; fall back to the first available
                logger.warning(
                    "Configured camera '{}' not in info.json camera_keys {}; "
                    "using first available.",
                    self.cfg.camera,
                    cam_keys,
                )
                self.cfg.camera = cam_keys[0]
        else:
            logger.warning("meta/info.json not found — using defaults (fps={}).",
                           self.cfg.fps)

        # tasks.parquet — task_index → task string
        self._task_map: Dict[int, str] = self._load_task_map(meta_dir)

        # episodes.parquet — episode_index, length, task_index
        self._episode_meta: List[Dict[str, Any]] = self._load_episodes_parquet(meta_dir)

    def _load_task_map(self, meta_dir: Path) -> Dict[int, str]:
        tasks_pq = meta_dir / "tasks.parquet"
        if not tasks_pq.is_file():
            return {}
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(tasks_pq)
            names = table.schema.names
            idx_col = next(
                (c for c in ("task_index", "index") if c in names), None
            )
            txt_col = next(
                (c for c in ("task", "task_description", "short_horizon_task")
                 if c in names),
                None,
            )
            if not idx_col or not txt_col:
                return {}
            return {
                int(table[idx_col][i].as_py()): str(table[txt_col][i].as_py())
                for i in range(table.num_rows)
            }
        except Exception as exc:
            logger.warning("Could not load tasks.parquet: {}", exc)
            return {}

    def _load_episodes_parquet(self, meta_dir: Path) -> List[Dict[str, Any]]:
        eps_pq = meta_dir / "episodes.parquet"
        if not eps_pq.is_file():
            logger.warning("meta/episodes.parquet not found — will discover from data/.")
            return []
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(eps_pq)
            names = table.schema.names
            rows = []
            for i in range(table.num_rows):
                rec: Dict[str, Any] = {col: table[col][i].as_py() for col in names}
                if "episode_index" not in rec:
                    rec["episode_index"] = rec.get("index", i)
                rows.append(rec)
            rows.sort(key=lambda r: int(r.get("episode_index", 0)))
            return rows
        except Exception as exc:
            logger.warning("Could not load episodes.parquet: {}", exc)
            return []

    # ------------------------------------------------------------------
    # Episode list construction
    # ------------------------------------------------------------------

    def _build_episode_list(self) -> List[Dict[str, Any]]:
        """
        Merge parquet metadata with filesystem discovery.

        Returns list of episode dicts each guaranteed to have:
          episode_index, length, task (str), data_path (Path), video_path (Path).
        """
        rows = list(self._episode_meta)

        # If parquet gave us nothing, scan the data/ directory
        if not rows:
            rows = self._scan_data_dir()

        episodes = []
        for rec in rows:
            ep_idx = int(rec.get("episode_index", 0))

            data_path = self._data_path(ep_idx)
            video_path = self._video_path(ep_idx)

            # Resolve task text
            task_idx = rec.get("task_index")
            task = self._task_map.get(int(task_idx), "") if task_idx is not None else ""
            if not task:
                task = str(rec.get("task", rec.get("short_horizon_task", "robot manipulation")))

            # Prefer parquet-reported length; fall back to zero (probed on first access)
            length = int(rec.get("length", 0) or 0)

            if not data_path.is_file() and not video_path.is_file():
                continue  # skip episodes with no data at all

            if length > 0 and length < self.cfg.min_episode_length:
                continue

            episodes.append({
                "episode_index": ep_idx,
                "length": length,
                "task": task,
                "data_path": data_path,
                "video_path": video_path,
            })

        logger.info("DROIDDataset: {} usable episodes after filtering.", len(episodes))
        return episodes

    def _scan_data_dir(self) -> List[Dict[str, Any]]:
        """Fallback: discover episodes by scanning data/chunk-*/ parquet files."""
        import re
        data_root = self.root / "data"
        if not data_root.is_dir():
            return []
        rows = []
        for ep_pq in sorted(data_root.rglob("episode_*.parquet")):
            m = re.search(r"episode_(\d+)", ep_pq.stem)
            if not m:
                continue
            ep_idx = int(m.group(1))
            rows.append({"episode_index": ep_idx, "length": 0, "task": ""})
        return rows

    def _data_path(self, episode_index: int) -> Path:
        """Path to the per-episode parquet file under data/chunk-XXX/."""
        chunk = episode_index // self.cfg.episodes_per_chunk
        return (
            self.root
            / "data"
            / f"chunk-{chunk:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )

    def _video_path(self, episode_index: int) -> Path:
        """Path to the MP4 for the configured camera view."""
        chunk = episode_index // self.cfg.episodes_per_chunk
        return (
            self.root
            / "videos"
            / f"chunk-{chunk:03d}"
            / self.cfg.camera
            / f"episode_{episode_index:06d}.mp4"
        )

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    def _setup_tokenizer(self, tokenizer) -> None:
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.tokenizer_name, trust_remote_code=True,
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        extra = [
            tok for tok in get_all_custom_tokens()
            if tok not in self.tokenizer.get_vocab()
        ]
        if extra:
            self.tokenizer.add_special_tokens({"additional_special_tokens": extra})

        self._image_tok = get_token("image_token")
        self._action_tok = get_token("action_token")
        self._state_tok = get_token("state_token")
        self._world_video_tok = get_token("world_video_token")
        self.pad_token_id = self.tokenizer.pad_token_id

    # ------------------------------------------------------------------
    # Episode data loading
    # ------------------------------------------------------------------

    def _load_episode_parquet(
        self, data_path: Path
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Parse a per-episode parquet file.

        Returns:
            actions  : float32 [T, action_dim]
            states   : float32 [T, state_dim]
            length   : int  (number of frames / rows)
        """
        try:
            import pyarrow.parquet as pq
        except ImportError:
            logger.warning("pyarrow not available; returning dummy actions/states.")
            return (
                np.zeros((0, self.cfg.action_dim), np.float32),
                np.zeros((0, self.cfg.state_dim), np.float32),
                0,
            )

        if not data_path.is_file():
            return (
                np.zeros((0, self.cfg.action_dim), np.float32),
                np.zeros((0, self.cfg.state_dim), np.float32),
                0,
            )

        try:
            table = pq.read_table(data_path)
        except Exception as exc:
            logger.warning("Could not read {}: {}", data_path, exc)
            return (
                np.zeros((0, self.cfg.action_dim), np.float32),
                np.zeros((0, self.cfg.state_dim), np.float32),
                0,
            )

        T = table.num_rows
        names = set(table.schema.names)

        # ── Actions ─────────────────────────────────────────────────────────
        action_col = next(
            (c for c in ("action", "action.cartesian_velocity", "actions") if c in names),
            None,
        )
        if action_col:
            raw_act = table[action_col].to_pylist()
            acts = self._nested_to_array(raw_act, T)
        else:
            acts = np.zeros((T, 0), np.float32)

        D_a = self.cfg.action_dim
        if acts.shape[1] >= D_a:
            acts = acts[:, :D_a]
        else:
            pad = np.zeros((T, D_a - acts.shape[1]), np.float32)
            acts = np.concatenate([acts, pad], axis=1)

        # ── States ───────────────────────────────────────────────────────────
        state_col = next(
            (c for c in ("observation.state", "state", "observation.robot_state")
             if c in names),
            None,
        )
        if state_col:
            raw_st = table[state_col].to_pylist()
            sts = self._nested_to_array(raw_st, T)
        else:
            sts = np.zeros((T, 0), np.float32)

        D_s = self.cfg.state_dim
        if sts.shape[1] >= D_s:
            sts = sts[:, :D_s]
        else:
            pad = np.zeros((T, D_s - sts.shape[1]), np.float32)
            sts = np.concatenate([sts, pad], axis=1)

        return acts.astype(np.float32), sts.astype(np.float32), T

    @staticmethod
    def _nested_to_array(rows: List[Any], T: int) -> np.ndarray:
        """
        Convert a list of rows (each a float or list[float]) to float32 [T, D].
        Handles both scalar and vector action/state formats.
        """
        if not rows:
            return np.zeros((T, 1), np.float32)
        first = rows[0]
        if isinstance(first, (int, float)):
            return np.array(rows, dtype=np.float32).reshape(T, 1)
        try:
            arr = np.array(rows, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(T, 1)
            return arr
        except (ValueError, TypeError):
            # Ragged rows — pad each to max width
            maxd = max((len(r) if isinstance(r, (list, tuple)) else 1) for r in rows)
            out = np.zeros((T, maxd), np.float32)
            for i, r in enumerate(rows):
                if isinstance(r, (list, tuple)):
                    d = min(len(r), maxd)
                    out[i, :d] = r[:d]
                else:
                    out[i, 0] = float(r)
            return out

    # ------------------------------------------------------------------
    # Video frame reading (PyAV, same approach as AiroaMomaDataset)
    # ------------------------------------------------------------------

    def _get_video_length(self, video_path: Path) -> int:
        """Return frame count without full decode."""
        try:
            import av
            container = av.open(str(video_path))
            stream = container.streams.video[0]
            if stream.frames and stream.frames > 0:
                container.close()
                return int(stream.frames)
            if stream.duration and stream.average_rate:
                dur_s = float(stream.duration) * stream.time_base
                fps = float(stream.average_rate)
                if dur_s > 0 and fps > 0:
                    container.close()
                    return max(1, round(dur_s * fps))
            count = sum(1 for p in container.demux(stream) if p.size > 0)
            container.close()
            return max(count, 1)
        except Exception as exc:
            logger.warning("Could not get length of {}: {}", video_path, exc)
            return 0

    def _read_frames_at_indices(
        self, video_path: Path, indices: Sequence[int]
    ) -> List[torch.Tensor]:
        """Decode specific frame indices using PyAV with a sequential counter."""
        try:
            import av
        except ImportError as exc:
            raise ImportError("PyAV is required: pip install av") from exc
        from PIL import Image

        blank = torch.zeros(3, self.cfg.img_size, self.cfg.img_size)
        if not video_path.is_file():
            return [blank.clone() for _ in indices]

        try:
            container = av.open(str(video_path))
        except Exception as exc:
            logger.warning("Cannot open {}: {}", video_path, exc)
            return [blank.clone() for _ in indices]

        stream = container.streams.video[0]
        required = set(int(i) for i in indices)
        frame_map: Dict[int, torch.Tensor] = {}
        counter = 0

        for packet in container.demux(stream):
            for frame in packet.decode():
                if counter in required:
                    img = frame.to_image().convert("RGB")
                    frame_map[counter] = self.img_transform(img)
                    required.discard(counter)
                    if not required:
                        break
                counter += 1
            if not required:
                break

        container.close()

        return [frame_map.get(int(i), blank.clone()) for i in indices]

    def _dummy_frames(self, n: int) -> torch.Tensor:
        return torch.zeros(n, 3, self.cfg.img_size, self.cfg.img_size)

    # ------------------------------------------------------------------
    # Conversation / tokenisation
    # ------------------------------------------------------------------

    def _build_text(self, task: str, num_images: int) -> Tuple[str, str]:
        """
        Build the (user, assistant) string pair for one sample.

        One ``<image>`` token per context frame, one ``<state>`` token,
        one ``<halo_world_video>`` per future frame, one ``<action>`` token.
        """
        img_tags = " ".join([self._image_tok] * num_images)
        user_body = (
            f"{img_tags} {self._state_tok} Task: {task}\n"
            "Predict the robot motion and describe the next action."
        )
        wv_tokens = self._world_video_tok * self.cfg.num_predict_frames
        assistant_body = (
            f"Executing: {task}\n"
            f"{wv_tokens}{self._action_tok}"
        )
        return user_body, assistant_body

    def _tokenise(
        self, user_body: str, assistant_body: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from config import HaloVLMConfig
        model_cfg = HaloVLMConfig()
        system_msg = f"{self.cfg.system_prefix}{model_cfg.system_prompt}{self.cfg.turn_end}"
        human_msg = f"{self.cfg.human_prefix}{user_body}{self.cfg.turn_end}"
        asst_msg = f"{self.cfg.assistant_prefix}{assistant_body}{self.cfg.turn_end}"
        full_text = system_msg + human_msg + asst_msg

        enc = self.tokenizer(
            full_text,
            max_length=self.cfg.max_seq_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # Mask system + user spans so loss only trains on the assistant reply
        parts = [system_msg, human_msg, asst_msg]
        is_asst = [False, False, True]
        labels = self._mask_non_assistant(full_text, parts, is_asst, labels)
        labels[attention_mask == 0] = -100

        return input_ids, attention_mask, labels

    def _mask_non_assistant(
        self,
        full_text: str,
        parts: List[str],
        is_asst: List[bool],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        cursor = 0
        for part, assistant in zip(parts, is_asst):
            start = full_text.find(part, cursor)
            if start == -1:
                continue
            end = start + len(part)
            if not assistant:
                prefix_ids = self.tokenizer(
                    full_text[:end],
                    max_length=self.cfg.max_seq_len,
                    truncation=True,
                    add_special_tokens=False,
                )["input_ids"]
                start_ids = self.tokenizer(
                    full_text[:start],
                    max_length=self.cfg.max_seq_len,
                    truncation=True,
                    add_special_tokens=False,
                )["input_ids"]
                tok_start = len(start_ids)
                tok_end = len(prefix_ids)
                if tok_end <= labels.numel():
                    labels[tok_start:tok_end] = -100
            cursor = end
        return labels

    # ------------------------------------------------------------------
    # Window sampling helpers
    # ------------------------------------------------------------------

    def _sample_window(
        self, T: int, idx: int
    ) -> Tuple[List[int], List[int]]:
        """
        Sample ``num_sample_frames + num_predict_frames`` indices with ``frame_stride``.

        Training: per-item seeded random start for diversity while remaining
        reproducible.  Val/test: always starts at 0.
        """
        n = self.cfg.num_sample_frames
        p = self.cfg.num_predict_frames
        total = n + p
        stride = self.cfg.frame_stride
        span = (total - 1) * stride + 1

        if T < span:
            # Shrink stride to fit
            stride = max(1, (T - 1) // max(total - 1, 1))
            span = (total - 1) * stride + 1

        t0_max = max(0, T - span)
        if self.split != "train" or t0_max == 0:
            t0 = 0
        else:
            rng = random.Random(self.cfg.seed + idx)
            t0 = rng.randint(0, t0_max)

        all_idx = [t0 + j * stride for j in range(total)]
        all_idx = [min(i, T - 1) for i in all_idx]

        return all_idx[:n], all_idx[n:]

    # ------------------------------------------------------------------
    # __len__ / __getitem__
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep = self.episodes[idx]
        ep_idx = int(ep["episode_index"])
        task = ep["task"] or "robot manipulation"
        data_path = ep["data_path"]
        video_path = ep["video_path"]

        # ── Resolve episode length ──────────────────────────────────────────
        T = int(ep["length"] or 0)
        if T == 0:
            # Probe from parquet row count (fast, no video decode)
            if data_path.is_file():
                try:
                    import pyarrow.parquet as pq
                    T = pq.read_metadata(data_path).num_rows
                except Exception:
                    pass
            if T == 0 and video_path.is_file():
                T = self._get_video_length(video_path)
            if T == 0:
                T = self.cfg.num_sample_frames + self.cfg.num_predict_frames

        # ── Sample frame indices ────────────────────────────────────────────
        n = self.cfg.num_sample_frames
        p = self.cfg.num_predict_frames
        ctx_indices, fut_indices = self._sample_window(T, idx)

        # ── Load actions + states from parquet ─────────────────────────────
        acts_all, sts_all, T_pq = self._load_episode_parquet(data_path)

        # Slice the action/state window aligned to context frames
        actions, action_mask, states, state_mask = self._slice_actions_states(
            acts_all, sts_all, ctx_indices
        )

        # ── Load video frames ───────────────────────────────────────────────
        all_frame_indices = ctx_indices + fut_indices
        if video_path.is_file():
            frame_tensors = self._read_frames_at_indices(video_path, all_frame_indices)
            images = torch.stack(frame_tensors[:n], dim=0)
            future_frames = torch.stack(frame_tensors[n:], dim=0)
        else:
            logger.warning("Video missing: {} — using dummy frames", video_path)
            images = self._dummy_frames(n)
            future_frames = self._dummy_frames(p)

        # ── Conversation tokens ─────────────────────────────────────────────
        user_body, asst_body = self._build_text(task, num_images=n)
        input_ids, attention_mask_tok, labels = self._tokenise(user_body, asst_body)

        # Pool states: one <state> token in the prompt → single vector [1, state_dim]
        if states.numel() > 0:
            states_in = states.mean(dim=0, keepdim=True)     # [1, state_dim]
            state_mask_in = torch.ones(1, dtype=torch.long)
        else:
            states_in = torch.zeros(1, self.cfg.state_dim)
            state_mask_in = torch.zeros(1, dtype=torch.long)

        return {
            "images": images,                      # [n, 3, H, W]
            "future_frames": future_frames,        # [p, 3, H, W]
            "input_ids": input_ids,                # [seq_len]
            "attention_mask": attention_mask_tok,  # [seq_len]
            "labels": labels,                      # [seq_len]
            "actions": actions,                    # [T, action_dim]
            "action_mask": action_mask,            # [T]
            "states": states_in,                   # [1, state_dim]
            "state_mask": state_mask_in,           # [1]
            "num_images": images.size(0),
            "pad_token_id": self.pad_token_id,
        }

    def _slice_actions_states(
        self,
        acts: np.ndarray,
        sts: np.ndarray,
        frame_indices: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract and truncate action + state rows aligned to ``frame_indices``.

        DROID parquet already has one action per frame, so we just index-select
        the rows that correspond to the sampled context window.
        """
        D_a = self.cfg.action_dim
        D_s = self.cfg.state_dim

        if acts.shape[0] == 0:
            # No parquet data — return empty tensors (training continues
            # with the language/vision branches only).
            return (
                torch.zeros(0, D_a),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, D_s),
                torch.zeros(0, dtype=torch.long),
            )

        T = acts.shape[0]
        idx_clamped = [min(max(0, i), T - 1) for i in frame_indices]

        sel_acts = acts[idx_clamped]   # [n, D_a]
        sel_sts = sts[idx_clamped] if sts.shape[0] > 0 else np.zeros((len(idx_clamped), D_s), np.float32)

        # Consecutive action differences give a delta-action representation.
        # When the parquet already stores delta actions (DROID default), the
        # raw values are used directly and this diff is a second-order delta —
        # acceptable for imitation learning.  Override this behaviour by
        # setting action_dim to the raw parquet width if you need absolute actions.
        if sel_acts.shape[0] > 1:
            diff = np.diff(sel_acts, axis=0)   # [n-1, D_a]
        else:
            diff = sel_acts                    # [1, D_a] when single frame

        max_t = self.cfg.max_action_len
        diff = diff[:max_t]
        action_mask = torch.ones(diff.shape[0], dtype=torch.long)

        return (
            torch.from_numpy(diff),
            action_mask,
            torch.from_numpy(sel_sts.astype(np.float32)),
            torch.ones(sel_sts.shape[0], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_droid_dataloader(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 0,
    tokenizer=None,
    img_size: int = 224,
    max_seq_len: int = 512,
    max_action_len: int = 256,
    action_dim: int = DROID_ACTION_DIM,
    state_dim: int = DROID_STATE_DIM,
    camera: str = "observation.images.exterior_image_1_left",
    num_sample_frames: int = 5,
    num_predict_frames: int = 5,
    frame_stride: int = 6,
    split: str = "train",
    shuffle: bool = True,
    max_samples: Optional[int] = None,
    pin_memory: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Build a DataLoader for DROID compatible with ``scripts/train.py``.

    Keys returned per batch match those of the MoMa and EO loaders so the
    training loop needs no changes.

    Args:
        data_root: Local path to the lerobot/droid_100 clone.
        num_workers: Set to 0 if PyAV is not fork-safe on your system.
        camera: Which camera view to use for image frames.
            Options: ``observation.images.exterior_image_1_left`` (default),
            ``observation.images.exterior_image_2_left``,
            ``observation.images.wrist_image_left``.
        num_sample_frames: Context frames given to the model.
        num_predict_frames: Ground-truth future frames returned for the DiT head.
        frame_stride: Temporal gap between sampled frames (larger = more motion).
    """
    cfg = DROIDConfig(
        data_root=data_root,
        img_size=img_size,
        max_seq_len=max_seq_len,
        max_action_len=max_action_len,
        action_dim=action_dim,
        state_dim=state_dim,
        camera=camera,
        num_sample_frames=num_sample_frames,
        num_predict_frames=num_predict_frames,
        frame_stride=frame_stride,
        max_samples=max_samples,
        **{k: v for k, v in kwargs.items()
           if k in DROIDConfig.__dataclass_fields__},
    )
    ds = DROIDDataset(config=cfg, tokenizer=tokenizer, split=split)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle and split == "train",
        num_workers=num_workers,
        collate_fn=eo_collate_fn,
        pin_memory=pin_memory and num_workers > 0,
        drop_last=len(ds) > batch_size,
    )
