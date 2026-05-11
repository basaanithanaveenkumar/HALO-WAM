"""
EO-Data1.5M Dataloader for Halo-VLA.

Supports the IPEC-COMMUNITY/EO-Data1.5M dataset from HuggingFace,
including all 17 subsets (5 interleaved + 12 QA).

Dataset schema per sample:
    source       : str   — source dataset identifier
    conversation : list  — multi-turn dicts [{"from": ..., "value": ...}, ...]
    image        : PIL.Image or list[PIL.Image] — visual observations
    action       : list/ndarray or None — robot action chunks (continuous)
    state        : list/ndarray or None — robot proprioceptive state

Reference: https://huggingface.co/datasets/IPEC-COMMUNITY/EO-Data1.5M
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from loguru import logger

# ---------------------------------------------------------------------------
# Image token placeholder used inside conversation text
# ---------------------------------------------------------------------------
from config.tokens import SPECIAL_TOKENS, get_all_custom_tokens, get_token, get_token_id

IMAGE_TOKEN = get_token("image_token")
ACTION_TOKEN = get_token("action_token")
STATE_TOKEN = get_token("state_token")

# ---------------------------------------------------------------------------
# All available subsets
# ---------------------------------------------------------------------------
INTERLEAVE_SUBSETS = [
    "interleave-free_chat",
    "interleave-random_qa",
    "interleave-temporal",
    "interleave-trajectory",
    "interleave-video_caption",
]

QA_SUBSETS = [
    "qa-affordance_qa",
    "qa-episode_caption",
    "qa-failure_detection",
    "qa-multiview_qa",
    "qa-object_referring_qa",
    "qa-physical_common_sense",
    "qa-points_qa",
    "qa-process_verification",
    "qa-relation_reasoning",
    "qa-subtask_qa",
    "qa-task_planning",
    "qa-trajectory_qa",
]

ALL_SUBSETS = INTERLEAVE_SUBSETS + QA_SUBSETS


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class EODatasetConfig:
    """Configuration for the EO-Dataset dataloader."""

    dataset_name: str = "IPEC-COMMUNITY/EO-Data1.5M"
    subset: str = "interleave-temporal"
    split: str = "train"
    img_size: int = 224
    max_seq_len: int = 77
    max_action_len: int = 64
    action_dim: int = 32   # EO-Data uses 32-dim actions (varies by embodiment)
    state_dim: int = 32
    max_images: int = 8  # max images per sample for interleaved data
    tokenizer_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    max_samples: Optional[int] = None   # limit dataset size (None = all)
    streaming: bool = False
    cache_dir: Optional[str] = None
    # Image normalisation (ImageNet defaults)
    img_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    img_std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    # Conversation formatting
    system_prefix: str = "<|im_start|>system\n"
    human_prefix: str = "<|im_start|>user\n"
    assistant_prefix: str = "<|im_start|>assistant\n"
    turn_end: str = "<|im_end|>\n"
    # Special token IDs (set after tokenizer init)
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    image_token_id: int = 3
    action_token_id: int = 4
    state_token_id: int = 5


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------
def build_image_transform(img_size: int, mean: Tuple, std: Tuple):
    """Build torchvision transforms for images."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


# ---------------------------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------------------------
class EODataset(Dataset):
    """
    PyTorch Dataset for IPEC-COMMUNITY/EO-Data1.5M.

    Each __getitem__ returns a dict with keys:
        - images        : Tensor [N, 3, H, W]  (N = number of images, 1..max_images)
        - input_ids     : Tensor [seq_len]       tokenised conversation
        - attention_mask : Tensor [seq_len]
        - labels        : Tensor [seq_len]       shifted targets (-100 for ignored)
        - actions       : Tensor [T, action_dim] continuous action chunks
        - action_mask   : Tensor [T]             1 where action is valid
        - states        : Tensor [T, state_dim]  proprioceptive state
        - state_mask    : Tensor [T]
        - num_images    : int                    actual number of images
        - source        : str                    source dataset id
    """

    def __init__(
        self,
        config: Optional[EODatasetConfig] = None,
        hf_dataset=None,
        tokenizer=None,
        **kwargs,
    ):
        super().__init__()
        self.cfg = config or EODatasetConfig(**kwargs)
        self._setup_tokenizer(tokenizer)
        self._setup_transform()
        self._load_dataset(hf_dataset)

    # ------------------------------------------------------------------ init
    def _setup_tokenizer(self, tokenizer):
        """Initialise or configure the tokenizer."""
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.tokenizer_name,
                trust_remote_code=True,
            )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add special tokens for image / action / state if missing
        special_tokens: Dict[str, str] = {}
        additional_special: List[str] = []
        for tok in get_all_custom_tokens():
            if tok not in self.tokenizer.get_vocab():
                additional_special.append(tok)
        if additional_special:
            special_tokens["additional_special_tokens"] = additional_special
        if special_tokens:
            self.tokenizer.add_special_tokens(special_tokens)

        # Cache token ids
        self.cfg.pad_token_id = self.tokenizer.pad_token_id
        self.cfg.bos_token_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else self.tokenizer.pad_token_id
        )
        self.cfg.eos_token_id = self.tokenizer.eos_token_id
        self.cfg.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        self.cfg.action_token_id = self.tokenizer.convert_tokens_to_ids(ACTION_TOKEN)
        self.cfg.state_token_id = self.tokenizer.convert_tokens_to_ids(STATE_TOKEN)

    def _setup_transform(self):
        self.img_transform = build_image_transform(
            self.cfg.img_size, self.cfg.img_mean, self.cfg.img_std
        )

    def _load_dataset(self, hf_dataset):
        """Load the HuggingFace dataset (or accept a pre-loaded one)."""
        if hf_dataset is not None:
            self.dataset = hf_dataset
            return

        from datasets import load_dataset

        logger.info(
            "Loading {} / {} [{}] …",
            self.cfg.dataset_name,
            self.cfg.subset,
            self.cfg.split,
        )
        self.dataset = load_dataset(
            self.cfg.dataset_name,
            name=self.cfg.subset,
            split=self.cfg.split,
            streaming=self.cfg.streaming,
            cache_dir=self.cfg.cache_dir,
        )
        # Optionally truncate to max_samples for fast iteration
        if self.cfg.max_samples is not None and self.cfg.max_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(self.cfg.max_samples))
            logger.info("Truncated to {} samples (--max_samples)", len(self.dataset))

        logger.info("Dataset loaded: {} samples", len(self.dataset))
        logger.info("Building pair index (splitting multi-turn conversations)...")
        self._build_index()

    def _build_index(self):
        """
        Build a compact cumulative-sum array so we can map a virtual index
        to (sample_idx, pair_idx) without storing a tuple per pair.

        Only reads the lightweight ``conversation`` column — never loads
        images, actions, or states — so it is fast and memory-efficient.
        The stored array has one int32 entry per sample (≈125 KB for 31 K
        samples) instead of a list of tuples.
        """
        import numpy as np

        n_samples = len(self.dataset)
        counts = np.empty(n_samples, dtype=np.int32)

        # Access only the 'conversation' column for speed
        for i in range(n_samples):
            conv = self.dataset[i].get("conversation", [])
            # Count user–assistant pairs by scanning roles
            n_pairs = 0
            j = 0
            while j < len(conv):
                role = conv[j].get("from", conv[j].get("role", "")).lower()
                if role in ("human", "user") and j + 1 < len(conv):
                    next_role = conv[j + 1].get("from", conv[j + 1].get("role", "")).lower()
                    if next_role not in ("human", "user"):
                        n_pairs += 1
                        j += 2
                        continue
                j += 1
            counts[i] = max(n_pairs, 1)  # at least 1 item per sample

        self._cum_pairs = np.cumsum(counts)          # [n_samples]
        self._total_items = int(self._cum_pairs[-1])

        logger.info(
            "Expanded to {} items ({} samples × avg {:.1f} pairs)",
            self._total_items, n_samples,
            self._total_items / max(n_samples, 1),
        )

    def _virtual_to_real(self, idx: int) -> Tuple[int, int]:
        """Map a flat virtual index → (sample_idx, pair_idx) via binary search."""
        import numpy as np
        sample_idx = int(np.searchsorted(self._cum_pairs, idx, side="right"))
        pair_idx = idx if sample_idx == 0 else idx - int(self._cum_pairs[sample_idx - 1])
        return sample_idx, pair_idx

    # ---------------------------------------------------------- length / get
    def __len__(self) -> int:
        return self._total_items

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_idx, pair_idx = self._virtual_to_real(idx)
        sample = self.dataset[sample_idx]

        # ----- Images -----
        images = self._process_images(sample)

        # ----- Conversation → token ids (single user–assistant pair) -----
        input_ids, attention_mask, labels = self._process_conversation(sample, pair_idx=pair_idx)

        # ----- Actions -----
        actions, action_mask = self._process_actions(sample)

        # ----- States -----
        states, state_mask = self._process_states(sample)

        return {
            "images": images,                 # [N, 3, H, W]
            "input_ids": input_ids,           # [seq_len]
            "attention_mask": attention_mask,  # [seq_len]
            "labels": labels,                 # [seq_len]
            "actions": actions,               # [T, action_dim]
            "action_mask": action_mask,        # [T]
            "states": states,                 # [T, state_dim]
            "state_mask": state_mask,          # [T]
            "num_images": images.size(0),
            "pad_token_id": self.cfg.pad_token_id,
        }

    # ------------------------------------------------------------- images
    def _process_images(self, sample: Dict) -> torch.Tensor:
        """
        Extract and transform images.

        The `image` field can be:
            - a single PIL Image
            - a list of PIL Images (multi-view / temporal)
            - bytes that can be decoded
        Returns Tensor [N, 3, H, W], padded to max_images with zeros.
        """
        from PIL import Image
        import io

        raw = sample.get("image")
        imgs: List[torch.Tensor] = []

        if raw is None:
            # No image → return a black placeholder
            return torch.zeros(1, 3, self.cfg.img_size, self.cfg.img_size)

        # Normalise to a list
        if not isinstance(raw, (list, tuple)):
            raw = [raw]

        for item in raw[: self.cfg.max_images]:
            if isinstance(item, Image.Image):
                img = item.convert("RGB")
            elif isinstance(item, bytes):
                img = Image.open(io.BytesIO(item)).convert("RGB")
            elif isinstance(item, dict) and "bytes" in item:
                img = Image.open(io.BytesIO(item["bytes"])).convert("RGB")
            elif isinstance(item, str):
                img = Image.open(item).convert("RGB")
            else:
                # Try PIL conversion as fallback
                try:
                    img = Image.fromarray(np.uint8(item)).convert("RGB")
                except Exception:
                    logger.warning("Skipping unrecognised image format: {}", type(item))
                    continue
            imgs.append(self.img_transform(img))

        if not imgs:
            return torch.zeros(1, 3, self.cfg.img_size, self.cfg.img_size)

        return torch.stack(imgs, dim=0)  # [N, 3, H, W]

    # ------------------------------------------------------- conversation
    def _pair_conversations(self, sample: Dict) -> List[List[Dict]]:
        """
        Split a multi-turn conversation into individual (user, assistant) pairs.

        Each pair is returned as a list of two dicts so that a single HF sample
        with N user–assistant exchanges produces N independent training items,
        each containing exactly one user question and one assistant response.
        """
        conversation = sample.get("conversation", [])
        pairs: List[List[Dict]] = []
        i = 0
        while i < len(conversation):
            turn = conversation[i]
            role = turn.get("from", turn.get("role", "human")).lower()
            if role in ("human", "user"):
                # Grab the following assistant turn if it exists
                if i + 1 < len(conversation):
                    next_turn = conversation[i + 1]
                    next_role = next_turn.get("from", next_turn.get("role", "")).lower()
                    if next_role not in ("human", "user"):
                        pairs.append([turn, next_turn])
                        i += 2
                        continue
                # No matching assistant turn — skip lone user turn
                i += 1
            else:
                i += 1
        return pairs if pairs else [[]]

    def _process_conversation(
        self, sample: Dict, pair_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build a token sequence for a single (system + user + assistant) triple.

        Args:
            sample:   Raw HF sample dict.
            pair_idx: Which user–assistant pair to use (0-indexed).

        Returns:
            input_ids      : LongTensor [seq_len]
            attention_mask  : LongTensor [seq_len]
            labels          : LongTensor [seq_len]  (-100 for non-assistant tokens)
        """
        pairs = self._pair_conversations(sample)
        if not pairs or not pairs[0]:
            return self._empty_text()

        pair_idx = min(pair_idx, len(pairs) - 1)
        pair = pairs[pair_idx]

        text_parts: List[str] = []
        role_is_assistant: List[bool] = []

        # 1. System prompt
        from config import HaloVLMConfig
        model_cfg = HaloVLMConfig()
        system_msg = f"{self.cfg.system_prefix}{model_cfg.system_prompt}{self.cfg.turn_end}"
        text_parts.append(system_msg)
        role_is_assistant.append(False)

        # 2. One user turn + one assistant turn
        for turn in pair:
            role = turn.get("from", turn.get("role", "human")).lower()
            value = turn.get("value", turn.get("content", ""))
            if role in ("human", "user"):
                text_parts.append(f"{self.cfg.human_prefix}{value}{self.cfg.turn_end}")
                role_is_assistant.append(False)
            else:
                text_parts.append(f"{self.cfg.assistant_prefix}{value}{self.cfg.turn_end}")
                role_is_assistant.append(True)

        full_text = "".join(text_parts)

        # Tokenise — no padding here; collate pads to per-batch max
        encoding = self.tokenizer(
            full_text,
            max_length=self.cfg.max_seq_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)             # [seq_len]
        attention_mask = encoding["attention_mask"].squeeze(0)    # [seq_len]

        # Build labels: mask out human turns (set to -100 so loss ignores them)
        labels = input_ids.clone()
        # Simple heuristic: tokenise each turn individually to find boundaries
        labels = self._mask_human_turns(
            full_text, text_parts, role_is_assistant, labels
        )
        # Also mask padding
        labels[attention_mask == 0] = -100

        return input_ids, attention_mask, labels

    def _mask_human_turns(
        self,
        full_text: str,
        text_parts: List[str],
        role_is_assistant: List[bool],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Set labels to -100 for tokens corresponding to human turns."""
        cursor = 0
        for part, is_assistant in zip(text_parts, role_is_assistant):
            start_char = full_text.find(part, cursor)
            if start_char == -1:
                continue
            end_char = start_char + len(part)

            if not is_assistant:
                # Find token offsets for this span
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
                if tok_end <= self.cfg.max_seq_len:
                    labels[tok_start:tok_end] = -100

            cursor = end_char
        return labels

    def _empty_text(self):
        """Return minimal empty tensors when no conversation is present."""
        ids = torch.full((1,), self.cfg.pad_token_id, dtype=torch.long)
        mask = torch.zeros(1, dtype=torch.long)
        labels = torch.full((1,), -100, dtype=torch.long)
        return ids, mask, labels

    # --------------------------------------------------------- actions
    def _process_actions(
        self, sample: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process robot action chunks — no zero-padding, original sizes.

        EO-Data format:
            action       : [num_chunks, steps_per_chunk, action_dim]
                           e.g. (1, 16, 32) — may also be 1-D or 2-D
            action_is_pad: optional mask indicating padded steps

        Returns:
            actions     : FloatTensor [T, action_dim]  (actual timesteps)
            action_mask : LongTensor  [T]              (1 = valid)
        """
        raw = sample.get("action")

        if raw is None:
            return torch.zeros(0, self.cfg.action_dim), torch.zeros(0, dtype=torch.long)

        arr = self._to_numpy(raw)
        if arr is None or arr.size == 0:
            return torch.zeros(0, self.cfg.action_dim), torch.zeros(0, dtype=torch.long)

        # Flatten 3-D [num_chunks, steps, dim] → [total_steps, dim]
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)

        D = min(arr.shape[1], self.cfg.action_dim)
        actions = torch.tensor(arr[:, :D], dtype=torch.float32)    # [T, D]
        action_mask = torch.ones(actions.shape[0], dtype=torch.long)

        # Honour action_is_pad if present
        pad_flag = sample.get("action_is_pad")
        if pad_flag is not None:
            pad_arr = self._to_numpy(pad_flag)
            if pad_arr is not None:
                pad_arr = pad_arr.flatten()[: actions.shape[0]]
                for i, p in enumerate(pad_arr):
                    if p:
                        action_mask[i] = 0

        return actions, action_mask

    # ---------------------------------------------------------- states
    def _process_states(
        self, sample: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process robot proprioceptive state — no zero-padding, original sizes.

        Returns:
            states     : FloatTensor [T, state_dim]  (actual timesteps)
            state_mask : LongTensor  [T]
        """
        raw = sample.get("state")

        if raw is None:
            return torch.zeros(0, self.cfg.state_dim), torch.zeros(0, dtype=torch.long)

        arr = self._to_numpy(raw)
        if arr is None or arr.size == 0:
            return torch.zeros(0, self.cfg.state_dim), torch.zeros(0, dtype=torch.long)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])

        D = min(arr.shape[1], self.cfg.state_dim)
        states = torch.tensor(arr[:, :D], dtype=torch.float32)
        state_mask = torch.ones(states.shape[0], dtype=torch.long)

        return states, state_mask

    # ---------------------------------------------------------- utils
    @staticmethod
    def _to_numpy(raw) -> Optional[np.ndarray]:
        """Convert various action/state representations to numpy."""
        if isinstance(raw, np.ndarray):
            return raw.astype(np.float32)
        if isinstance(raw, torch.Tensor):
            return raw.numpy().astype(np.float32)
        if isinstance(raw, (list, tuple)):
            try:
                return np.array(raw, dtype=np.float32)
            except (ValueError, TypeError):
                return None
        return None


# ---------------------------------------------------------------------------
# Collate function for variable-length image sequences
# ---------------------------------------------------------------------------
def eo_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate that pads images, actions, and states to the
    per-batch maximum (no fixed padding).

    Returns:
        images        : [B, max_N, 3, H, W]
        image_mask    : [B, max_N]            (1 for real images, 0 for pad)
        input_ids     : [B, seq_len]
        attention_mask : [B, seq_len]
        labels        : [B, seq_len]
        actions       : [B, max_T_act, action_dim]
        action_mask   : [B, max_T_act]
        states        : [B, max_T_state, state_dim]
        state_mask    : [B, max_T_state]
    """
    B = len(batch)

    # --- Images (variable count per sample) ---
    max_n_images = max(b["num_images"] for b in batch)
    C = batch[0]["images"].size(1)
    H = batch[0]["images"].size(2)
    W = batch[0]["images"].size(3) if batch[0]["images"].dim() == 4 else batch[0]["images"].size(2)
    padded_images = torch.zeros(B, max_n_images, C, H, W)
    image_mask = torch.zeros(B, max_n_images, dtype=torch.long)

    # --- Actions (variable timesteps per sample) ---
    action_dim = batch[0]["actions"].size(-1) if batch[0]["actions"].dim() == 2 else 0
    max_t_act = max(b["actions"].size(0) for b in batch)
    max_t_act = max(max_t_act, 1)  # at least 1 to avoid empty tensors
    padded_actions = torch.zeros(B, max_t_act, action_dim)
    padded_action_mask = torch.zeros(B, max_t_act, dtype=torch.long)

    # --- States (variable timesteps per sample) ---
    state_dim = batch[0]["states"].size(-1) if batch[0]["states"].dim() == 2 else 0
    max_t_state = max(b["states"].size(0) for b in batch)
    max_t_state = max(max_t_state, 1)
    padded_states = torch.zeros(B, max_t_state, state_dim)
    padded_state_mask = torch.zeros(B, max_t_state, dtype=torch.long)

    input_ids = []
    attention_masks = []
    labels_list = []

    for i, b in enumerate(batch):
        # Images
        n = b["num_images"]
        padded_images[i, :n] = b["images"][:n]
        image_mask[i, :n] = 1

        # Text
        input_ids.append(b["input_ids"])
        attention_masks.append(b["attention_mask"])
        labels_list.append(b["labels"])

        # Actions
        t_act = b["actions"].size(0)
        if t_act > 0:
            padded_actions[i, :t_act] = b["actions"]
            padded_action_mask[i, :t_act] = b["action_mask"]

        # States
        t_state = b["states"].size(0)
        if t_state > 0:
            padded_states[i, :t_state] = b["states"]
            padded_state_mask[i, :t_state] = b["state_mask"]

    # --- Text (dynamic padding to per-batch max) ---
    pad_token_id = batch[0].get("pad_token_id", 0)
    max_text_len = max(ids.size(0) for ids in input_ids)
    padded_input_ids = torch.full((B, max_text_len), pad_token_id, dtype=torch.long)
    padded_attention_mask = torch.zeros(B, max_text_len, dtype=torch.long)
    padded_labels = torch.full((B, max_text_len), -100, dtype=torch.long)
    for i in range(B):
        L = input_ids[i].size(0)
        padded_input_ids[i, :L] = input_ids[i]
        padded_attention_mask[i, :L] = attention_masks[i]
        padded_labels[i, :L] = labels_list[i]

    return {
        "images": padded_images,                          # [B, max_N, 3, H, W]
        "image_mask": image_mask,                         # [B, max_N]
        "input_ids": padded_input_ids,                    # [B, max_text_len]
        "attention_mask": padded_attention_mask,           # [B, max_text_len]
        "labels": padded_labels,                          # [B, max_text_len]
        "actions": padded_actions,                        # [B, max_T_act, action_dim]
        "action_mask": padded_action_mask,                # [B, max_T_act]
        "states": padded_states,                          # [B, max_T_state, state_dim]
        "state_mask": padded_state_mask,                  # [B, max_T_state]
    }


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------
def build_eo_dataloader(
    subset: str = "interleave-temporal",
    split: str = "train",
    batch_size: int = 4,
    num_workers: int = 4,
    tokenizer=None,
    img_size: int = 224,
    max_seq_len: int = 512,
    max_action_len: int = 64,
    action_dim: int = 32,
    state_dim: int = 32,
    streaming: bool = False,
    cache_dir: Optional[str] = None,
    shuffle: bool = True,
    pin_memory: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Build a ready-to-use DataLoader for EO-Data1.5M.

    Example usage::

        from data import build_eo_dataloader

        loader = build_eo_dataloader(
            subset="interleave-temporal",
            batch_size=8,
            num_workers=4,
        )
        for batch in loader:
            images = batch["images"]           # [B, N, 3, 224, 224]
            input_ids = batch["input_ids"]     # [B, 512]
            actions = batch["actions"]         # [B, 64, 7]
            ...
    """
    cfg = EODatasetConfig(
        subset=subset,
        split=split,
        img_size=img_size,
        max_seq_len=max_seq_len,
        max_action_len=max_action_len,
        action_dim=action_dim,
        state_dim=state_dim,
        streaming=streaming,
        cache_dir=cache_dir,
        max_samples=kwargs.pop("max_samples", None),
    )

    dataset = EODataset(config=cfg, tokenizer=tokenizer, **kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and not streaming,
        num_workers=num_workers,
        collate_fn=eo_collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EO-Data1.5M dataloader test")
    parser.add_argument("--subset", default="interleave-temporal")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    pass  # loguru auto-configures

    loader = build_eo_dataloader(
        subset=args.subset,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        shuffle=False,
    )

    print(f"\nDataset size: {len(loader.dataset)}")
    print(f"Batches:      {len(loader)}\n")

    batch = next(iter(loader))

    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key:20s} → {val.shape}  dtype={val.dtype}")
        else:
            print(f"  {key:20s} → {type(val).__name__}: {val}")

    print("\n✓ Dataloader works correctly.")
