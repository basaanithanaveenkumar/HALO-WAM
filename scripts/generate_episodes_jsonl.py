"""
generate_episodes_jsonl.py
--------------------------
Scans the airoa-moma local clone and writes a fresh episodes.jsonl that is
compatible with AiroaMomaDataset in dataloader/airoa_moma_dataset.py.

The HF dataset uses file-NNN.mp4 naming inside chunk-XXX directories, NOT
episode_YYYYYY.mp4.  This script resolves the real video paths, reads the
parquet episode metadata, and emits one JSON line per usable episode.

Usage
-----
    python scripts/generate_episodes_jsonl.py \
        --data_root /home/ha/datasets/airoa-moma \
        [--camera head] \
        [--out_path /home/ha/datasets/airoa-moma/episodes.jsonl]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def find_video_files(data_root: Path, camera: str) -> dict[int, Path]:
    """
    Return a mapping  episode_index -> absolute video path.

    The actual on-disk layout is:
        <data_root>/videos/observation.image.<camera>/chunk-XXX/file-NNN.mp4

    The HF parquet assigns episode_index sequentially across chunks:
        chunk-000 holds episode indices 0 .. 999  (file-000 .. file-999)
        chunk-001 holds episode indices 1000..1999
        ...
    So  episode_index = chunk * episodes_per_chunk + file_number
    where episodes_per_chunk = 1000 (HF convention).
    """
    EPISODES_PER_CHUNK = 1000
    videos_root = data_root / "videos" / f"observation.image.{camera}"

    if not videos_root.is_dir():
        print(f"[WARN] Camera directory not found: {videos_root}", file=sys.stderr)
        return {}

    ep_to_path: dict[int, Path] = {}
    chunk_dirs = sorted(videos_root.glob("chunk-*"))
    if not chunk_dirs:
        print(f"[WARN] No chunk-* directories under {videos_root}", file=sys.stderr)
        return {}

    for chunk_dir in chunk_dirs:
        m = re.search(r"chunk-(\d+)", chunk_dir.name)
        if not m:
            continue
        chunk_idx = int(m.group(1))
        for mp4 in sorted(chunk_dir.glob("file-*.mp4")):
            fm = re.search(r"file-(\d+)", mp4.stem)
            if not fm:
                continue
            file_num = int(fm.group(1))
            ep_idx = chunk_idx * EPISODES_PER_CHUNK + file_num
            ep_to_path[ep_idx] = mp4

    return ep_to_path


def load_parquet_episodes(data_root: Path) -> list[dict]:
    """Read all meta/episodes/**/*.parquet and return list of row dicts."""
    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] pandas is required: pip install pandas pyarrow", file=sys.stderr)
        sys.exit(1)

    ep_dir = data_root / "meta" / "episodes"
    if not ep_dir.is_dir():
        print(f"[WARN] No meta/episodes directory found at {ep_dir}", file=sys.stderr)
        return []

    frames = []
    for pq in sorted(ep_dir.rglob("*.parquet")):
        df = pd.read_parquet(pq)
        frames.append(df)

    if not frames:
        print("[WARN] No parquet files found under meta/episodes/", file=sys.stderr)
        return []

    import pandas as pd
    combined = pd.concat(frames, ignore_index=True)

    # Normalise column names to snake_case
    combined.columns = [c.lower().replace(" ", "_") for c in combined.columns]

    rows = []
    for _, row in combined.iterrows():
        rows.append(row.to_dict())
    return rows


def build_jsonl(
    data_root: Path,
    camera: str,
    out_path: Path,
    overwrite: bool = True,
) -> int:
    """Build and write episodes.jsonl.  Returns number of episodes written."""
    ep_to_path = find_video_files(data_root, camera)
    print(f"[INFO] Found {len(ep_to_path)} video files for camera='{camera}'")

    parquet_rows = load_parquet_episodes(data_root)
    print(f"[INFO] Loaded {len(parquet_rows)} episode rows from parquet metadata")

    # Build a lookup by episode_index from parquet
    parquet_by_idx: dict[int, dict] = {}
    for row in parquet_rows:
        # Common column names in HF lerobot datasets
        idx = int(
            row.get("episode_index", row.get("episode_id", row.get("index", -1)))
        )
        if idx >= 0:
            parquet_by_idx[idx] = row

    written = 0
    records = []

    # Use the union of video files and parquet rows
    all_indices = sorted(set(ep_to_path.keys()) | set(parquet_by_idx.keys()))

    for ep_idx in all_indices:
        vid = ep_to_path.get(ep_idx)
        if vid is None:
            # Have parquet metadata but no video — skip
            continue

        meta = dict(parquet_by_idx.get(ep_idx, {}))

        # Mandatory fields the dataloader reads
        meta["episode_index"] = ep_idx

        # video_path relative to data_root (so the dataloader can reconstruct it)
        meta["video_rel_path"] = str(vid.relative_to(data_root))

        # length: prefer parquet value, fall back to 0 (loader handles missing)
        if "length" not in meta or meta["length"] is None:
            meta["length"] = 0

        # tasks / short_horizon_task
        if "tasks" not in meta:
            meta["tasks"] = meta.get("task", None)
        if isinstance(meta.get("tasks"), str):
            meta["tasks"] = [meta["tasks"]]

        # Serialise — convert any non-JSON-native types
        clean: dict = {}
        for k, v in meta.items():
            if hasattr(v, "item"):           # numpy scalar
                v = v.item()
            elif hasattr(v, "tolist"):       # numpy array
                v = v.tolist()
            elif isinstance(v, float) and (v != v):  # NaN
                v = None
            clean[k] = v

        records.append(clean)
        written += 1

    if out_path.exists() and not overwrite:
        print(f"[WARN] {out_path} already exists; use --overwrite to replace it.")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[INFO] Wrote {written} episodes → {out_path}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate episodes.jsonl for AiroaMomaDataset")
    ap.add_argument("--data_root", default="/home/ha/datasets/airoa-moma",
                    help="Path to the local airoa-moma clone")
    ap.add_argument("--camera", default="head", choices=["head", "hand"],
                    help="Camera to use for video path generation")
    ap.add_argument("--out_path", default=None,
                    help="Output path for episodes.jsonl (default: <data_root>/episodes.jsonl)")
    ap.add_argument("--overwrite", action="store_true", default=True,
                    help="Overwrite existing episodes.jsonl")
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        print(f"[ERROR] data_root not found: {data_root}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out_path) if args.out_path else data_root / "episodes.jsonl"

    n = build_jsonl(data_root, args.camera, out_path, overwrite=args.overwrite)
    if n == 0:
        print("[ERROR] No episodes written — check your data_root and camera argument.")
        sys.exit(1)


if __name__ == "__main__":
    main()
