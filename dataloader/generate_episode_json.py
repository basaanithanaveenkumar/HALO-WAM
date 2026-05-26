import pandas as pd
from pathlib import Path

meta_dir = Path('/home/ha/datasets/airoa-moma/meta')
files = sorted(meta_dir.glob('episodes/**/*.parquet'))

df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

out = Path('/home/ha/datasets/airoa-moma/episodes.jsonl')
df.to_json(out, orient='records', lines=True, force_ascii=False)

print(f"Written {len(df)} rows to {out} ({out.stat().st_size} bytes)")