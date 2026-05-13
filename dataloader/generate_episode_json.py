import pandas as pd, json
from pathlib import Path

meta_dir = Path('/home/ha/datasets/airoa-moma/meta')

# Read episode parquets
files = sorted(meta_dir.glob('episodes/**/*.parquet'))
print('Found:', files)
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
print('Columns:', df.columns.tolist())
print(df.head(2))

# Write episodes.jsonl
out = Path('/home/ha/datasets/airoa-moma/episodes.jsonl')
with open(out, 'w') as f:
    for row in df.to_dict(orient='records'):
        f.write(json.dumps(row) + '\n')
print(f'Written {len(df)} episodes to {out}')
"