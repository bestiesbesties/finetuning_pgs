import os
import pandas as pd
import json
import ast
from .._config import config

def to_jsonl(df, path):
    if isinstance(df, list):
        df = pd.concat(df)
    with open(path, "w") as f:
        for i, r in df.iterrows():
            print("i:", i)
            # print("r['pred']", r['pred'])
            tags = r["pred"]
            # tags = ast.literal_eval(r["pred"])  ## of split, afhankelijk van formaat
            f.write(json.dumps({"text": r["text"], "label": tags}) + "\n")

def to_df(path):
    with open(path, "r", encoding="utf-8") as f:
        return pd.DataFrame(data=[json.loads(line) for line in f if line.strip()]) 