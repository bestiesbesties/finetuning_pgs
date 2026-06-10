import pandas as pd
import os
import json
import ast

DATA_FOLDER = "temp"
DATA_NAME = "temp_sequence_details"

df = pd.read_csv(os.path.join(DATA_FOLDER, DATA_NAME + ".csv"))

with open(os.path.join(DATA_FOLDER, DATA_NAME + ".jsonl"), "w") as f:
    for i, r in df.iterrows():
        print("i:", i)
        tags = ast.literal_eval(r["tags"])  # of split, afhankelijk van formaat
        f.write(json.dumps({"text": r["text"], "label": tags}) + "\n")