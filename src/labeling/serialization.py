import os
import pandas as pd
import json
# import pickle
from src import _general

def export_results_iteratively(inputdata):
    block_path = _general.get_last_id("main", "block")
    run_path = _general.get_last_id(block_path, "run")
    filepath = os.path.join(run_path, "results.json")

    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
    else:
        data = {}
    data[str(len(data))] = inputdata
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def to_jsonl(df, path):
    from .hashing import check_df

    if isinstance(df, list):
        df = pd.concat(df, ignore_index=True)
        # pickle.dump(df, open("temp_.pkl", "wb"))
    try:
        df = check_df(df)
    except Exception as e:
        print(e)
    with open(path, "w") as f:
        for i, r in df.iterrows():
            f.write(json.dumps({
                    "text":r["text"],
                    "label": r["pred"],
                    "metadata" : { "changed" : r["changed"] }
                }) + "\n")

def to_df(path):
    with open(path, "r", encoding="utf-8") as f:
        df = pd.DataFrame(data=[json.loads(line) for line in f if line.strip()])
        df.drop(columns=["id", "Comments", "metadata"], errors="ignore", inplace=True)
        df.columns = ["text", "pred"]
        return df