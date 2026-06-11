import os
import hashlib
import sqlite3
import ast
import json
from .serialization import to_df

from src.labeling.serialization import to_df, to_jsonl

conn = sqlite3.connect("hash.db")
cur = conn.cursor()
cur.execute("""
    CREATE TABLE IF NOT EXISTS hashtable (
        hash TEXT PRIMARY KEY,
        pred TEXT
    )
""")

def _upsert(text, pred):
    hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    try:
        cur.execute("INSERT INTO hashtable (hash, pred) VALUES (?, ?)", [hash, json.dumps(pred)])
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def _check(text):
    cur.execute(
        "SELECT pred FROM hashtable WHERE hash = ? LIMIT 1",
        [hashlib.sha256(text.encode("utf-8")).hexdigest()]
    )
    row = cur.fetchone()
    return row

def upsert_jsonl(path):
    df = to_df(path)
    for i, v in df.iterrows():
        _upsert(v["text"], v["pred"])

def check_jsonl(path):
    df = to_df(path)
    print(df)

    for i, v in df.iterrows():
        row = _check(v["text"])
        if row is not None:
            x = json.loads(row[0])
            y = df.loc[i, "pred"] 
            if x != y:
                print("updating: ", i)
                df.loc[i, "pred"] =  x
            
    to_jsonl(df, path)

# upsert_jsonl("adminultra.jsonl")
check_jsonl("main/block_0/run_0/export.jsonl")
