import os
import hashlib
import sqlite3
import ast
import json
from .serialization import to_df, to_jsonl

# from src.labeling.serialization import to_df, to_jsonl

conn = sqlite3.connect("hash.db")
cur = conn.cursor()
cur.execute("""
    CREATE TABLE IF NOT EXISTS hashtable2 (
        hash TEXT PRIMARY KEY,
        pred TEXT,
        text TEXT
    )
""")

def _upsert(text, pred):
    hash_value = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cur.execute("""
        INSERT INTO hashtable2 (hash, text, pred)
        VALUES (?, ?, ?)
        ON CONFLICT(hash) DO UPDATE SET
            text = excluded.text,
            pred = excluded.pred
    """, (hash_value, text, json.dumps(pred)))
    conn.commit()
    return True

def upsert_df(df):
    for i, v in df.iterrows():
        _upsert(v["text"], v["pred"])
    return df

def _check(text):
    cur.execute(
        "SELECT pred FROM hashtable2 WHERE hash = ? LIMIT 1",
        [hashlib.sha256(text.encode("utf-8")).hexdigest()]
    )
    row = cur.fetchone()
    if row:
        return ast.literal_eval(row[0]) 
    else:
        return None

def check_df(df):
    # df.to_csv("temp_.csv")
    df["changed"] = False
    for i, v in df.iterrows():
        row = _check(v["text"])
        if row is not None:
            # print("updating: ", i, row)
            df.at[i, "pred"] = row
            df.at[i, "changed"] = True                                                                                                                                                                                              
    return df
