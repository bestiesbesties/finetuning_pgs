import sqlite3
import json

DB_PATH = "hash.db"
TABLE_NAME = "hashtable2"
OUTPUT_FILE = "export_sqlite.jsonl"

# Kolomnamen in de SQLite-tabel
TEXT_COLUMN = "text"
LABEL_COLUMN = "pred"

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

query = f"""
SELECT {TEXT_COLUMN}, {LABEL_COLUMN}
FROM {TABLE_NAME}
"""

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for row in cursor.execute(query):
        
        labels = row[LABEL_COLUMN]
        if isinstance(labels, str):
            labels = json.loads(labels)

        record = {
            "text": row[TEXT_COLUMN],
            "label": labels
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

conn.close()

print(f"Exported: {OUTPUT_FILE}")