import os
from importlib import reload
import pandas as pd
from src import _general

df = pd.read_csv(os.path.join("data", "parsed-pgs33.csv" ))
main_path = _general.initialize_main(df)
block_path = _general.initialize_block(main_path, 0)

from src import _general
block_path = "main/block_0"
pipeline = "c_h"
pipeline
run_path = _general.initialize_run(block_path, 9, pipeline=pipeline)


# from src.labeling import hashing, serialization
# df = serialization.to_df("admin copy.jsonl")
# hashing.upsert_df(df)