import os
import pandas as pd
from src._config import config
from src.labeling import serialization

DATA_NAME = "run_0"
path = os.path.join(config["run_data_folder"], DATA_NAME)

hist = pd.read_csv(os.path.join("data", "SE_200_2000_hist.csv"))
serialization.to_jsonl(hist, "testhist.jsonl")