import os
import json

def load_config():
    assert os.path.exists("config.json")
    with open("config.json", "r") as f:
        config = json.load(f)
    config["label_meta"]["id2label"] = {int(k): v for k, v in config["label_meta"]["id2label"].items()}
    return config