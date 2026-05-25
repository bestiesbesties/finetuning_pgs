import os
import json

def load_config():
    assert os.path.exists("config.json")
    with open("config.json", "r") as f:
        config = json.load(f)
    return config