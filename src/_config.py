import os
import json

def _load_config():
    assert os.path.exists("config.json")
    with open("config.json", "r") as f:
        config = json.load(f)
    return config

config = _load_config()

def _ensure_folderpath(folderpath):
    if not os.path.exists(folderpath):
        print("folderpath does not exist: ", folderpath)
        os.makedirs(folderpath)

_ensure_folderpath(os.path.join("models", "built"))
_ensure_folderpath(os.path.join("temp"))