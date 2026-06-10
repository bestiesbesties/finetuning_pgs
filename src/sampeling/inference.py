import os
import json
import requests
from .._config import config

def infer(text:str) -> list:
    return requests.post(config["inference_endpoint"], json = {"text" : text}).json()
