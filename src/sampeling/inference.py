import os
import json
import requests
from .._config import config

def infer(text):
    return requests.post(config["inference_endpoint"] + "inference", json = {"text" : text}).json()

def switch(path):
    requests.post(config["inference_endpoint"] + "switch", json = {"text" : path})