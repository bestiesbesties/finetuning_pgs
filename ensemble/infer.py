import os
import json
import requests
import torch
from . import calculations, save

ENDPOINT = "http://127.0.0.1:8081/inference"

def get_id2label(model_name:str) -> dict:
    model_path = os.path.join("models", model_name, "config.json")
    if os.path.exists(model_path):
        with open(model_path, "r") as f:
            config = json.load(f)
    return config["id2label"]

def infer(text:str) -> list:
    response = requests.post(ENDPOINT, json = {"text" : text}).json()
    models = response.keys()
    tokens_detail = []
    for model_index, model_name in enumerate(models):
        id2label = get_id2label(model_name)
    
        for token_index, (token, span, logits) in enumerate(zip(response[model_name]["tokens"], response[model_name]["offsets"], response[model_name]["logits"], )):
            logits = torch.Tensor(logits)
            softmax = torch.softmax(logits, dim=0)
            entropy = calculations.calculate_entropy(softmax)

            argmax = torch.argmax(logits)
            label = id2label[str(argmax.tolist())]

            tokens_detail.append((model_index, token_index, token, span, [ round(item, 5) for item in softmax.tolist() ], entropy.item(), label))
    save.cache_tokens_detail(tokens_detail)
    return tokens_detail