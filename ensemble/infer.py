import os
import json
import requests
import torch
import pandas as pd
ENDPOINT = "http://127.0.0.1:8081/inference"
# TEXT = "Publieke verantwoordingsbegrippen vormen de beoordelingskaders van T. Schillemans en Zaare Goedemans."

def calculate_entropy(probs, eps=1e-12):
    probs = probs.clamp(min=eps) ## cast kleine waardes naar minimale waarde
    entropy = -torch.sum(probs * torch.log(probs), dim=0) ## verzwak grote getallen zodat som kleiner wordt als ergens meer zekerheid aan toe is gekend
    return entropy

CRF_LABEL_COUNT = 8
MAX_ENTROPY = calculate_entropy(torch.Tensor([1/CRF_LABEL_COUNT] * CRF_LABEL_COUNT))

def get_id2label(model_name:str) -> dict:
    model_path = os.path.join("models", model_name, "config.json")
    if os.path.exists(model_path):
        with open(model_path, "r") as f:
            config = json.load(f)
    return config["id2label"]

def save_tokens_detail(tokens_detail:list) -> None:
    df = pd.DataFrame(tokens_detail, columns=["model_index", "token_index", "token", "span", "probs", "entropy", "label"])
    df.to_csv(os.path.join("data", "temp_token_details.csv"), index=False)

def infer() -> list:
    while True:
        text = input(":")
        response = requests.post(ENDPOINT, json = {"text" : text}).json()
        models = response.keys()
        tokens_detail = []
        for model_index, model_name in enumerate(models):
            id2label = get_id2label(model_name)
        
            for token_index, (token, span, logits) in enumerate(zip(response[model_name]["tokens"], response[model_name]["offsets"], response[model_name]["logits"], )):
                logits = torch.Tensor(logits)
                softmax = torch.softmax(logits, dim=0)
                entropy = calculate_entropy(softmax)
                normalized_entropy = entropy / MAX_ENTROPY

                argmax = torch.argmax(logits)
                label = id2label[str(argmax.tolist())]

                tokens_detail.append((model_index, token_index, token, span, [ round(item, 5) for item in softmax.tolist() ], normalized_entropy.item(), label))
        save_tokens_detail(tokens_detail)

# entropies.append(normalized_entropy)
# print("sum(entropies) / len(entropies): ", sum(entropies) / len(entropies))

infer()