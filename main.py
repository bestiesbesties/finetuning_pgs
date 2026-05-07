import os
import torch
import pandas as pd
from ensemble import infer, calculations, confidence, save

CRF_LABEL_COUNT = 8
MAX_ENTROPY = calculations.calculate_entropy(torch.Tensor([1/CRF_LABEL_COUNT] * CRF_LABEL_COUNT))

df = pd.read_csv(os.path.join("data", "hashed_chunks.csv"))

while True:
    text = input(":")
    if text == "":
        text = df.sample(1)["text"].to_list()[0]
    tokens_detail = infer .infer(text)
    sequence_details  = confidence.calculate_confidence(text, MAX_ENTROPY)
    save.export_sequence_detail(sequence_details)