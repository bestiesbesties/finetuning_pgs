import os
import pandas as pd
import json
from ensemble import aggragate_preds

MODEL_COUNT = 3

def printer(iter):
    for x in iter:
        print(x, "\n")

def bio_to_groups(df):
    spans = []
    start = None
    ent_type = None 
    for i, row in df.iterrows():
        i = row["token_index"]
        if row["label"] == "O" or row["label"] is None:
            if ent_type is not None:
                spans.append((start, i, ent_type))
                start = None
                ent_type = None
            continue
        prefix, typ = row["label"] .split("-", 1)
        if prefix == "B":
            if ent_type is not None:
                spans.append((start, i, ent_type))
            start = i
            ent_type = typ
        elif prefix == "I":
            if ent_type != typ:
                if ent_type is not None:
                    spans.append((start, i, ent_type))
                start = i
                ent_type = typ
    if ent_type is not None: 
        spans.append((start, len(df), ent_type))
    return spans

def groups_to_entities(iter, df):
    return [
        (
            json.loads(df[df["token_index"] == x[0]]["span"].iloc[0])[0],
            json.loads(df[df["token_index"] == x[1]]["span"].iloc[0])[1],
            x[2]
        )
        for x in iter
    ]

def transform_to_expanded_set(intervals):
    s = set()
    for start, end, _ in intervals:
        s.update(range(start, end))
    return s

# def multi_jaccard_similarity(sets):
#     union = set().union(*sets)
#     inter = set.intersection(*sets)
#     return len(inter) / len(union) if union else 1.0

def mean(entropies):
    return sum(entropies) / len(entropies)

def calculate_confidence(text, MAX_ENTROPY):
    df = pd.read_csv(os.path.join("temp", "temp_token_details.csv"))
    model_preds = []
    model_entropies = []
    for model_index in range(0, MODEL_COUNT):
        df_copy = df[df["model_index"] == model_index]
        groups = bio_to_groups(df_copy)
        print("groups: ", groups)
        model_pred = groups_to_entities(groups, df_copy)
        model_preds.append(model_pred)
        entropies = df_copy["entropy"].to_list()
        avg_entropy = mean(entropies)
        model_entropies.append(avg_entropy)
    printer(model_preds)
    printer(model_entropies)
    combined_span_ranges = [ transform_to_expanded_set(model_pred) for model_pred in model_preds ]
    printer(combined_span_ranges)
    jaccard_similarity = aggragate_preds.calculate_mean_sequence_jaccard_simmilarity(model_preds)
    print("jaccard_similarity: ", jaccard_similarity)
    inme = 1 - (sum([model_entropy / MAX_ENTROPY for model_entropy in model_entropies]) / 3)
    print("inme: ", inme)
    # confidence = (0.5 * inme) + (0.5 * jaccard_similarity)
    confidence = inme * jaccard_similarity
    pred = aggragate_preds.aggregate_preds(model_preds)
    return [text, pred, confidence.item()]
