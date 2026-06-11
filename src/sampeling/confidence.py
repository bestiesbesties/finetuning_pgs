import os
import json
import torch
import pandas as pd
import numpy as np
from . import inference, jaccard, voting
from .._config import config

def _get_id2label(model_path:str) -> dict:
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    return config["id2label"]

def _calculate_entropy(probs, eps=1e-12):
    probs = probs.clamp(min=eps) ## cast kleine waardes naar minimale waarde
    entropy = -torch.sum(probs * torch.log(probs), dim=0) ## verzwak grote getallen zodat som kleiner wordt als ergens meer zekerheid aan toe is gekend
    return entropy

def _token_detail(logits, id2label):
    logits = torch.Tensor(logits)
    softmax = torch.softmax(logits, dim=0)
    print("softmax: ", softmax)
    entropy = _calculate_entropy(softmax)
    print("entropy: ", entropy)

    argmax = torch.argmax(logits)
    print("argmax: ", argmax)
    token_pred = id2label[str(argmax.tolist())] ## TODO value instead of tolist 

    return entropy, token_pred

## TODO un-inject FROM TPS
def _group_bio(preds):
    spans = []
    start = None
    current_tag = None

    for i, label in enumerate(preds):
        if label == "O":
            if start is not None:
                spans.append((start, i - 1, current_tag))
                start = None
                current_tag = None
            continue

        tag = label.split("-", 1)[-1]  # DOC, ACT, etc.

        if start is None:
            start = i
            current_tag = tag
        elif tag != current_tag:
            spans.append((start, i - 1, current_tag))
            start = i
            current_tag = tag

    if start is not None:
        spans.append((start, len(preds) - 1, current_tag))

    return spans

def _safe_mean(x, safe=0) :
    return sum(x) / len(x) if sum(x) > 0 else safe

def _inverse_model_similairity(indeces, J): ## TODO further granularity
        exp_indeces = list(range(indeces[0], indeces[1])) ## TODO exp to up
        print("exp_indeces: ", exp_indeces)
        x = [ J[j].to_list() for j in exp_indeces ] ## Take own rows (square matrix)
        print("x: ", x)
        y = [ jaccard._drop_indeces(x, exp_indeces) for x in x ] ## Exclude own similairity
        print("y: ", y)
        z = [_safe_mean(y) for y in y] ## Take similairity mean 
        print("z: ", z)
        return _safe_mean(z) ## Take model mean

def _sequence_detail(text, id2label):
    sequence_detail = inference.infer(text)

    for model in sequence_detail.keys():
        print("model: ", model)
        
        _t = [ _token_detail(logit, id2label) for logit in sequence_detail[model]["logits"] ]
        sequence_detail[model]["entropies"] = [v[0].item() for v in _t]
        sequence_detail[model]["preds"] = [v[1] for v in _t]
        # sequence_detail[model]["groups"] = [ _expand_span(_offset_span(span, sequence_detail[model]["offsets"])) for span in _group_bio(sequence_detail[model]["preds"]) ]
        sequence_detail[model]["groups"] = [ group for group in _group_bio(sequence_detail[model]["preds"]) ]

    return sequence_detail

def route(text, model_path):
    sequence_detail = _sequence_detail(text, _get_id2label(model_path))
    print("sequence_detail: ", sequence_detail)
    print(_get_id2label(model_path))
    J = pd.DataFrame(jaccard._matrix(sequence_detail))
    print(J)
    sequence_indeces = jaccard._get_matrix_indeces(sequence_detail)
    ## CONF
    sequence_similairity = _safe_mean([ _inverse_model_similairity(model_indeces, J) for model_indeces in sequence_indeces ], safe=1)
    model_entropies = [ _safe_mean(sequence_detail[model]["entropies"]) for model in list(sequence_detail.keys()) ]
    normalized_sequence_entropy = sum([model_entropy / config["max_entropy"] for model_entropy in model_entropies]) / len(list(sequence_detail.keys())) 
    print("( 1 - normalized_sexquence_entropy ): ", ( 1 - normalized_sequence_entropy ))
    print("sequence_similairity: ", sequence_similairity)
    conf =  ( 1 - normalized_sequence_entropy ) *  sequence_similairity
    ## PRED
    sequence_preds = [ group for key in sequence_detail.keys() for group in sequence_detail[key]["groups"] ]
    print("sequence_preds: ", sequence_preds)
    clusters = voting.cluster_spans(J)
    print("clusters: ", clusters)
    pred = [ voting.aggregate_cluster(sequence_preds, cluster) for cluster in clusters ]
    print("pred: ", pred)
    pred = [ jaccard._offset_span(x, sequence_detail["300A"]["offsets"]) for x in pred ]
    print("conf: ", conf)
    print("pred: ", pred)
    return (conf, pred)