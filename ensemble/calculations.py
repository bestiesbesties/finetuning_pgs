import torch 

def jaccard_similarity(sets):
    union = set().union(*sets)
    inter = set.intersection(*sets)
    return len(inter) / len(union) if union else 1.0

def calculate_entropy(probs, eps=1e-12):
    probs = probs.clamp(min=eps) ## cast kleine waardes naar minimale waarde
    entropy = -torch.sum(probs * torch.log(probs), dim=0) ## verzwak grote getallen zodat som kleiner wordt als ergens meer zekerheid aan toe is gekend
    return entropy