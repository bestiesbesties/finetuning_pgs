from collections import Counter

def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0

def cluster_spans(spans, threshold=0.5):
    clusters = []

    for s in spans:
        placed = False

        for c in clusters:
            if any(iou(s, t) > threshold for t in c):
                c.append(s)
                placed = True
                break

        if not placed:
            clusters.append([s])

    return clusters

def aggregate_cluster(cluster):
    labels = [s[2] for s in cluster]
    label = Counter(labels).most_common(1)[0][0]

    start = min(s[0] for s in cluster)
    end = max(s[1] for s in cluster)

    return [start, end, label]

# model_preds = [
#     [(18, 31, 'PER'), (88, 97, 'PER')],

#     [(18, 31, 'PER')],

#     [(18, 31, 'PER'), (88, 109, 'PER')] 
# ]


def safe_mean(x):
    return sum(x) / len(x) if sum(x) > 0 else 0

def calculate_mean_sequence_jaccard_simmilarity(model_preds):
    model_indeces = [0, 1 ,2]
    model_sims = []
    ## model
    for model_index, _model_preds in enumerate(model_preds):
        print("model_index:", model_index)
        counter_model_indeces = [_model_index for _model_index in model_indeces if _model_index != model_index]
        counter_model_sims = []
        ## counter model
        for counter_model_index in counter_model_indeces:
            print("counter_model_index:", counter_model_index)
            counter_preds = model_preds[counter_model_index]
            print("counter_preds:", counter_preds)
            sequence_sims = []
            ## token
            for model_pred in _model_preds:
                token_sims = [iou(model_pred, counter_pred) for counter_pred in counter_preds]
                token_sims_mean = safe_mean(token_sims)
                print("model_pred:", model_pred, "token_sims:", token_sims, "token_sims_mean:", token_sims_mean)
                sequence_sims.append(token_sims_mean)
            if not _model_preds and not counter_preds:
                print("appended:", 1)
                sequence_sims.append(1)
            sequence_sims_mean = safe_mean(sequence_sims)
            print("sequence_sims_mean:", sequence_sims_mean)
            counter_model_sims.append(sequence_sims_mean)
    

        counter_model_sims_mean = safe_mean(counter_model_sims)
        print("counter_model_sims_mean: ", sum(counter_model_sims) / len(counter_model_sims))
        model_sims.append(counter_model_sims_mean)
    model_sims_mean = safe_mean(model_sims)
    print("model_sims_mean:", model_sims_mean)
    return model_sims_mean

def aggregate_preds(model_preds):
    all_spans = [s for model in model_preds for s in model]
    print("all_spans :", all_spans)
    clusters = cluster_spans(all_spans, threshold=0.5)
    final_spans = [aggregate_cluster(c) for c in clusters]
    return final_spans
