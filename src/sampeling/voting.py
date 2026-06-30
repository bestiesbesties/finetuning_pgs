from collections import Counter
from . import jaccard

def aggregate_cluster(preds, cluster):
    cluster = [ preds[i] for i in cluster ] ## grab model group from clustered relative id
    # print("cluster: ", cluster)
    labels = [s[2] for s in cluster]
    label = Counter(labels).most_common(1)[0][0]

    start = min(s[0] for s in cluster)
    end = max(s[1] for s in cluster) ## TODO test with assert
    assert end >= start

    return [start, end, label]

def cluster_spans(J):
    THRESHOLD = 0.33
    clusters = []

    for i in range(J.shape[0]) :

        placed = False

        for c in clusters:
            # print("xm : ", [ J.iloc[i, j] for j in c ])
            if any( J.iloc[i, j] >= THRESHOLD for j in c):
                c.append(i)
                placed = True
                break

        if not placed:
            clusters.append([i])

    return clusters