import numpy as np

def _similarity(a, b):
    return len(a & b) / len(a | b) if (a or b) else 1.0

def _matrix(sequence_detail): ## TODO overhead type
    data = []
    for model in list(sequence_detail.keys()):
        data += [ _expand_span(_offset_span(group, sequence_detail[model]["offsets"])) for group in sequence_detail[model]["groups"] ]
    n = len(data)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # print("data[i]: ", data[i])
            # print("data[j]: ", data[j])
            J[i, j] = _similarity(data[i], data[j])
    return J

def _get_matrix_indeces(sequence_detail):
    start = 0
    indeces = []
    for sub_model in list(sequence_detail.keys()):
        end = start + len(sequence_detail[sub_model]["groups"])
        indeces.append((start, end))
        start = end
    return indeces

def _drop_indeces(iter, drop):
    return [ x for i, x in enumerate(iter) if i not in drop ]

def _offset_span(span, offset):
    s_t ,e_t, p = span
    s_c = int(offset[s_t][0])
    e_c = int(offset[e_t][1])
    assert e_c > s_c
    return [s_c, e_c, p]

def _expand_span(group):
    return set(range(group[0], group[1]))
