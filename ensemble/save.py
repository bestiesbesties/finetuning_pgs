import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cache_tokens_detail(tokens_detail:list) -> None:
    df = pd.DataFrame(tokens_detail, columns=["model_index", "token_index", "token", "span", "probs", "entropy", "label"])
    df.to_csv(os.path.join("data", "temp_token_details.csv"), index=False)


def export_sequence_detail(sequence_detail: list) -> None:
    fp = os.path.join("data", "temp_sequence_details.csv")
    new_row = pd.DataFrame(
        [sequence_detail],
        columns=["text", "tags", "confidence"]
    )
    if os.path.exists(fp):
        df = pd.read_csv(fp)
        df = pd.concat([new_row, df], ignore_index=True) 
    else:
        df = new_row
    df.to_csv(fp, index=False)
    _save_confidence_dist(df["confidence"])

def _save_confidence_dist(data):
    bins = np.arange(0, 1.1, 0.1)
    fig, ax = plt.subplots()
    ax.hist(data, bins=bins)
    ax.set_xticks(bins)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join("data","temp_confidence_histogram.png"))
    plt.close(fig)
