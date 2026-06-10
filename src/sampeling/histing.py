import pandas as pd
import numpy as np

def stratified_sample(df, N):
    df_conf = df["conf"].sort_values()
    n_low = int(round(N * 0.7))
    n_high = int(round(N * 0.1))
    n_rand = N - n_low - n_high
    low_indeces = df_conf.head(n_low).index
    high_indeces = df_conf.tail(n_high).index

    remaining = df_conf.iloc[n_low : len(df_conf )- n_high]
    rand_indeces = remaining.sample(n=min(n_rand, len(remaining))).index

    final_idx = low_indeces.union(high_indeces).union(rand_indeces)
    final_idx = pd.Index(final_idx).to_series().sample(n=N).index

    return df.loc[final_idx].reset_index(drop=True)