import os
import pandas as pd
from .._config import config
from sklearn.model_selection import train_test_split
from src.sampeling import confidence, histing

def undersample(path):
    df = pd.read_csv(path, index_col=0)
    return df.sample(config["block_undersample_size"], random_state=config["uniform_random_state"])

def train_test(df):
    if isinstance(df, str):
        df = pd.read_csv(df)
    train, test = train_test_split(
        df,
        test_size=config["block_test_size"],
        random_state=config["uniform_random_state"]
    )
    return train, test

def train_eval_test(df):
    df = df.sample(frac=1, random_state=config["uniform_random_state"])
    train, temp = train_test_split(
        df,
        test_size=0.3,
        random_state=config["uniform_random_state"]
    )
    target_len = int(len(temp) / 2)
    test = temp.iloc[:target_len + 1]
    eval = temp.iloc[target_len:]
    return train, eval, test

def prune(df):
    print("len(df): ", len(df))

    ## chars
    df["chars_length"] = df["text"].str.len()
    chars_median = int(df["chars_length"].median())
    print("chars_median: ", chars_median)

    ## words
    df["words_length"] = df["text"].str.split().str.len()
    words_median = int(df["words_length"].median())
    print("words_median: ", words_median)

    ## filter 
    df = df[df["words_length"] <= words_median * 2]
    print("len(df): ", len(df))
    df.drop(["chars_length", "words_length"], axis=1, inplace=True)
    return df

def transform_pool(data_path, run_path, n, pipeline):
    pool = pd.read_csv(data_path, index_col=0)
    pipeline = pipeline.lower() if isinstance(pipeline, str) else ""
    if pipeline == "c_h" or pipeline == "c_r":
        print("confidence routing")
        results = pool["text"].apply(lambda x: confidence.route(x, os.path.join(run_path, "model")))
        pool["conf"] = results.str[0]
        pool["pred"] = results.str[1]
        if pipeline == "c_h":
            print("histing")
            sample = histing.stratified_sample(pool, n)
        elif pipeline == "c_r":
            print("random sampeling")
            sample = pool.sample(n=n, random_state=config["uniform_random_state"])
    else:
        print("empty random sampeling")
        sample = pool.sample(n=n, random_state=config["uniform_random_state"])
        sample["pred"] = [[] for _ in range(len(sample))]
    print("dropping: ", sample.index)
    pool.drop(sample.index, inplace=True)
    pool.to_csv(data_path, index=True) 
    return sample

def _create_bio_tags(row, tokenizer, model): ## TODO load from global/module dependency inject
    ## !! TODO ambiguous source match code with TPS
    text = row["text"]
    entities = row["pred"]

    encoding = tokenizer(
        text,
        truncation=True,
        return_offsets_mapping=True
    )

    offsets = encoding["offset_mapping"]
    labels = [model.config.label2id["O"]] * len(offsets)

    for idx, (start, end) in enumerate(offsets):
        if start == end:
            labels[idx] = -100

    for ent_start, ent_end, ent_type in entities:
        first = True

        for idx, (tok_start, tok_end) in enumerate(offsets):

            if tok_start == tok_end:
                continue

            if tok_start >= ent_start and tok_end <= ent_end:
                prefix = "B" if first else "I"
                labels[idx] = model.config.label2id[f"{prefix}-{ent_type}"]
                first = False

    encoding["labels"] = labels
    encoding.pop("offset_mapping")

    return encoding

def tokenize_data(data:list, tokenizer, model): ## TODO load from global/module dependency inject
    print(data.head())
    tokenized_data = [_create_bio_tags(row, tokenizer, model) for _, row in data.iterrows()]

    # for i, seq in enumerate(tokenized_data[:2]):
    #     print("Index", i, "Value", seq)
    #     assert len(seq["input_ids"]) == len(seq["labels"]), f"len mismatch at train {i}"

    return tokenized_data