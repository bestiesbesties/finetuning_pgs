import os
from ._config import config
from .data import data_helper
import glob
import pandas as pd
from .labeling import serialization, hashing
from .sampeling import inference
from datasets import Dataset

def get_last_id(path, addon):
    count = 0
    path = os.path.join(path, addon)
    while os.path.isdir(f"{path}_{count}"):
        count += 1
    return f"{path}_{count - 1 if count > 0 else count}"

def _save_data_frames(x, mapping):
    for k, v in mapping.items():
        v.to_csv(os.path.join(x, k + ".csv"), index=True)

def initialize_main(df):
    x = os.path.join(config["folder_structure"][0]) 
    os.makedirs(x, exist_ok=True)
    df = data_helper.prune(df)
    train, eval, test = data_helper.train_eval_test(df)

    _save_data_frames(x, {
        "source" : df,
        "train" : train,
        "eval" : eval,
        "test" : test
        })
    return x

def initialize_block(main_path, block_id):
    x = os.path.join(main_path, config["folder_structure"][1] + f"_{block_id}")
    os.makedirs(x , exist_ok=True)
    df = data_helper.undersample(os.path.join(main_path, "train.csv"))
    train, eval = data_helper.train_test(df)
    _save_data_frames(x, {
        "train" : train,
        "train_pool" : train,
        "eval" : eval,
        "eval_pool" : eval
        })
    return x

def initialize_run(block_path, run_id, pipeline):
    x = os.path.join(block_path, config["folder_structure"][2] + f"_{run_id}")
    os.makedirs(x , exist_ok=True)
    base_model_path = "." + os.sep + config["base_model_name"]

    print("initializing model")
    from .finetuning import model_helper, trainer_helper
    tokenizer, model = model_helper.load_model(base_model_path, config["labels"], reinitialize=True)
    if run_id > 0:
        print("train model")
        train_dataset, test_dataset = accum_finetuning_datasets(tokenizer, model, block_path)
        trainer_helper.train(tokenizer, model, train_dataset, test_dataset)
    model_helper.save_model(x, tokenizer, model)       

    inference.switch(os.path.join(x, "model"))
    hist = data_helper.transform_pool(os.path.join(block_path, "train_pool.csv"), x, n=80, pipeline=pipeline)
    eval = data_helper.transform_pool(os.path.join(block_path, "eval_pool.csv"), x, n=20, pipeline=None)

    serialization.to_jsonl([hist, eval], os.path.join(x, "export.jsonl"))
    return x

def accum_finetuning_datasets(tokenizer, model, block_path):
    dfs = [ hashing.upsert_df(serialization.to_df(path)) for path in  _search_all_run_ids(block_path) ]
    train = pd.concat([ df.head(80) for df in dfs ])
    print("len(train): ", len(train))
    eval = pd.concat([ df.tail(20) for df in dfs ])
    print("len(eval): ", len(eval))
    train_dataset = Dataset.from_list(data_helper.tokenize_data(train, tokenizer, model))
    test_dataset = Dataset.from_list(data_helper.tokenize_data(eval, tokenizer, model))
    return train_dataset, test_dataset

def _search_all_run_ids(block_path):
    pattern = os.path.join(block_path, config["folder_structure"][2] + "_*", "admin.jsonl")
    run_ids = glob.glob(pattern)
    print("run_ids: ", run_ids)
    return run_ids