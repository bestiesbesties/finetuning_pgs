import json

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def _create_bio_tags(example, tokenizer, model): ## TODO load from global/module dependency inject
    ## !! TODO ambiguous source match code with TPS
    text = example["text"]
    entities = example["label"]

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
    print("Tokenizing")
    tokenized_data = [_create_bio_tags(seq, tokenizer, model) for seq in data]

    for i, seq in enumerate(tokenized_data[:2]):
        print("Index", i, "Value", seq)
        assert len(seq["input_ids"]) == len(seq["labels"]), f"len mismatch at train {i}"

    return tokenized_data
