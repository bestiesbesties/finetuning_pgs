MAX_ENTROPY = confidence._calculate_entropy(torch.Tensor([1/len(config["labels"])] * len(config["labels"])))

git clone https://huggingface.co/pdelobelle/robbert-v2-dutch-ner

pip install doccano (in een schone env)
doccano innit
doccano createuser --username admin --password pass
doccano webserver --port 8000 (in terminal 1)
doccano task (in terminal 2)