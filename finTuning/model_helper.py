from transformers import AutoModelForTokenClassification, AutoTokenizer

def load_model(path:str, labels:list, reinitialize=False):
    tokenizer = AutoTokenizer.from_pretrained(path)
    ## TODO use RobertalForTokenClassification.from_pretrained?
    if reinitialize: ## TODO add '.' to pathstring?
        model = AutoModelForTokenClassification.from_pretrained(path, num_labels=len(labels) , ignore_mismatched_sizes=True)
        model = _update_config(model, labels)
        model = _freeze_unfreeze(model, 3)
    else:
        model = AutoModelForTokenClassification.from_pretrained(path)
    return tokenizer, model

def save_model(path, tokenizer, trainer):
    tokenizer.save_pretrained("./models/ner_finetuned")
    trainer.save_model("./models/ner_finetuned")

def _update_config(model, labels:list):
    print("hidden_size", model.config.hidden_size)
    print("num_labels", len(labels))
    print("Setting id2label & label2id")
    model.config.id2label = {i: label for i, label in enumerate(labels)}
    model.config.label2id = {label: i for i, label in enumerate(labels)}
    return model

def _freeze_unfreeze(model, layer_n):
    num_roberta_parameters = 0
    for param in model.roberta.parameters():
        num_roberta_parameters += 1
        param.requires_grad = False
    print("num_roberta_parameters: ", num_roberta_parameters)

    num_roberta_encoder_layer_parameters = 0
    for layer in model.roberta.encoder.layer[-layer_n:]:
        for param in layer.parameters():
            num_roberta_encoder_layer_parameters += 1
            param.requires_grad = True
    print("num_roberta_encoder_layer_parameters: ", num_roberta_encoder_layer_parameters)

    num_classifier_parameters = 0
    for param in model.classifier.parameters():
        num_classifier_parameters += 1
        param.requires_grad = True
    print("num_classifier_parameters: ", num_classifier_parameters)
    return model
