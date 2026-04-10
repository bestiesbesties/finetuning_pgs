from transformers import pipeline, Pipeline, AutoModelForTokenClassification, AutoTokenizer
import os

def load_encoder(model_name) -> Pipeline:
    model_path = "./models/" +  model_name
    print("model_path: ", model_path)
    if not os.path.exists(model_path):
        raise RuntimeError(f"'{model_name}' not in ./models folder")
    encoder = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    return pipeline(
                "ner", 
                model=encoder,
                tokenizer=tokenizer,
                aggregation_strategy = None
                )

models = [
    load_encoder("bert-base-multilingual-cased-ner-hrl"),
    # load_encoder("xlm-roberta-large-finetuned-conll03-english"),
    # load_encoder("wikineural-multilingual-ner")
]

# import pandas as pd
# df = pd.read_csv(os.path.join("data", "hashed_chunks.csv"))

# def inf(text):
#     ents_combined =
#     print(ents_combined)
#     return 
    # ents_per = []
    # for ents in ents_combined:
    #     ents_per.append([ ent for ent in ents if ent["entity_group"] == "PER" ])
    # for predict in ents_per:
    #     print(predict)


# def confidence():



    # print("ents: ", ents_combined)

# while True:
    # text = df["text"].sample(1).iloc[0]
text = "De begrippen vormen de beoordelingskaders van T. Schillemans en Zaare Goedemans."
ensemble_preds = [ model(text) for model in models ]
for preds in ensemble_preds:
    print("model 1")
    for pred in preds:
        print("pred:", pred)



# import spacy 
# model_name = "nl_core_news_lg"

# nlp = spacy.load(model_name)
# text = "De begrippen vormen een beoordelingskader met T. Schillemans en Zaare Goedemans."
# sdoc = nlp(text)

# for ent in sdoc.ents:
#     print(type(ent))
#     print(ent.ent_id)
#     print(ent.label_)
#     print(ent.start_char)
#     print(ent.end_char)
#     print(ent.text)