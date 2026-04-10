from flask import Flask, request, jsonify 
from transformers import pipeline, Pipeline, AutoModelForTokenClassification, AutoToken

def load_encoder(hf_modelname) -> Pipeline:

    return pipeline(
                "ner", 
                model=hf_modelname,
                aggregation_strategy = "simple"
                )

models = [
    load_encoder("Davlan/bert-base-multilingual-cased-ner-hrl"),
    load_encoder("FacebookAI/xlm-roberta-large-finetuned-conll03-english"),
    load_encoder("51la5/roberta-large-NER")
]

def inference(text:str):
    return models[i](text)

app = Flask(__name__)

@app.route("/inference", methods = ["POST"])
def inference_endpoint():
    data = request.get_json()
    print("data: ", data)
    text = data.get("text", "")
    print("text: ", text)
    ents = [ model(text) for model in models ]

    print("ents:", ents)
    print("len(ents): ", len(ents))

    structure = [{
        "label" : ent["entity_group"],
        "start_offset" : ent["start"] ,
        "end_offset" : ent["end"],
        "confidence" : ent["score"]
    }
    for ent in ents
    ]
    ##?universeel terwijl tokenizer in transformer?...
    print("structure: ", structure)
    return jsonify(structure)

if __name__ == "__main__":
    print("Launching inference webserver")
    app.run(port=8081)