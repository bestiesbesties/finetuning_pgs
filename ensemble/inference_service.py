from flask import Flask, request, jsonify 
from transformers import pipeline, Pipeline

def load_model(hf_modelname) -> Pipeline:
    return pipeline(
                "ner", 
                model=hf_modelname,
                aggregation_strategy="simple"
                )

# bert = load_model("Davlan/bert-base-multilingual-cased-ner-hrl") ##GOOGLE
# roberta = load_model("LecJackS/xlm-roberta-base-finetuned-conll2003") ## FACEBOOK

model_pipeline = load_model("Davlan/bert-base-multilingual-cased-ner-hrl") 

def inference(text:str):
    return model_pipeline(text)

app = Flask(__name__)

@app.route("/inference", methods = ["POST"])
def inference_endpoint():
    data = request.get_json()
    print("data: ", data)
    text = data.get("text", "")
    print("text: ", text)
    ents = inference(text)
    print("len(ents): ", len(ents))

    structure = [{
        "label" : ent["entity_group"],
        "start_offset" : ent["start"] ,
        "end_offset" : ent["end"]
    }
    for ent in ents
    ]
    ##?universeel terwijl tokenizer in transformer?...
    print("structure: ", structure)
    return jsonify(structure)

if __name__ == "__main__":
    print("Launching inference webserver")
    app.run(port=8081)