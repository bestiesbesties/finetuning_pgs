from flask import Flask, request, jsonify 
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch 
import os
BATCH = 0

def printer(iterable):
    for i, x in enumerate(iterable):
        print(i, x, "\n")

class Encoder():
    def __init__(self, model_name):    
        self.model_name = model_name
        self.model_path = self._scope_model_path(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path, local_files_only=True)

    def _scope_model_path(self, model_name) -> str:
        model_path = "./models/" +  model_name
        print("model_path: ", model_path)
        if not os.path.exists(model_path):
            raise RuntimeError(f"'model' not in models folder")
        return model_path
    
    def forward(self, t:str):
        inputs = self.tokenizer(t, return_tensors="pt", truncation=True, padding=True)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[BATCH])
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        return (tokens, outputs.logits[BATCH])

models = [
    Encoder("bert-base-multilingual-cased-ner-hrl"),
    Encoder("wikineural-multilingual-ner")
]

app = Flask(__name__)
@app.route("/inference", methods = ["POST"])
def inference_endpoint():
    data = request.get_json()
    text = data.get("text", "")
    ios = [ model.forward(text) for model in models] ## TODO check of chronologische volgorde altijd hetzelfde blijft
    
    for i, io in enumerate(ios):
        print(i)
        print("len(io[0]): ", len(io[0]))
        print("len(io[1]): ", len(io[1]))

    structure = {}
    for model, io in zip(models, ios):
        structure[model.model_name] =  {
            "tokens" : io[0],
            "logits" :  io[1].detach().cpu().tolist()
        }
    
    return jsonify(structure)

if __name__ == "__main__":
    print("Launching inference webserver")
    app.run(port=8081)