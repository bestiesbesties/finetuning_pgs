import os
from flask import Flask, request, jsonify
import sys
from src import _config, _general, encoder

last_block = _general.get_last_id(_config.config["folder_structure"][0], _config.config["folder_structure"][1])
last_run = _general.get_last_id(last_block, _config.config["folder_structure"][2])
seeds = _config.config["seeds"]
semi_seeds = ["A", "B", "C"]

app = Flask(__name__)
app.model = None
@app.route("/inference", methods = ["POST"])
def inference_endpoint():
    if app.model:
        data = request.get_json()
        text = data.get("text", "")
        ios = [ app.model.forward_with_dropout_seed(text, seed) for seed in seeds] ## TODO check chronological order remains

        structure = {}
        for seed, semi_seed, io in zip(seeds, semi_seeds, ios):
        ## Cutoff <s> and </s>
            structure[str(seed) + semi_seed] =  { ## TODO determine model name
                "tokens" : io[0][1:-1] ,
                "offsets" : io[1][1:-1] ,
                "logits" :  io[2].detach().cpu().tolist()[1:-1]  ## TODO is detaching needed 
            }
        # print(structure[list(structure.keys())[0]])
        return jsonify(structure)
    else:
         return jsonify({"error": "No model"})


@app.route("/switch", methods=["POST"])
def switch_endpoint():
    data = request.get_json()
    path = data.get("text", "")
    if not path:
        return jsonify({"error": "No path"})
    try:
        print("switching to:", path)
        app.model = encoder.Encoder(path)
        # print("server id2label:", app.model.model.config.id2label)
        # print("server label2id:", app.model.model.config.label2id)
        return jsonify({"status": "ok"})

    except Exception:
        return jsonify({"error": "No initialization"})

if __name__ == "__main__":
    print("Launching inference webserver")
    app.run(port=8081)