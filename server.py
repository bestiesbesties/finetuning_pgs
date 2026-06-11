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
    if not app.model:
         app.model = encoder.Encoder(os.path.join(last_run, "model"))
    data = request.get_json()
    text = data.get("text", "")
    ios = [ app.model.forward_with_dropout_seed(text, seed) for seed in seeds] ## TODO check chronological order remains

    # for i, io in enumerate(ios): ## TODO document what this checks
    #     print(i, "len(io[0]): ", len(io[0]), "len(io[1]): ", len(io[1]), "len(io[2]): ", len(io[2]))

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

if __name__ == "__main__":
    print("Launching inference webserver")
    app.run(port=8081)