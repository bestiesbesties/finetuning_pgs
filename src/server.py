from flask import Flask, request, jsonify 
from encoder import Encoder
from _config import config

model = Encoder("built/369b0")
seeds = config["seeds"]
semi_seeds = ["A", "B", "C"]

app = Flask(__name__)
@app.route("/inference", methods = ["POST"])
def inference_endpoint():
    data = request.get_json()
    text = data.get("text", "")
    ios = [ model.forward_with_dropout_seed(text, seed) for seed in seeds] ## TODO check chronological order remains

    for i, io in enumerate(ios): ## TODO document what this checks
        print(i, "len(io[0]): ", len(io[0]), "len(io[1]): ", len(io[1]), "len(io[2]): ", len(io[2]))

    structure = {}
    for seed, semi_seed, io in zip(seeds, semi_seeds, ios):

        structure[str(seed) + semi_seed] =  { ## TODO determine model name
            "tokens" : io[0],
            "offsets" : io[1],
            "logits" :  io[2].detach().cpu().tolist() ## TODO is detaching needed 
        }
    print(structure[list(structure.keys())[0]])
    return jsonify(structure)

if __name__ == "__main__":
    print("Launching inference webserver")
    app.run(port=8081)