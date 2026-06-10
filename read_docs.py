import parser
import os
from hashlib import sha256
import pandas as pd        
os.makedirs("data", exist_ok=True)
df = pd.DataFrame(columns=["origin", "text"])
p = parser.Parser()
DATA_FOLDER = "downloaded_documents"
import json
status = {"succes" : [], "error" : []}

def digest_document(origin:str, path:str):
    try:
        chunks = p(path)
        for chunk in chunks:
            hash = sha256()
            hash.update(chunk.encode("utf-8"))
            digest = hash.hexdigest()
            if digest not in df.index:
                df.loc[digest] = [origin, chunk]
        return True
    except Exception as e:
        print(e)
        return False

controle_objects = os.listdir(DATA_FOLDER)
for controle_object in controle_objects:
    controle_object_path = os.path.join(DATA_FOLDER, controle_object) 
    print("controle_object_path", controle_object_path, "\n")
    documents = os.listdir(controle_object_path)
    print("documents: ", documents)
    for document in documents:
        if document not in status["succes"] and document not in status["error"]:
            opr = digest_document(controle_object, os.path.join(DATA_FOLDER, controle_object, document))
            print("opr: ", opr)
            status["succes"].append(document) if opr else status["error"].append(document)
        print("saving")
        df.to_csv(os.path.join("data", "hashed_chunks.csv"), index=True)
        with open(os.path.join("data",  "status.json"), "w") as file:
            json.dump(status, file, indent=4)
   
