import os
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
print("!> !> !> Forcing MPS (Apple) if available with CPU fallback on condition")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH = 0

class Encoder():
    def __init__(self, model_source):
        self.model_source = model_source
        self.model_path = self._scope_model_path()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path, local_files_only=True)
        self.model.to(DEVICE)
        print("model_class: ", self.model.__class__)
        print("model: ", self.model)

    def _scope_model_path(self) -> str:
        model_path = "." + os.path.sep + self.model_source
        if not os.path.exists(model_path):
            raise RuntimeError("Not in 'models' folder: ", model_path)
        return model_path
    
    def forward_with_dropout_seed(self, text:str, seed:int):
        self.model.train()
        torch.manual_seed(seed)

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True,  return_offsets_mapping=True)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[BATCH])
        offsets = inputs.offset_mapping[BATCH].tolist()
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        return (tokens, offsets, outputs.logits[BATCH])