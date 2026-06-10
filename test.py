import os
from finTuning import general_helper
CONFIG = general_helper.load_config()
RANDOM_STATE = 800
LABELS = CONFIG["labels"]
MODEL_PATH ="./models/robbert-v2-dutch-ner"
DATA_PATH = os.path.join("temp", "admin3.jsonl")

## model loading
from finTuning import model_helper
tokenizer, model = model_helper.load_model(MODEL_PATH, LABELS, reinitialize=True)

## supervised data transforming
from finTuning import data_helper
data = data_helper.load_jsonl(DATA_PATH)
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=RANDOM_STATE
)
from datasets import Dataset
train_dataset = Dataset.from_list(data_helper.tokenize_data(train_data, tokenizer, model))
test_dataset = Dataset.from_list(data_helper.tokenize_data(test_data, tokenizer, model))

## metrics & training
SEED = CONFIG["seed"]
LEARNING_RATE = 0.00002
BATCH_SIZE = 8
EPOCHS = 4
WEIGHT_DECAY = 0.01
from finTuning import training_helper
trainer = training_helper.train(tokenizer, model, SEED, LEARNING_RATE, BATCH_SIZE, EPOCHS, WEIGHT_DECAY, train_dataset,  test_dataset)
print("trainer.evaluate():", trainer.evaluate())

model_helper.save_model(SEED, 0, tokenizer, trainer)
