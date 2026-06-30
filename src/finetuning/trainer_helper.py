from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
import evaluate
from collections import Counter
seqeval = evaluate.load("seqeval")
from ..labeling import serialization

from .._config import config

def train(tokenizer, model, train_dataset, test_dataset):
    training_args = TrainingArguments(
        seed=config["uniform_random_state"],
        output_dir="./delete", ## TODO output dir uitzetten
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        weight_decay=config["weight_decay"],
        save_strategy="epoch",
        eval_strategy="epoch",
        dataloader_pin_memory=False,
    )

    # print("train id2label: ", model.config.id2label)
    # print("train label2id: ", model.config.label2id)

    def _compute_metrics(p):
        predictions, labels = p

        predictions = np.argmax(predictions, axis=2)

        true_preds = []
        true_golds = []

        for pred, lab in zip(predictions, labels):
            for p, l in zip(pred, lab):
                if l == -100:
                    continue
                true_preds.append(model.config.id2label[p])
                true_golds.append(model.config.id2label[l])

        results = seqeval.compute(
            predictions=[true_preds],
            references=[true_golds],
            zero_division=0
        )

        print("SAVING METRICS + DEBUG")
        
        serialization.export_results_iteratively({
                "precision" : round(results["overall_precision"], 4),
                "recall" : round(results["overall_recall"], 4),
                "f1" : round(results["overall_f1"], 4),
                "accuracy" : round(results["overall_accuracy"], 4),
                "GOLD" : Counter(true_golds),
                "PREDS" : Counter(true_preds),
                "ENTITY RATE -> gold" : sum(t != "O" for t in true_golds),
                "| pred" : sum(t != "O" for t in true_preds)
            })

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=_compute_metrics,
        data_collator=DataCollatorForTokenClassification(tokenizer)
    )
    trainer.train()
    return trainer