from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
import evaluate
seqeval = evaluate.load("seqeval")

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
    )

    def _compute_metrics(p):
        predictions, labels = p

        predictions = np.argmax(predictions, axis=2)

        true_predictions = []
        true_labels = []

        for pred, lab in zip(predictions, labels):

            current_preds = []
            current_labels = []

            for p_i, l_i in zip(pred, lab):

                if l_i == -100:
                    continue

                current_preds.append(model.config.id2label[p_i])
                current_labels.append(model.config.id2label[l_i])

            true_predictions.append(current_preds)
            true_labels.append(current_labels)

        results = seqeval.compute(
            predictions=true_predictions,
            references=true_labels
        )

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