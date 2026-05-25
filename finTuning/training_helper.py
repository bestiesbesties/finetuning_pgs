from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
import evaluate
seqeval = evaluate.load("seqeval")

def train(tokenizer, model, seed, learning_rate, batch_size, epochs, weight_decay, train_dataset, test_dataset):
    ## TODO gridsearch
    ## TODO gradient descent (AdamW) or Bayesian optimization (optuna)
    training_args = TrainingArguments(
        seed=seed,
        output_dir="./results", ## TODO output  dir
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
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

                # ignore padding
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
