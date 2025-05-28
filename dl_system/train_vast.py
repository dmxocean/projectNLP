import json
from functools import partial
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from .utils import (
    # reformat_json, USED FOR DATASET CREATION BEFOREHAND
    process_batch,
)


def compute_metrics(eval_preds):
    """Computes P/R/F1 for NER using seqeval"""
    predictions_logits, labels = eval_preds  # original logits
    predictions_ids = np.argmax(predictions_logits, axis=2)  # our predictions

    true_predictions = [
        [LABELS_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions_ids, labels)
    ]
    true_labels = [
        [LABELS_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions_ids, labels)
    ]

    report = classification_report(true_labels, true_predictions)
    print(report)

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


if __name__ == "__main__":
    train_data_path = "data/training_custom.jsonl"
    test_data_path = "data/testing_custom.jsonl"

    data_files = {"train": train_data_path, "test": test_data_path}

    raw_data = load_dataset("json", data_files=data_files)
    print(raw_data)

    long_tokenizer = AutoTokenizer.from_pretrained(
        "BSC-TeMU/roberta-base-biomedical-clinical-es"
    )
    model_max_length = long_tokenizer.model_max_length

    labels_originals = ["NEG", "NSCO", "UNC", "USCO"]
    label_prefixes = ["S", "B", "I", "E"]
    LABELS_LIST = ["O"]
    for original_label in labels_originals:
        for label_prefix in label_prefixes:
            LABELS_LIST.append(f"{label_prefix}-{original_label}")

    label_to_id = {label: i for i, label in enumerate(LABELS_LIST)}
    print(label_to_id)

    mapped_dataset = raw_data.map(
        partial(
            process_batch,
            tokenizer=long_tokenizer,
            max_length=model_max_length,
            label_to_id=label_to_id,
        ),
        batched=True,
        batch_size=1000,
    )

    # data loader
    data_collator = DataCollatorForTokenClassification(tokenizer=long_tokenizer)

    model_checkpoint = "BSC-TeMU/roberta-base-biomedical-clinical-es"

    id_to_label = {id_: label for label, id_ in label_to_id.items()}
    num_labels = len(LABELS_LIST)

    print(LABELS_LIST)

    try:
        print(f"Loading model: {model_checkpoint}...")
        model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            num_labels=num_labels,
            id2label=id_to_label,
            label2id=label_to_id,
        )
        print("Model loaded successfully!")

        # Optional: Check if GPU is available and move model to GPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Moving model to GPU: {torch.cuda.get_device_name(0)}")
            model.to(device)
        else:
            device = torch.device("cpu")
            print("GPU not available, using CPU.")

    except OSError as e:
        print(f"Error loading model checkpoint '{model_checkpoint}': {e}")
        print(
            "Please ensure the checkpoint name is correct and you have internet access."
        )
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")

    NUM_EPOCHS_DEMO = 100
    TRAIN_BATCH_SIZE_DEMO = 16
    EVAL_BATCH_SIZE_DEMO = 32
    MODEL_OUTPUT_DIR = "./ner_longformer_demo_results"
    LOGGING_STEPS_DEMO = 10
    LEARNING_RATE_DEMO = 1e-5
    EVAL_SAVE_STEPS_DEMO = 100

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS_DEMO,
        per_device_train_batch_size=TRAIN_BATCH_SIZE_DEMO,
        per_device_eval_batch_size=EVAL_BATCH_SIZE_DEMO,
        learning_rate=LEARNING_RATE_DEMO,
        weight_decay=0.001,
        eval_strategy="epoch",
        eval_steps=EVAL_SAVE_STEPS_DEMO,
        save_strategy="no",
        logging_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="f1",  # our target metric
        report_to=["wandb"],  # as always wandb is used
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=mapped_dataset["train"],
        eval_dataset=mapped_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Trainer initialized successfully!")
    trainer.evaluate()

    print("Starting...")
    try:
        trainer.train()
    except Exception as e:
        print(e)
    print("Ended...")
