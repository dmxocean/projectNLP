import json
from functools import partial
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from datasets.config import MAX_DATASET_CONFIG_ID_READABLE_LENGTH
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


def reformat_json(input_json_path: str, output_file_path: str) -> None:
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_file_path, "w", encoding="utf-8") as outfile:
        for doc_index, doc in enumerate(data):
            doc_id = doc.get("data", {}).get("id", f"doc_{doc_index}")  # Safer access
            text = doc.get("data", {}).get("text", "")

            if not text:  # Skip if no text
                print(
                    f"Warning: Document {doc_id} (index {doc_index}) has no text. Skipping."
                )
                continue

            doc_annotations = []
            predictions = doc.get("predictions", [])

            if predictions[0]["result"]:
                for pred in predictions[0]["result"]:
                    start = pred["value"]["start"]
                    end = pred["value"]["end"]
                    label = pred["value"]["labels"][0]

                    doc_annotations.append(
                        {
                            "start": start,
                            "end": end,
                            "label": label,
                        }
                    )
            doc_annotations.sort(key=lambda x: x["start"])
            output_record = {
                "id": doc_id,
                "text": text,
                "annotations": doc_annotations,
            }

            outfile.write(json.dumps(output_record, ensure_ascii=False) + "\n")


def process_batch(batch, tokenizer, max_length, label_to_id):
    token_info = tokenizer(
        batch["text"],
        padding=False,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
    )

    all_labels = []

    for i in range(len(batch["text"])):
        labels = align_labels_to_tokens(
            token_info["offset_mapping"][i],
            batch["annotations"][i],
            label_to_id,
        )
        all_labels.append(labels)

    return {
        "input_ids": token_info["input_ids"],
        "attention_mask": token_info["attention_mask"],
        "labels": all_labels,
    }


def align_labels_to_tokens(
    token_offsets: list[
        tuple[int, int]
    ],  # List of (start_char, end_char) for each token
    original_annotations: list[
        dict[str, Any]
    ],  # List of {'start': s, 'end': e, 'label': l}
    label_to_id: dict[str, int],  # Mapping from BIOES tag name to ID
) -> list[int]:
    """
    Aligns character-level annotations to token-level BIOES tags.

    Args:
        token_offsets: List of (start_char, end_char) tuples from the tokenizer.
        original_annotations: List of annotation dicts {'start': s, 'end': e, 'label': l}.
                              Assumed to be sorted by start position.
        label_to_id: Dictionary mapping BIOES label names (e.g., "B-NEG", "O") to IDs.

    Returns:
        A list of integer label IDs corresponding to each token. Special tokens
        (like CLS, SEP) are assigned -100.
    """
    num_tokens = len(token_offsets)
    # Initialize all token labels to 'O' (Outside)
    token_labels = [label_to_id["O"]] * num_tokens

    # Iterate through each character-level annotation
    for annotation in original_annotations:
        annot_start = annotation["start"]
        annot_end = annotation["end"]
        label_type = annotation["label"]  # e.g., "NEG", "NSCO"

        # Find indices of tokens that overlap with the current annotation
        span_indices = []
        for i, (tok_start, tok_end) in enumerate(token_offsets):
            # --- CRITICAL: Ignore special tokens (CLS, SEP) for annotation alignment ---
            # These often have offsets (0, 0) or might span the entire sequence in some tokenizers.
            # We only want to align annotations with *actual* content tokens.
            # A robust check is to ensure the token has a non-zero span.
            if tok_start == tok_end:
                continue

            # Check for overlap: token starts before annotation ends AND token ends after annotation starts
            if tok_start < annot_end and tok_end > annot_start:
                span_indices.append(i)

        if not span_indices:
            # Annotation doesn't align with any valid token (could be whitespace, etc.)
            continue

        # Determine BIOES tag based on the number of tokens covered
        first_token_idx = span_indices[0]
        last_token_idx = span_indices[-1]

        if len(span_indices) == 1:
            # Single-token entity
            tag = f"S-{label_type}"
            token_labels[first_token_idx] = label_to_id.get(tag, label_to_id["O"])
        else:
            # Multi-token entity
            # Beginning token
            tag_b = f"B-{label_type}"
            token_labels[first_token_idx] = label_to_id.get(tag_b, label_to_id["O"])
            # Ending token
            tag_e = f"E-{label_type}"
            token_labels[last_token_idx] = label_to_id.get(tag_e, label_to_id["O"])
            # Inside tokens (if any)
            for i in span_indices[1:-1]:
                tag_i = f"I-{label_type}"
                token_labels[i] = label_to_id.get(tag_i, label_to_id["O"])

        # Note: This implementation assumes annotations don't structurally overlap
        # (e.g., one token belonging to both NEG and NSCO). If they do, the later
        # annotation in the `original_annotations` list will overwrite the earlier one's tag.
        # Given NEG/NSCO/UNC/USCO, this overwrite is likely the desired behavior or
        # indicates an upstream annotation issue.

    # --- Assign -100 to special tokens ---
    # Iterate again *after* assigning real labels to prevent overwriting -100
    for i, (tok_start, tok_end) in enumerate(token_offsets):
        if tok_start == tok_end:
            # Check if it's a *real* special token (e.g. CLS/SEP often map to (0,0))
            # Heuristic: often the first or last token, or check against tokenizer special IDs if needed
            # Assign -100, which is ignored by the loss function
            token_labels[i] = -100

    return token_labels


def compute_metrics(eval_preds):
    """Computes P/R/F1 for NER using seqeval"""
    predictions_logits, labels = eval_preds  # Original logits
    # Get the most likely prediction ID (index of the highest logit)
    predictions_ids = np.argmax(predictions_logits, axis=2)  # Actual predicted IDs

    true_predictions = [
        [LABELS_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(
            predictions_ids, labels
        )  # Use predictions_ids here
    ]
    true_labels = [
        [LABELS_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions_ids, labels)
    ]

    report = classification_report(true_labels, true_predictions)
    print(report)

    # Use seqeval to compute metrics
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


if __name__ == "__main__":
    json_document = "data/raw/training.json"
    out_file = "dl_system/data/training.jsonl"
    reformat_json(json_document, out_file)

    raw_data = load_dataset("json", data_files=out_file)
    raw_data = raw_data["train"].train_test_split(test_size=0.1)

    long_tokenizer = AutoTokenizer.from_pretrained(
        "hyperonym/xlm-roberta-longformer-base-16384"
    )
    model_max_length = 16384

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

    model_checkpoint = "severinsimmler/xlm-roberta-longformer-base-16384"

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
        # Handle error appropriately, maybe exit or raise
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        # Handle error appropriately

    NUM_EPOCHS_DEMO = 1  # Keep low for a quick demo
    TRAIN_BATCH_SIZE_DEMO = 1  # MUST be small for 16k sequence length model
    EVAL_BATCH_SIZE_DEMO = 2  # Can be slightly larger
    MODEL_OUTPUT_DIR = (
        "./ner_longformer_demo_results"  # Where results/checkpoints are saved
    )
    LOGGING_STEPS_DEMO = 1  # How often to log training loss
    LEARNING_RATE_DEMO = 1e-4
    EVAL_SAVE_STEPS_DEMO = 10

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS_DEMO,
        per_device_train_batch_size=TRAIN_BATCH_SIZE_DEMO,
        per_device_eval_batch_size=EVAL_BATCH_SIZE_DEMO,
        learning_rate=LEARNING_RATE_DEMO,
        weight_decay=0.01,  # Standard weight decay
        eval_strategy="steps",  # Evaluate at the end of each epoch
        eval_steps=EVAL_SAVE_STEPS_DEMO,
        save_strategy="steps",  # Save checkpoint at the end of each epoch
        save_steps=EVAL_SAVE_STEPS_DEMO,
        logging_strategy="steps",  # Log training loss during epochs
        logging_steps=LOGGING_STEPS_DEMO,
        load_best_model_at_end=True,  # Load the best model based on validation metric/loss
        metric_for_best_model="f1",  # Use F1 score to determine the best model
        push_to_hub=False,  # Set to True if you want to upload to Hugging Face Hub
        report_to=["wandb"],  # Disable external reporting like W&B for simple demo
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=mapped_dataset["train"],
        eval_dataset=mapped_dataset[
            "test"
        ],  # Make sure your validation split is named 'test'
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
