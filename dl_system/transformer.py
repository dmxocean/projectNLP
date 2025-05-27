import torch
import torch.nn as nn
import math
import wandb

import json
from functools import partial
from typing import Any, List, Dict, Tuple, Optional # Added Optional

import torch.optim as optim # For optimizer
from torch.utils.data import DataLoader # For creating batches

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification

class TransformerNER(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embed_dim: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 512,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ):
        x = self.embedding(input_ids) * math.sqrt(self.embed_dim)  # Scale embeddings

        x = x + self.pos_embedding

        # the mask need to be inverted
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0

        encoder_output = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )

        logits = self.classifier(encoder_output)
        # logits shape: (batch_size, seq_len, num_labels)

        return logits


def reformat_json(input_json_path: str, output_file_path: str) -> None:
    # (Using the exact function from your train_vast.py)
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(output_file_path, "w", encoding="utf-8") as outfile:
        for doc_index, doc in enumerate(data):
            doc_id = doc.get("data", {}).get("id", f"doc_{doc_index}")
            text = doc.get("data", {}).get("text", "")
            if not text:
                print(f"Warning: Document {doc_id} (index {doc_index}) has no text. Skipping.")
                continue
            doc_annotations = []
            predictions = doc.get("predictions", [])
            # Robustness check for predictions structure
            if predictions and isinstance(predictions, list) and len(predictions) > 0 and \
               predictions[0].get("result") and isinstance(predictions[0]["result"], list):
                for pred in predictions[0]["result"]:
                    value = pred.get("value", {})
                    start = value.get("start")
                    end = value.get("end")
                    labels_list_from_pred = value.get("labels")
                    if start is not None and end is not None and \
                       labels_list_from_pred and isinstance(labels_list_from_pred, list) and labels_list_from_pred:
                        label = labels_list_from_pred[0]
                        doc_annotations.append({"start": start, "end": end, "label": label})
            doc_annotations.sort(key=lambda x: x["start"])
            output_record = {"id": doc_id, "text": text, "annotations": doc_annotations}
            outfile.write(json.dumps(output_record, ensure_ascii=False) + "\n")

def align_labels_to_tokens(
    token_offsets: List[Tuple[int, int]],
    original_annotations: List[Dict[str, Any]],
    label_to_id: Dict[str, int],
) -> List[int]:
    # (Using the exact function from your train_vast.py)
    num_tokens = len(token_offsets)
    token_labels = [label_to_id["O"]] * num_tokens
    for annotation in original_annotations:
        annot_start = annotation["start"]
        annot_end = annotation["end"]
        label_type = annotation["label"]
        span_indices = []
        for i, (tok_start, tok_end) in enumerate(token_offsets):
            if tok_start == tok_end:
                continue
            if tok_start < annot_end and tok_end > annot_start:
                span_indices.append(i)
        if not span_indices:
            continue
        first_token_idx = span_indices[0]
        last_token_idx = span_indices[-1]
        if len(span_indices) == 1:
            tag = f"S-{label_type}"
            token_labels[first_token_idx] = label_to_id.get(tag, label_to_id["O"])
        else:
            tag_b = f"B-{label_type}"
            token_labels[first_token_idx] = label_to_id.get(tag_b, label_to_id["O"])
            tag_e = f"E-{label_type}"
            token_labels[last_token_idx] = label_to_id.get(tag_e, label_to_id["O"])
            for i in span_indices[1:-1]:
                tag_i = f"I-{label_type}"
                token_labels[i] = label_to_id.get(tag_i, label_to_id["O"])
    for i, (tok_start, tok_end) in enumerate(token_offsets):
        if tok_start == tok_end:
            token_labels[i] = -100
    return token_labels

def process_batch_for_custom_model(batch, tokenizer, max_length, label_to_id):
    # (Using the exact function from your train_vast.py, renamed for clarity if needed)
    token_info = tokenizer(
        batch["text"],
        padding=False, # Padding will be handled by DataCollator
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
        "attention_mask": token_info["attention_mask"], # For the custom model
        "labels": all_labels,
    }

if __name__ == "__main__":
    # 0. Configuration
    TRAIN_FILE_RAW = "../data/raw/training.json" # Path to your original training JSON
    TEST_FILE_RAW = "../data/raw/test.json"   # Path to your original testing JSON
    TRAIN_FILE_JSONL = "data/training_custom.jsonl"
    TEST_FILE_JSONL = "data/testing_custom.jsonl"

    run = wandb.init(
        entity='uab-deeplearning-2025',
        project='huggingface',
    )

    #print("Reformatting JSON files to JSONL...")
    #reformat_json(TRAIN_FILE_RAW, TRAIN_FILE_JSONL)
    #reformat_json(TEST_FILE_RAW, TEST_FILE_JSONL)
    #print("Reformatting complete.")

    data_files = {"train": TRAIN_FILE_JSONL, "test": TEST_FILE_JSONL}
    raw_datasets = load_dataset("json", data_files=data_files)

    HF_TOKENIZER_NAME = "distilbert-base-multilingual-cased" # Example, choose one suitable
    hf_tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_NAME)
    VOCAB_SIZE = hf_tokenizer.vocab_size

    labels_originals = ["NEG", "NSCO", "UNC", "USCO"]
    label_prefixes = ["S", "B", "I", "E"]
    LABELS_LIST = ["O"]
    for original_label in labels_originals:
        for label_prefix in label_prefixes:
            LABELS_LIST.append(f"{label_prefix}-{original_label}")
    label_to_id = {label: i for i, label in enumerate(LABELS_LIST)}
    NUM_LABELS = len(LABELS_LIST)

    MODEL_MAX_SEQ_LEN = 512

    print("Mapping datasets...")
    mapped_datasets = raw_datasets.map(
        partial(
            process_batch_for_custom_model, # Use the processing function
            tokenizer=hf_tokenizer,
            max_length=MODEL_MAX_SEQ_LEN,
            label_to_id=label_to_id,
        ),
        batched=True,
        batch_size=1000, # Process 1000 examples from raw_datasets at a time
        remove_columns=raw_datasets["train"].column_names # Remove old columns
    )
    print("Dataset mapping complete.")
    print(mapped_datasets)

    data_collator = DataCollatorForTokenClassification(tokenizer=hf_tokenizer)

    BATCH_SIZE = 64
    train_dataloader = DataLoader(
        mapped_datasets["train"],
        shuffle=True,
        collate_fn=data_collator, # Use the HF collator
        batch_size=BATCH_SIZE
    )
    # test_dataloader for evaluation (not used in this simple loop)
    eval_dataloader = DataLoader(mapped_datasets["test"], collate_fn=data_collator, batch_size=BATCH_SIZE)


    # 4. Initialize your Custom TransformerNER model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    custom_ner_model = TransformerNER(
        vocab_size=VOCAB_SIZE,
        num_labels=NUM_LABELS,
        embed_dim=256, # Example, ensure it matches what your layers expect
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=512,
        max_seq_len=MODEL_MAX_SEQ_LEN, # Should match tokenizer's max_length
        dropout=0.1
    ).to(DEVICE)
    print("CustomTransformerNER model initialized.")

    # 5. Loss Function
    # CrossEntropyLoss ignores -100 by default if specified in ignore_index
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 6. Optimizer
    LEARNING_RATE = 1e-4
    optimizer = optim.AdamW(custom_ner_model.parameters(), lr=LEARNING_RATE)

    # 7. Simple Training Loop
    NUM_EPOCHS = 30 # Number of epochs for the demo
    PRINT_EVERY_N_STEPS = 10

    print("Starting simple training loop...")
    custom_ner_model.train() # Set model to training mode

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        total_epoch_loss = 0
        total_val_loss = 0
        num_batches = 0

        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE) # For your custom model's forward pass
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()

            # Forward pass through your custom model
            logits = custom_ner_model(input_ids, attention_mask=attention_mask)
            # logits shape: (batch_size, seq_len, num_labels)
            # labels shape: (batch_size, seq_len)

            # Calculate loss
            # Reshape logits and labels for CrossEntropyLoss:
            # Logits: (batch_size * seq_len, num_labels)
            # Labels: (batch_size * seq_len)
            loss = criterion(logits.view(-1, NUM_LABELS), labels.view(-1))

            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = total_epoch_loss / num_batches
        print(f"End of Epoch {epoch+1}, Average Training Loss: {avg_epoch_loss:.4f}")

        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            logits = custom_ner_model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits.view(-1, NUM_LABELS), labels.view(-1))
            total_val_loss += loss.item()

        avg_eval_loss = total_val_loss / num_batches

        log_dir = {'train/loss': avg_epoch_loss, 'val/loss': avg_eval_loss}
        wandb.log(log_dir)
