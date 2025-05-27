import json
from functools import partial
from typing import Any, List, Dict, Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from sklearn.metrics import classification_report as sklearn_classification_report

import wandb


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
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

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
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids) * math.sqrt(self.embed_dim)
        x = x + self.pos_embedding[:, :seq_len, :]

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0

        encoder_output = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )
        logits = self.classifier(encoder_output)
        return logits


# --- Data Processing Helper Functions ---
def reformat_json(input_json_path: str, output_file_path: str) -> None:
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(output_file_path, "w", encoding="utf-8") as outfile:
        for doc_index, doc in enumerate(data):
            doc_id = doc.get("data", {}).get("id", f"doc_{doc_index}")
            text = doc.get("data", {}).get("text", "")
            if not text:
                print(
                    f"Warning: Document {doc_id} (index {doc_index}) has no text. Skipping."
                )
                continue
            doc_annotations = []
            predictions_outer = doc.get("predictions", [])
            if (
                predictions_outer
                and isinstance(predictions_outer, list)
                and len(predictions_outer) > 0
                and predictions_outer[0].get("result")
                and isinstance(predictions_outer[0]["result"], list)
            ):
                for pred_inner in predictions_outer[0]["result"]:
                    value = pred_inner.get("value", {})
                    start = value.get("start")
                    end = value.get("end")
                    labels_list_from_pred = value.get("labels")
                    if (
                        start is not None
                        and end is not None
                        and labels_list_from_pred
                        and isinstance(labels_list_from_pred, list)
                        and labels_list_from_pred
                    ):
                        label = labels_list_from_pred[0]
                        doc_annotations.append(
                            {"start": start, "end": end, "label": label}
                        )
            doc_annotations.sort(key=lambda x: x["start"])
            output_record = {"id": doc_id, "text": text, "annotations": doc_annotations}
            outfile.write(json.dumps(output_record, ensure_ascii=False) + "\n")


def align_labels_to_tokens(
    token_offsets: List[Tuple[int, int]],
    original_annotations: List[Dict[str, Any]],
    label_to_id: Dict[str, int],
) -> List[int]:
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
        first_token_idx, last_token_idx = span_indices[0], span_indices[-1]
        if len(span_indices) == 1:
            token_labels[first_token_idx] = label_to_id.get(
                f"S-{label_type}", label_to_id["O"]
            )
        else:
            token_labels[first_token_idx] = label_to_id.get(
                f"B-{label_type}", label_to_id["O"]
            )
            token_labels[last_token_idx] = label_to_id.get(
                f"E-{label_type}", label_to_id["O"]
            )
            for i in span_indices[1:-1]:
                token_labels[i] = label_to_id.get(f"I-{label_type}", label_to_id["O"])
    for i, (tok_start, tok_end) in enumerate(token_offsets):
        if tok_start == tok_end:
            token_labels[i] = -100
    return token_labels


def process_batch_for_custom_model(batch, tokenizer, max_length, label_to_id):
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


# <<< MODIFIED EVALUATION FUNCTION >>>
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    num_labels: int,
    labels_list_for_report: List[str],  # Use full LABELS_LIST for sklearn report
):
    model.eval()
    total_eval_loss = 0

    all_true_flat_ids = []  # For sklearn, expects flat list of labels
    all_pred_flat_ids = []  # For sklearn

    correct_predictions_token = 0
    total_active_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)  # Shape: (batch_size, seq_len)

            logits = model(
                input_ids, attention_mask=attention_mask
            )  # Shape: (batch_size, seq_len, num_labels)
            loss = criterion(logits.view(-1, num_labels), labels.view(-1))
            total_eval_loss += loss.item()

            preds_ids = torch.argmax(logits, dim=-1)  # Shape: (batch_size, seq_len)

            # Collect active (non -100) labels and predictions for sklearn report and accuracy
            for i in range(labels.size(0)):  # Iterate over each sequence in the batch
                seq_labels = labels[i]
                seq_preds = preds_ids[i]

                active_indices = (
                    seq_labels != -100
                )  # Mask for active tokens in this sequence

                active_true_ids = seq_labels[active_indices]
                active_pred_ids = seq_preds[active_indices]

                all_true_flat_ids.extend(active_true_ids.cpu().tolist())
                all_pred_flat_ids.extend(active_pred_ids.cpu().tolist())

                correct_predictions_token += (
                    (active_pred_ids == active_true_ids).sum().item()
                )
                total_active_tokens += active_true_ids.numel()

    avg_eval_loss = total_eval_loss / len(dataloader)
    token_accuracy = (
        (correct_predictions_token / total_active_tokens)
        if total_active_tokens > 0
        else 0
    )

    # Generate sklearn classification report
    # Sklearn report needs target names (label strings) corresponding to the label IDs
    # It's good to provide all possible labels used for training.
    # Filter out IDs if they don't appear in either true or preds to avoid warnings if labels_list_for_report has unused labels
    unique_label_ids_in_data = sorted(list(set(all_true_flat_ids + all_pred_flat_ids)))
    target_names_for_report = [
        labels_list_for_report[l_id] for l_id in unique_label_ids_in_data
    ]

    sklearn_report_str = sklearn_classification_report(
        all_true_flat_ids,
        all_pred_flat_ids,
        labels=unique_label_ids_in_data,  # Pass the actual IDs present
        target_names=target_names_for_report,  # Corresponding names
        zero_division=0,
        digits=3,  # More precision
    )

    # For wandb HTML, just use the string report wrapped in <pre>
    report_html_for_wandb = f"<pre>{sklearn_report_str}</pre>"

    # You can also get a dict version from sklearn if needed for other logging
    # sklearn_report_dict = sklearn_classification_report(..., output_dict=True)

    return avg_eval_loss, token_accuracy, report_html_for_wandb  # Return HTML string


# <<< END MODIFIED EVALUATION FUNCTION >>>

if __name__ == "__main__":
    CONFIG = {
        "train_file_raw": "../data/raw/training.json",
        "test_file_raw": "../data/raw/test.json",
        "train_file_jsonl": "data/training_custom.jsonl",
        "test_file_jsonl": "data/testing_custom.jsonl",
        "hf_tokenizer_name": "distilbert-base-multilingual-cased",
        "model_max_seq_len": 256,  # Reduced for faster demo with custom model
        "batch_size": 8,  # Increased slightly
        "embed_dim": 128,  # Reduced for faster demo
        "nhead": 4,
        "num_encoder_layers": 2,
        "dim_feedforward": 256,  # Reduced
        "dropout": 0.1,
        "learning_rate": 5e-5,
        "num_epochs": 3,
        "print_every_n_steps": 10,  # Reduced for more frequent logging
    }
    wandb.init(
        project="custom-ner-transformer-sklearn", config=CONFIG
    )  # Changed project name slightly

    # reformat_json(CONFIG["train_file_raw"], CONFIG["train_file_jsonl"])
    # reformat_json(CONFIG["test_file_raw"], CONFIG["test_file_jsonl"])

    data_files = {
        "train": CONFIG["train_file_jsonl"],
        "test": CONFIG["test_file_jsonl"],
    }
    raw_datasets = load_dataset("json", data_files=data_files)

    hf_tokenizer = AutoTokenizer.from_pretrained(CONFIG["hf_tokenizer_name"])
    VOCAB_SIZE = hf_tokenizer.vocab_size

    labels_originals = ["NEG", "NSCO", "UNC", "USCO"]
    label_prefixes = ["S", "B", "I", "E"]
    LABELS_LIST = ["O"]  # This will be used globally by evaluate_model
    for original_label in labels_originals:
        for label_prefix in label_prefixes:
            LABELS_LIST.append(f"{label_prefix}-{original_label}")
    label_to_id = {label: i for i, label in enumerate(LABELS_LIST)}
    NUM_LABELS = len(LABELS_LIST)

    print("Mapping datasets...")
    mapped_datasets = raw_datasets.map(
        partial(
            process_batch_for_custom_model,
            tokenizer=hf_tokenizer,
            max_length=CONFIG["model_max_seq_len"],
            label_to_id=label_to_id,
        ),
        batched=True,
        batch_size=1000,
        remove_columns=raw_datasets["train"].column_names,
    )
    print(
        f"Dataset mapping complete. Train size: {len(mapped_datasets['train'])}, Test size: {len(mapped_datasets['test'])}"
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=hf_tokenizer)
    train_dataloader = DataLoader(
        mapped_datasets["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=CONFIG["batch_size"],
    )
    eval_dataloader = DataLoader(
        mapped_datasets["test"],
        collate_fn=data_collator,
        batch_size=CONFIG["batch_size"],
    )

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    custom_ner_model = TransformerNER(
        vocab_size=VOCAB_SIZE,
        num_labels=NUM_LABELS,
        embed_dim=CONFIG["embed_dim"],
        nhead=CONFIG["nhead"],
        num_encoder_layers=CONFIG["num_encoder_layers"],
        dim_feedforward=CONFIG["dim_feedforward"],
        max_seq_len=CONFIG["model_max_seq_len"],
        dropout=CONFIG["dropout"],
    ).to(DEVICE)
    print("CustomTransformerNER model initialized.")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(custom_ner_model.parameters(), lr=CONFIG["learning_rate"])

    print("Starting training loop with evaluation and wandb logging...")
    wandb.watch(
        custom_ner_model, criterion, log="all", log_freq=CONFIG["print_every_n_steps"]
    )

    for epoch in range(CONFIG["num_epochs"]):
        custom_ner_model.train()
        total_epoch_loss = 0
        num_batches = 0
        print(f"\n--- Epoch {epoch + 1}/{CONFIG['num_epochs']} ---")

        for step, batch in enumerate(train_dataloader):
            # (Training step logic remains the same)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            optimizer.zero_grad()
            logits = custom_ner_model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits.view(-1, NUM_LABELS), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()
            num_batches += 1

            if (step + 1) % CONFIG["print_every_n_steps"] == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch + 1}, Step {step + 1}/{len(train_dataloader)}, Train Loss: {loss.item():.4f}, LR: {current_lr:.2e}"
                )
                wandb.log(
                    {
                        "train_loss_step": loss.item(),
                        "learning_rate": current_lr,
                        "epoch_float": epoch + (step + 1) / len(train_dataloader),
                    }
                )

        avg_epoch_train_loss = total_epoch_loss / num_batches
        print(
            f"End of Epoch {epoch + 1}, Average Training Loss: {avg_epoch_train_loss:.4f}"
        )

        # Perform evaluation at the end of each epoch
        avg_eval_loss, token_accuracy, eval_report_html = evaluate_model(
            custom_ner_model,
            eval_dataloader,
            DEVICE,
            criterion,
            NUM_LABELS,
            LABELS_LIST,  # Pass global LABELS_LIST
        )
        print(
            f"Epoch {epoch + 1}, Eval Loss: {avg_eval_loss:.4f}, Token Accuracy: {token_accuracy:.4f}"
        )
        # Print the report string to console for quick view
        # print(f"\nEpoch {epoch+1} Sklearn Classification Report:\n{eval_report_html.replace('<pre>', '').replace('</pre>', '')}") # Print plain text

        wandb.log(
            {
                "epoch": epoch + 1,  # Log integer epoch for easier x-axis
                "avg_train_loss_epoch": avg_epoch_train_loss,
                "avg_eval_loss_epoch": avg_eval_loss,
                "token_accuracy_epoch": token_accuracy,
                f"eval_classification_report_epoch_{epoch + 1}": wandb.Html(
                    eval_report_html
                ),  # Log HTML report
            }
        )

    print("\nTraining finished.")

    print("\nPerforming final evaluation on the test set...")
    final_loss, final_accuracy, final_report_html = evaluate_model(
        custom_ner_model,
        eval_dataloader,
        DEVICE,
        criterion,
        NUM_LABELS,
        LABELS_LIST,  # Pass global LABELS_LIST
    )
    print(f"\nFinal Test Loss: {final_loss:.4f}")
    print(f"Final Test Token Accuracy: {final_accuracy:.4f}")
    print(
        "\nFinal Sklearn Classification Report:\n",
        final_report_html.replace("<pre>", "").replace("</pre>", ""),
    )  # Print plain text

    wandb.log(
        {
            "final_test_loss": final_loss,
            "final_test_token_accuracy": final_accuracy,
            "final_classification_report": wandb.Html(final_report_html),
        }
    )

    wandb.finish()
