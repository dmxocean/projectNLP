import json
from functools import partial
from typing import Any, List, Dict, Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from sklearn.metrics import classification_report as sklearn_classification_report

import wandb


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Expects a tensor of weights for each class, or None
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (N, C) where C = num_classes
        # targets: (N)

        # Calculate Cross Entropy loss without reduction, applying ignore_index
        # This also handles the case where targets are out of bounds for logits.
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", ignore_index=self.ignore_index
        )

        # For numerical stability, pt should be calculated carefully.
        # If ce_loss is very large, exp(-ce_loss) can be zero.
        # pt is the probability of the true class.
        pt = torch.exp(-ce_loss)

        # Calculate Focal component: (1 - pt)^gamma
        focal_term = (1 - pt).pow(self.gamma)

        # The raw focal loss for each element
        loss_elements = focal_term * ce_loss

        # Apply alpha weighting if provided
        if self.alpha is not None:
            if not isinstance(self.alpha, torch.Tensor):
                raise TypeError(
                    "Alpha must be a torch.Tensor of weights per class or None."
                )
            if self.alpha.ndim != 1 or self.alpha.size(0) != logits.size(1):
                raise ValueError(
                    f"Alpha tensor must be 1D and have size C (num_classes={logits.size(1)}), got {self.alpha.shape}"
                )

            # Ensure alpha is on the same device as targets
            alpha_weights = self.alpha.to(targets.device)

            # Gather alpha values corresponding to each target class
            # Only apply alpha to non-ignored indices
            active_mask_for_alpha = targets != self.ignore_index
            alpha_per_target = torch.ones_like(
                targets, dtype=logits.dtype
            )  # Default alpha = 1

            # Get valid targets for indexing alpha_weights
            valid_targets = targets[active_mask_for_alpha]
            if valid_targets.numel() > 0:  # Check if there are any valid targets
                alpha_per_target[active_mask_for_alpha] = alpha_weights[valid_targets]

            loss_elements = alpha_per_target * loss_elements

        # Apply reduction only to non-ignored elements
        active_mask_for_reduction = targets != self.ignore_index
        active_loss_elements = loss_elements[active_mask_for_reduction]

        if active_loss_elements.numel() == 0:  # All elements were ignored
            return torch.tensor(
                0.0,
                device=logits.device,
                requires_grad=True if logits.requires_grad else False,
            )

        if self.reduction == "mean":
            return active_loss_elements.mean()
        elif self.reduction == "sum":
            return active_loss_elements.sum()
        else:  # 'none'
            # If 'none', we should still return only the active elements or ensure caller handles it
            return loss_elements  # Caller needs to be aware of ignore_index if reduction is 'none'
            # For this script, 'mean' or 'sum' is expected.


# --- TransformerNER Class (from your transformer.py) ---
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


# --- Data Processing Helper Functions (from your transformer.py) ---
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


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,  # Will now be FocalLoss instance
    num_labels: int,
    labels_list_for_report: List[str],
    label_to_id: Dict[str, int],
):
    model.eval()
    total_eval_loss = 0
    all_true_entity_ids = []
    all_pred_entity_ids = []
    correct_entity_token_predictions = 0
    total_entity_tokens = 0
    o_label_id = label_to_id["O"]

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)
            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(
                logits.view(-1, num_labels), labels_batch.view(-1)
            )  # Criterion is now FocalLoss
            total_eval_loss += loss.item()
            preds_ids_batch = torch.argmax(logits, dim=-1)

            for i in range(labels_batch.size(0)):
                seq_true_labels = labels_batch[i]
                seq_pred_labels = preds_ids_batch[i]
                entity_tokens_mask = (seq_true_labels != -100) & (
                    seq_true_labels != o_label_id
                )
                true_ids_for_entities = seq_true_labels[entity_tokens_mask]
                pred_ids_for_entities = seq_pred_labels[entity_tokens_mask]
                all_true_entity_ids.extend(true_ids_for_entities.cpu().tolist())
                all_pred_entity_ids.extend(pred_ids_for_entities.cpu().tolist())
                correct_entity_token_predictions += (
                    (pred_ids_for_entities == true_ids_for_entities).sum().item()
                )
                total_entity_tokens += true_ids_for_entities.numel()

    avg_eval_loss = total_eval_loss / len(dataloader) if len(dataloader) > 0 else 0
    entity_token_accuracy = (
        (correct_entity_token_predictions / total_entity_tokens)
        if total_entity_tokens > 0
        else 0
    )

    unique_label_ids_in_data = sorted(
        list(set(all_true_entity_ids + all_pred_entity_ids))
    )
    target_names_for_report = []
    valid_unique_ids_for_report = []

    if unique_label_ids_in_data:
        for l_id in unique_label_ids_in_data:
            if l_id < len(labels_list_for_report):
                target_names_for_report.append(labels_list_for_report[l_id])
                valid_unique_ids_for_report.append(l_id)
            else:
                print(
                    f"Warning: Label ID {l_id} out of bounds for labels_list_for_report. Skipping in report."
                )

    if not valid_unique_ids_for_report:
        sklearn_report_str = (
            "No entities found or predicted to generate a report (excluding 'O')."
        )
    else:
        sklearn_report_str = sklearn_classification_report(
            all_true_entity_ids,
            all_pred_entity_ids,
            labels=valid_unique_ids_for_report,
            target_names=target_names_for_report,
            zero_division=0,
            digits=3,
        )
    report_html_for_wandb = f"<pre>{sklearn_report_str}</pre>"
    return avg_eval_loss, entity_token_accuracy, report_html_for_wandb


if __name__ == "__main__":
    CONFIG = {
        "train_file_raw": "../data/raw/training.json",
        "test_file_raw": "../data/raw/test.json",
        "train_file_jsonl": "data/training_custom_focal.jsonl",  # Changed output file names
        "test_file_jsonl": "data/testing_custom_focal.jsonl",
        "hf_tokenizer_name": "distilbert-base-multilingual-cased",
        "model_max_seq_len": 256,
        "batch_size": 8,
        "embed_dim": 128,
        "nhead": 4,
        "num_encoder_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "learning_rate": 5e-5,
        "num_epochs": 3,
        "print_every_n_steps": 10,
        "focal_loss_gamma": 2.0,  # <<< NEW: Gamma for Focal Loss
        # "focal_loss_alpha": [0.25, 0.75, ...] # Example: Optional alpha weights per class (tensor)
        # For simplicity, we'll use alpha=None initially
    }
    wandb.init(project="custom-ner-focal-loss", config=CONFIG)

    # print("Reformatting JSON files to JSONL...")
    # reformat_json(CONFIG["train_file_raw"], CONFIG["train_file_jsonl"])
    # reformat_json(CONFIG["test_file_raw"], CONFIG["test_file_jsonl"])
    # print("Reformatting complete.")

    data_files = {
        "train": CONFIG["train_file_jsonl"],
        "test": CONFIG["test_file_jsonl"],
    }
    # Ensure files exist, or uncomment reformat_json calls if they are raw
    try:
        raw_datasets = load_dataset("json", data_files=data_files)
    except FileNotFoundError:
        print(f"Error: One or both JSONL files not found: {data_files}")
        print(
            "Please ensure the .jsonl files exist, or uncomment the reformat_json calls to create them from raw .json files."
        )
        exit()

    hf_tokenizer = AutoTokenizer.from_pretrained(CONFIG["hf_tokenizer_name"])
    VOCAB_SIZE = hf_tokenizer.vocab_size

    labels_originals = ["NEG", "NSCO", "UNC", "USCO"]
    label_prefixes = ["S", "B", "I", "E"]
    LABELS_LIST = ["O"]
    for original_label in labels_originals:
        for label_prefix in label_prefixes:
            LABELS_LIST.append(f"{label_prefix}-{original_label}")
    label_to_id = {label: i for i, label in enumerate(LABELS_LIST)}
    NUM_LABELS = len(LABELS_LIST)

    # Optional: Define alpha weights for Focal Loss if you want to use them
    # E.g., calculate inverse frequency or set manually. Must be a tensor.
    # alpha_weights = torch.tensor([...], dtype=torch.float) # Length NUM_LABELS
    alpha_weights = None  # For simplest case, no alpha weighting initially

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

    # --- MODIFIED: Use FocalLoss ---
    criterion = FocalLoss(
        gamma=CONFIG["focal_loss_gamma"],
        alpha=alpha_weights,  # Pass None or your defined alpha_weights tensor
        ignore_index=-100,
    )
    print(
        f"Using FocalLoss with gamma={CONFIG['focal_loss_gamma']} and alpha={alpha_weights}"
    )
    # ---

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
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            optimizer.zero_grad()
            logits = custom_ner_model(input_ids, attention_mask=attention_mask)
            loss = criterion(
                logits.view(-1, NUM_LABELS), labels.view(-1)
            )  # Now uses FocalLoss
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

        avg_epoch_train_loss = total_epoch_loss / num_batches if num_batches > 0 else 0
        print(
            f"End of Epoch {epoch + 1}, Average Training Loss: {avg_epoch_train_loss:.4f}"
        )

        avg_eval_loss, entity_token_accuracy, eval_report_html = evaluate_model(
            custom_ner_model,
            eval_dataloader,
            DEVICE,
            criterion,
            NUM_LABELS,
            LABELS_LIST,
            label_to_id,
        )
        print(
            f"Epoch {epoch + 1}, Eval Loss: {avg_eval_loss:.4f}, Entity Token Accuracy: {entity_token_accuracy:.4f}"
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "avg_train_loss_epoch": avg_epoch_train_loss,
                "avg_eval_loss_epoch": avg_eval_loss,
                "entity_token_accuracy_epoch": entity_token_accuracy,
                f"eval_entity_report_epoch_{epoch + 1}": wandb.Html(eval_report_html),
            }
        )

    print("\nTraining finished.")

    print("\nPerforming final evaluation on the test set...")
    final_loss, final_entity_accuracy, final_report_html = evaluate_model(
        custom_ner_model,
        eval_dataloader,
        DEVICE,
        criterion,
        NUM_LABELS,
        LABELS_LIST,
        label_to_id,
    )
    print(f"\nFinal Test Loss: {final_loss:.4f}")
    print(f"Final Test Entity Token Accuracy: {final_entity_accuracy:.4f}")
    print(
        "\nFinal Sklearn Classification Report (Entities Only):\n",
        final_report_html.replace("<pre>", "").replace("</pre>", ""),
    )

    wandb.log(
        {
            "final_test_loss": final_loss,
            "final_test_entity_accuracy": final_entity_accuracy,
            "final_entity_classification_report": wandb.Html(final_report_html),
        }
    )

    wandb.finish()

