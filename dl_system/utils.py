import json
from typing import Any

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
    ],
    original_annotations: list[
        dict[str, Any]
    ],
    label_to_id: dict[str, int],
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
