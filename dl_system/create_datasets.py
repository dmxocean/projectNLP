import json

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


if __name__ == "__main__":

    json_document_train = "data/raw/training.json"
    out_file_train = "dl_system/data/training.jsonl"
    reformat_json(json_document_train, out_file_train)

    json_document_test = "data/raw/test.json"
    out_file_test = "dl_system/data/testing.jsonl"
    reformat_json(json_document_test, out_file_test)
