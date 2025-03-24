import re
import string

import nltk
import pandas as pd


def json_structure(data, max_depth=5, current_depth=0, path="root"):
    """
    Recursively explore and print the structure of JSON data up to a specified depth

    Parameters:
        data: The JSON data to explore
        max_depth: Maximum depth to explore
        current_depth: Current depth in the recursion

    Args:
        data: The JSON data to explore
        max_depth: Maximum depth to explore
        current_depth: Current depth in the recursion
        path: Current path in the JSON structure
    """

    # Indentation for visualizing depth
    indent = (4 * " ") * current_depth

    if current_depth >= max_depth:
        print(f"{indent}[Reached max depth at {path}]")
        return

    # Handle dictionary
    if isinstance(data, dict):
        print(f"{indent}{path} (dict with {len(data)} keys)")
        for key, value in data.items():
            new_path = f"{path}.{key}" if path != "root" else key
            json_structure(value, max_depth, current_depth + 1, new_path)

    # Handle list
    elif isinstance(data, list):
        print(f"{indent}{path} (list with {len(data)} items)")
        if data and current_depth < max_depth - 1:
            # Show structure of first item as an example
            sample_item = data[0]
            new_path = f"{path}[0]"
            json_structure(sample_item, max_depth, current_depth + 1, new_path)

            # If there are multiple different structures in the list, show another example
            if len(data) > 1 and not all(
                isinstance(type(item), type(sample_item)) for item in data
            ):
                different_type_item = next(
                    (item for item in data if type(item) is not type(sample_item)), None
                )
                if different_type_item:
                    new_path = f"{path}[different_type]"
                    json_structure(
                        different_type_item, max_depth, current_depth + 1, new_path
                    )

    # Handle other types
    elif isinstance(data, (str, int, float, bool, type(None))):
        # For primitive types, show a sample value
        if isinstance(data, str) and len(data) > 30:
            sample_value = f"{data[:30]}..."
        else:
            sample_value = data
        print(f"{indent}{path} ({type(data).__name__}): {sample_value}")


def json_keys(data):
    """
    Extract the key structural components of the dataset

    Parameters:
        data (list): The JSON data to explore

    Returns:
        dict: Dictionary containing the structural information
    """
    structure = {
        "main_keys": set(),
        "data_keys": set(),
        "prediction_keys": set(),
        "result_keys": set(),
        "value_keys": set(),
        "label_types": set(),
    }

    # Process documents to extract generic structure
    for doc in data[:15]:
        # Main level keys
        for key in doc.keys():
            structure["main_keys"].add(key)

        # Data level keys
        if "data" in doc:
            for key in doc["data"].keys():
                structure["data_keys"].add(key)

        # Prediction level
        if "predictions" in doc and doc["predictions"]:
            for pred in doc["predictions"]:
                for key in pred.keys():
                    structure["prediction_keys"].add(key)

                # Result level
                if "result" in pred:
                    for res in pred["result"]:
                        for key in res.keys():
                            structure["result_keys"].add(key)

                        # Value level and labels
                        if "value" in res:
                            for key in res["value"].keys():
                                structure["value_keys"].add(key)

                            # Collect label types
                            if "labels" in res["value"]:
                                for label in res["value"]["labels"]:
                                    structure["label_types"].add(label)

    # Convert sets to sorted lists for nicer output
    for key in structure:
        structure[key] = sorted(list(structure[key]))

    return structure


def extract_annotations(data):
    """
    Extract all annotations/predictions from the documents to a Dataframe

    Parameters:
        data (list): List of document JSON objects

    Returns:
        pd.DataFrame: DataFrame containing all annotations
    """

    # Dataframe new rows
    rows = []

    # Iterate through all documents
    for (
        doc_index,
        doc,
    ) in enumerate(data):
        doc_id = doc["data"]["id"]
        text = doc["data"]["text"]

        # Empty row for non predicted
        if "predictions" not in doc or not doc["predictions"]:
            row = {
                "doc_index": doc_index,  # Represent the position of each document in your data array
                "doc_id": doc_id,
                "result_id": None,  # No result ID since there are no predictions
                "start": None,  # No start position
                "end": None,  # No end position
                "label": "No Prediction",  # Special label to indicate absence
                "text": None,  # No specific text segment
            }
            rows.append(row)
            continue

        # Process predictions
        for pred_index, pred in enumerate(doc["predictions"]):
            if "result" in pred:
                for res_index, res in enumerate(pred["result"]):
                    if "value" in res and "labels" in res["value"]:
                        start = res["value"]["start"]
                        end = res["value"]["end"]

                        # Extract the labeled text segment
                        segment_text = text[start:end]

                        # Create a row for each label
                        for label in res["value"]["labels"]:
                            row = {
                                "doc_index": doc_index,
                                "doc_id": doc_id,
                                "result_id": res["id"],
                                "start": start,
                                "end": end,
                                "label": label,
                                "text": segment_text,
                            }
                            rows.append(row)

                        # Sort rows by start position
                        rows = sorted(rows, key=lambda x: x["start"])
                        rows = sorted(rows, key=lambda x: x["doc_index"])

    return pd.DataFrame(rows)


def extract_annotations_and_split_documents(data):
    """
    Extracts annotations, splits documents into lines, and adjusts annotation indices.

    This function combines the functionality of extracting annotations and
    splitting documents into lines while correctly adjusting the annotation
    indices to match the new line-based structure.

    Args:
        data (list): List of document JSON objects.

    Returns:
        pd.DataFrame: DataFrame containing all annotations, with indices
                      adjusted for line breaks.  Also saves each document
                      as a text file, split by lines.
    """

    rows = []
    for doc_index, doc in enumerate(data):
        doc_id = doc["data"]["id"]
        text = doc["data"]["text"]
        original_text = doc["data"]["text"]  # Keep the original for splitting

        # Get punctuation stops and split the document
        stops = get_punctuation_stops(text)
        line_start = 0
        line_number = 0
        processed_text = ""  # Accumulate lines with \n
        line_offsets = []  # Store (start, end) of each line in original text

        for stop in stops:
            line_text = original_text[line_start : stop + 1].strip()
            if line_text:
                line_offsets.append((line_start, stop + 1))
                processed_text += line_text + "\n"
                line_number += 1
            line_start = stop + 1

        # Handle remaining text
        remaining_text = original_text[line_start:].strip()
        if remaining_text:
            line_offsets.append((line_start, len(original_text)))
            processed_text += remaining_text + "\n"
            line_number += 1

        # Save the processed document (split into lines)
        with open(f"../data/documents/{doc_id}.txt", "w") as f:
            f.write(processed_text)

        # Handle empty predictions
        if "predictions" not in doc or not doc["predictions"]:
            row = {
                "doc_index": doc_index,
                "doc_id": doc_id,
                "result_id": None,
                "start": None,
                "end": None,
                "label": "No Prediction",
                "text": None,
                "line_number": None,  # Add line number.
            }
            rows.append(row)
            continue

        # Process predictions and adjust indices
        for pred_index, pred in enumerate(doc["predictions"]):
            if "result" in pred:
                for res_index, res in enumerate(pred["result"]):
                    if "value" in res and "labels" in res["value"]:
                        original_start = res["value"]["start"]
                        original_end = res["value"]["end"]
                        print(original_start, original_end)

                        # Find the line number and adjusted start/end
                        line_num, adjusted_start, adjusted_end = -1, -1, -1
                        for i, (line_s, line_e) in enumerate(line_offsets):
                            if line_s <= original_start <= line_e:
                                line_num = i
                                adjusted_start = original_start - line_s
                                # also adjust the end
                                adjusted_end = original_end - line_s
                                break  # Found the line

                        # Sanity Check
                        if line_num == -1:
                            print(
                                f"Warning: Could not find line for annotation in doc {doc_id}"
                            )
                            continue

                        # Now, reconstruct line text *from the line offsets and original text*
                        segment_text = original_text[
                            line_offsets[line_num][0] : line_offsets[line_num][1]
                        ][adjusted_start:adjusted_end]
                        print(adjusted_start, adjusted_end)

                        for label in res["value"]["labels"]:
                            row = {
                                "doc_index": doc_index,
                                "doc_id": doc_id,
                                "result_id": res["id"],
                                "start": adjusted_start-1,  # Adjusted start
                                "end": adjusted_end-1,  # Adjusted end
                                "label": label,
                                "text": segment_text,
                                "line_number": f"{doc_id}_{line_num}",  # Add line number
                            }
                            rows.append(row)

    # Create DataFrame and sort
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["doc_index", "line_number", "start"])
    return df


def get_punctuation_stops(text: str) -> list[int]:
    indices = []
    regex = r"\.(?!\d\w)"  # Match dots not followed by digits
    for match in re.finditer(regex, text):
        index = match.start()

        if text[index - 3 : index] == " dr":
            pass
        elif text[index - 3 : index] == "(dr":
            pass
        elif text[index - 3 : index] == "dra":
            pass
        elif text[index - 4 : index] == "(dra":
            pass
        elif text[index - 2 : index] == "..":
            pass
        elif text[index - 1 : index] == ".":
            pass
        elif text[index + 2] == ".":
            pass
        elif text[index - 1] == " ":
            pass
        elif (text[index - 1] in string.ascii_lowercase) and (
            text[index - 2] not in string.ascii_lowercase
        ):
            pass
        elif (text[index - 1] in string.ascii_lowercase) and (
            text[index + 1] in string.ascii_lowercase
        ):
            pass
        elif (text[index - 1] not in string.ascii_lowercase) and (
            text[index + 1] not in string.ascii_lowercase
        ):
            pass
        elif text[index - 6 : index] == "strept":
            pass
        elif text[index - 3 : index] == "dii":
            pass
        elif text[index - 3 : index] == "/dl":
            pass
        elif text[index - 3 : index] == "/ml":
            pass
        elif text[index - 2 : index] == "/l":
            pass
        elif text[index - 3 : index] == " ac":
            pass
        elif text[index - 4 : index] == " acs":
            pass
        else:
            indices.append(index)

    return indices


def extract_documents(data):
    """
    Extract all documents to individual text files
    """

    for doc_index, doc in enumerate(data):
        # save to txt
        with open(f"../data/documents/{doc['data']['id']}.txt", "w") as f:
            text = doc["data"]["text"].strip()
            stops = get_punctuation_stops(text)

            start = 0
            for stop in stops:
                part = text[start : stop + 1].strip()  # Include the dot
                if part:
                    f.write(part + "\n")
                start = stop + 1

            # Handle the remaining text after the last stop
            remaining = text[start:].strip()
            if remaining:
                f.write(remaining)
                f.write("\n")


def remove_punctuation(tokens):
    return [token for token in tokens if token not in string.punctuation]


def remove_punctuation_dataframe(df):
    print("hola")
    for index in df.index:
        row = df.loc[index]
        if row["label"] == "NEG" or row["label"] == "UNC":
            print(row["text"])
            tokens = nltk.tokenize.word_tokenize(row["text"])
            tokens = remove_punctuation(tokens)
            df.loc[index, "clean_text"] = " ".join(tokens)
    return df