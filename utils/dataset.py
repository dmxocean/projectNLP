import re
import os
import string

import nltk
import pandas as pd

from typing import Callable, Optional
from nltk.tokenize.api import TokenizerI  # For type hinting


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


# Define type alias for stop function
StopFunction = Callable[[str], list[int]]


class AnnotationProcessor:
    """
    Processes document annotations to split documents into lines, save them,
    and map original character-based annotation spans to token-based spans
    relative to each line.
    """

    def __init__(self, tokenizer: TokenizerI, get_stops_function: StopFunction):
        """
        Initializes the AnnotationProcessor.

        Args:
            tokenizer: A tokenizer instance that implements the NLTK TokenizerI
                       interface and crucially has a `span_tokenize` method
                       (e.g., nltk.tokenize.TreebankWordTokenizer()).
            get_stops_function: A function that takes a text string and returns
                                a sorted list of character indices representing
                                the end of segments (lines).
        """
        if not hasattr(tokenizer, "span_tokenize") or not callable(
            tokenizer.span_tokenize
        ):
            raise TypeError("Tokenizer must have a callable 'span_tokenize' method.")
        if not callable(get_stops_function):
            raise TypeError("get_stops_function must be a callable function.")

        self.tokenizer = tokenizer
        self.get_stops = get_stops_function
        print("AnnotationProcessor initialized.")

    def _map_char_to_token_span(
        self,
        token_char_spans: Optional[list[tuple[int, int]]],
        char_start: int,
        char_end: int,
    ) -> tuple[Optional[int], Optional[int]]:
        """
        Maps character span within a text (relative to its start) to a token span,
        using pre-calculated token character spans.

        Args:
            token_char_spans: List of (start_char, end_char) tuples for tokens in the text,
                              or None if tokenization failed.
            char_start: The starting character offset relative to the text start.
            char_end: The ending character offset (exclusive) relative to the text start.

        Returns:
            A tuple (token_start, token_end), where token_end is exclusive,
            or (None, None) if mapping fails.
        """
        if not token_char_spans:  # Check for None or empty list
            return None, None

        start_token_idx = -1
        end_token_idx_inclusive = -1

        # Find start token index
        for i, (tok_s, tok_e) in enumerate(token_char_spans):
            if tok_s <= char_start < tok_e or tok_s == char_start:
                start_token_idx = i
                break

        # Find end token index (inclusive) - Using revised logic
        if start_token_idx != -1:
            for i in range(len(token_char_spans) - 1, start_token_idx - 1, -1):
                tok_s, tok_e = token_char_spans[i]
                # If this token's start is before the character end offset, it's involved.
                if tok_s < char_end:
                    end_token_idx_inclusive = i
                    break

        # Check validity and return exclusive end index
        if (
            start_token_idx != -1
            and end_token_idx_inclusive != -1
            and start_token_idx <= end_token_idx_inclusive
        ):
            return start_token_idx, end_token_idx_inclusive + 1
        else:
            # print(f"Debug: Failed mapping char span ({char_start},{char_end}) with token spans: {token_char_spans[:5]}...")
            return None, None

    def _split_and_cache_lines(
        self, doc_id: str, original_text: str, silent: bool = False
    ) -> tuple[
        list[tuple[int, int]],
        dict[str, str],
        dict[str, list[tuple[int, int]]],
        list[str],
    ]:
        """
        Splits document text into lines, caches stripped text and token spans.

        Returns:
            tuple: (line_offsets, line_id_to_text_map, line_id_to_spans_map, lines_for_file)
                   line_offsets: List of (original_start, original_end) tuples for each line.
                   line_id_to_text_map: Dict mapping "docid_linenum" to stripped line text.
                   line_id_to_spans_map: Dict mapping "docid_linenum" to list of token char spans.
                   lines_for_file: List of stripped line strings for saving.
        """
        line_offsets = []
        line_id_to_text_map = {}
        line_id_to_spans_map = {}
        lines_for_file = []

        try:
            stops = self.get_stops(original_text)
            line_start = 0
            for line_num, stop in enumerate(stops):
                current_start = min(line_start, len(original_text))
                current_stop = min(stop + 1, len(original_text))
                if current_start >= current_stop:
                    continue

                original_line_slice = original_text[current_start:current_stop]
                orig_line_s = current_start
                # Calculate original end based on slice length *before* strip
                orig_line_e = current_start + len(original_line_slice)

                line_text_stripped = original_line_slice.strip()

                if line_text_stripped:
                    line_id = f"{doc_id}_{line_num}"
                    line_offsets.append((orig_line_s, orig_line_e))
                    line_id_to_text_map[line_id] = line_text_stripped
                    lines_for_file.append(line_text_stripped)
                    # Tokenize and cache spans
                    try:
                        token_spans = list(
                            self.tokenizer.span_tokenize(line_text_stripped)
                        )
                        line_id_to_spans_map[line_id] = token_spans
                    except Exception as e:
                        if not silent:
                            print(f"Error tokenizing line {line_id}: {e}")
                        line_id_to_spans_map[line_id] = None  # Mark as failed

                # Update line_start using the original stop index
                line_start = stop + 1

        except Exception as e:
            if not silent:
                print(f"Error splitting document {doc_id}: {e}")
            # Return empty structures if splitting fails
            return [], {}, {}, []

        return line_offsets, line_id_to_text_map, line_id_to_spans_map, lines_for_file

    def process_data(
        self,
        data: list[dict],
        output_dir: str = "../data/documents",
        silent: bool = False,
    ) -> pd.DataFrame:
        """
        Processes a list of documents to extract annotations, save line-split files,
        and map annotations to token indices relative to lines.

        Args:
            data: List of document JSON objects.
            output_dir: Directory to save line-split files.
            silent: Suppress warnings.

        Returns:
            DataFrame with processed annotations including 'word_idx', 'token_start', 'token_end'.
        """
        all_rows = []

        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
            except OSError as e:
                print(f"CRITICAL Error creating output directory {output_dir}: {e}")
                return pd.DataFrame()  # Cannot proceed

        print(f"Processing {len(data)} documents...")
        for doc_index, doc in enumerate(data):
            if not all(k in doc.get("data", {}) for k in ["id", "text"]):
                if not silent:
                    print(
                        f"Warning: Skipping document index {doc_index} due to missing 'data'/'id'/'text'."
                    )
                continue

            doc_id = doc["data"]["id"]
            original_text = doc["data"]["text"]

            # --- Pass 1 (Implicit): Split, cache, and get data for saving ---
            line_offsets, line_id_to_text_map, line_id_to_spans_map, lines_for_file = (
                self._split_and_cache_lines(doc_id, original_text, silent)
            )

            # --- Save line-split file ---
            if lines_for_file:
                try:
                    output_path = os.path.join(output_dir, f"{doc_id}.txt")
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(lines_for_file))
                except Exception as e:
                    if not silent:
                        print(f"Error writing file {output_path}: {e}")

            # --- Pass 2 (Implicit): Process annotations ---
            if "predictions" not in doc or not doc["predictions"]:
                continue

            for pred in doc.get("predictions", []):
                for res in pred.get("result", []):
                    if not isinstance(res.get("value"), dict) or not all(
                        k in res["value"] for k in ["start", "end", "labels"]
                    ):
                        if not silent:
                            print(
                                f"Warning: Skipping malformed annotation result in doc {doc_id}: {res.get('id', 'N/A')}"
                            )
                        continue

                    original_start = res["value"]["start"]
                    original_end = res["value"]["end"]
                    res_id = res.get(
                        "id", f"res_{doc_index}_{len(all_rows)}"
                    )  # More unique fallback ID
                    labels = res["value"]["labels"]
                    segment_text_original = original_text[original_start:original_end]

                    found_line = False
                    # Use enumerate over the *indices* of line_offsets to get line_num easily
                    for line_num in range(len(line_offsets)):
                        line_s, line_e = line_offsets[line_num]
                        line_id = f"{doc_id}_{line_num}"

                        # Check if annotation START falls within this original line span
                        if line_s <= original_start < line_e:
                            line_text_stripped = line_id_to_text_map.get(line_id)
                            token_char_spans = line_id_to_spans_map.get(line_id)

                            if line_text_stripped is None or token_char_spans is None:
                                if not silent:
                                    print(
                                        f"Error/Warning: Missing cached line text or token spans for {line_id} during annotation processing."
                                    )
                                continue  # Skip if essential cached data is missing

                            # --- Calculate char offsets relative to STRIPPED line text ---
                            # Find leading whitespace in the original slice to adjust offsets
                            original_line_slice_for_strip_calc = original_text[
                                line_s:line_e
                            ]
                            leading_whitespace = len(
                                original_line_slice_for_strip_calc
                            ) - len(
                                original_line_slice_for_strip_calc.lstrip(" \t\n\r")
                            )

                            char_start_in_line_rel_strip = max(
                                0, original_start - line_s - leading_whitespace
                            )
                            # Ensure end doesn't exceed stripped length
                            char_end_in_line_rel_strip = min(
                                len(line_text_stripped),
                                original_end - line_s - leading_whitespace,
                            )

                            if (
                                char_start_in_line_rel_strip
                                >= char_end_in_line_rel_strip
                            ):
                                if not silent:
                                    print(
                                        f"Warning: Invalid relative char span ({char_start_in_line_rel_strip}, {char_end_in_line_rel_strip}) for annot {res_id} in {line_id}."
                                    )
                                continue

                            # --- Map character span to token span using helper ---
                            token_start, token_end = self._map_char_to_token_span(
                                token_char_spans,
                                char_start_in_line_rel_strip,
                                char_end_in_line_rel_strip,
                            )

                            # --- Store results ---
                            if (
                                token_start is not None
                            ):  # Check if mapping was successful
                                for label in labels:
                                    all_rows.append(
                                        {
                                            "doc_index": doc_index,
                                            "doc_id": doc_id,
                                            "result_id": res_id,
                                            "word_idx": token_start,  # Start token index relative to line
                                            "start": token_start,  # Explicitly keep both if needed
                                            "end": token_end,  # Exclusive end token index relative to line
                                            "label": label,
                                            "text": segment_text_original,  # Text from original doc span
                                            "line_number": line_id,
                                        }
                                    )
                            elif not silent:
                                print(
                                    f"Warning: Failed mapping char span rel stripped ({char_start_in_line_rel_strip},{char_end_in_line_rel_strip}) in stripped line {line_id} for annot {res_id}"
                                )

                            found_line = True
                            break  # Annotation found in this line, move to next annotation

                    if not found_line and not silent:
                        print(
                            f"Warning: Could not find line containing annotation start ({original_start}) for result {res_id} in doc {doc_id}"
                        )

        df = pd.DataFrame(all_rows)
        print(f"Finished processing. Generated {len(df)} annotation rows.")
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
