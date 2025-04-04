import re
import os
import string

import nltk
import pandas as pd

from typing import Callable, Optional, List, Tuple, Dict, Any, Iterator
from nltk.tokenize.api import TokenizerI  # For type hinting


def parse_tokens_with_lexical_grammar(
    grammar: nltk.CFG, sentence_tokens: list[str]
) -> list[str]:
    """
    Applies a purely lexical NLTK grammar to map tokens in a sentence.

    This function iterates through the grammar's lexical rules (LEMMA -> 'word')
    to build a word-to-lemma lookup map. It then iterates through the input
    tokens, replacing any known token (case-insensitive) with its corresponding
    lemma symbol from the grammar. Tokens not found in the grammar are
    returned unchanged.

    This is NOT syntactic parsing but rather a form of lexical normalization or tagging.

    Args:
        grammar: An nltk.CFG object containing primarily lexical rules.
                 It must have been successfully created (not None).
        sentence_tokens: A list of strings representing the tokenized sentence.

    Returns:
        A list of strings where known tokens are replaced by their lemma
        symbols from the grammar. Returns the original list if grammar is
        invalid or sentence is empty.
    """
    if not isinstance(grammar, nltk.CFG) or not sentence_tokens:
        return sentence_tokens  # Return original if grammar invalid or no tokens

    word_to_lemma_map: dict[str, str] = {}
    try:
        for production in grammar.productions():
            # Check if it's a lexical rule like: LEMMA -> 'word'
            if production.is_lexical() and isinstance(production.rhs()[0], str):
                word = production.rhs()[
                    0
                ]  # The terminal word from CFG (should be lowercase)
                lemma = (
                    production.lhs().symbol()
                )  # The non-terminal lemma string (e.g., '_negativo')
                word_to_lemma_map[word] = lemma
    except Exception as e:
        print(f"Error processing grammar productions: {e}")
        return sentence_tokens  # Return original tokens on error

    if not word_to_lemma_map:
        print("Warning: No lexical rules found in the grammar to build a map.")
        # Fall through to map_tokens_to_lemmas, which will just return original tokens

    # 2. Map input tokens using the created map
    mapped_output = []
    for token in sentence_tokens:
        # Lookup the lowercase version of the token in the map
        # If not found, default to the original token itself
        lemma = word_to_lemma_map.get(token.lower(), token)
        mapped_output.append(lemma)

    return mapped_output


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


LineOffsets = List[Tuple[int, int]]
LineIdToTextMap = Dict[str, str]
TokenCharSpans = Optional[List[Tuple[int, int]]]
LineIdToSpansMap = Dict[str, TokenCharSpans]
AnnotationRow = Dict[str, Any]


class AnnotationProcessor:
    """
    Processes document annotations to split documents into lines (now based on
    NLTK sentence spans), save them, and map original character-based
    annotation spans to token-based spans relative to each line.
    """

    def __init__(
        self,
        tokenizer: TokenizerI,
        sentence_span_tokenizer: TokenizerI,
        grammar: Optional[nltk.CFG] = None,
    ):
        """Initializes the AnnotationProcessor."""
        # (Constructor code remains the same as the previous correct version)
        if not hasattr(tokenizer, "span_tokenize") or not callable(
            tokenizer.span_tokenize
        ):
            raise TypeError("Tokenizer must have a callable 'span_tokenize' method.")
        if not hasattr(sentence_span_tokenizer, "span_tokenize") or not callable(
            sentence_span_tokenizer.span_tokenize
        ):
            raise TypeError(
                "Sentence span tokenizer must have a callable 'span_tokenize' method."
            )
        self.tokenizer = tokenizer
        self.sentence_span_tokenizer = sentence_span_tokenizer
        self.grammar = grammar
        print("AnnotationProcessor initialized (using sentence span tokenizer).")

    # (Mapping function remains the same as the previous version - reverted to original logic)
    def _map_char_to_token_span(
        self,
        token_char_spans: TokenCharSpans,
        char_start: int,
        char_end: int,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Maps character span to token span using logic based on user's original version."""
        if not token_char_spans:
            return None, None
        start_token_idx = -1
        end_token_idx_inclusive = -1
        for i, (tok_s, tok_e) in enumerate(token_char_spans):
            if (tok_s <= char_start < tok_e) or (tok_s == char_start):
                start_token_idx = i
                break
            elif (
                i > 0
                and tok_s > char_start
                and token_char_spans[i - 1][1] == char_start
            ):
                start_token_idx = i
                break
            elif i == 0 and tok_s > char_start:
                start_token_idx = 0
                break
        if start_token_idx != -1:
            for i in range(len(token_char_spans) - 1, start_token_idx - 1, -1):
                tok_s, tok_e = token_char_spans[i]
                if char_end > tok_s:
                    end_token_idx_inclusive = i
                    break
        if (
            start_token_idx != -1
            and end_token_idx_inclusive != -1
            and start_token_idx <= end_token_idx_inclusive
        ):
            return start_token_idx, end_token_idx_inclusive + 1
        else:
            return None, None

    # (_split_and_cache_lines remains the same as the previous correct version)
    def _split_and_cache_lines(
        self, doc_id: str, original_text: str, silent: bool = False
    ) -> Tuple[LineOffsets, LineIdToTextMap, LineIdToSpansMap, List[str]]:
        """Splits document into lines/sentences and caches data."""
        line_offsets: LineOffsets = []
        line_id_to_text_map: LineIdToTextMap = {}
        line_id_to_spans_map: LineIdToSpansMap = {}
        lines_for_file: List[str] = []
        try:
            original_line_spans: List[Tuple[int, int]] = list(
                self.sentence_span_tokenizer.span_tokenize(original_text)
            )
            for line_num, (orig_line_s, orig_line_e) in enumerate(original_line_spans):
                orig_line_s = max(0, min(orig_line_s, len(original_text)))
                orig_line_e = max(0, min(orig_line_e, len(original_text)))
                if orig_line_s >= orig_line_e:
                    continue
                original_line_slice = original_text[orig_line_s:orig_line_e]
                line_text_stripped = original_line_slice.strip()
                if line_text_stripped:
                    line_id = f"{doc_id}_{line_num}"
                    line_offsets.append((orig_line_s, orig_line_e))
                    line_id_to_text_map[line_id] = line_text_stripped
                    lines_for_file.append(line_text_stripped)
                    try:
                        token_spans = list(
                            self.tokenizer.span_tokenize(line_text_stripped)
                        )
                        valid_token_spans = [
                            (max(0, s), max(s, e)) for s, e in token_spans if s <= e
                        ]
                        line_id_to_spans_map[line_id] = valid_token_spans
                    except Exception as e:
                        if not silent:
                            print(
                                f"Error tokenizing line {line_id} with word tokenizer: {e}"
                            )
                        line_id_to_spans_map[line_id] = None
        except Exception as e:
            if not silent:
                print(f"Error splitting document {doc_id} into lines/sentences: {e}")
            return [], {}, {}, []
        return line_offsets, line_id_to_text_map, line_id_to_spans_map, lines_for_file

    def process_data(
        self,
        data: List[Dict[str, Any]],
        output_dir: str = "../data/documents",
        silent: bool = True,
    ) -> pd.DataFrame:
        """
        Processes documents, saves line-split files, maps annotations.
        Uses original logic for accessing annotations via doc['predictions'].
        """
        all_rows: List[AnnotationRow] = []
        processed_doc_count = 0

        # (Directory creation code remains the same)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
            except OSError as e:
                print(f"CRITICAL Error creating output directory {output_dir}: {e}")
                return pd.DataFrame()

        print(f"Processing {len(data)} documents...")
        for doc_index, doc in enumerate(data):
            # (Document validation remains the same)
            if not isinstance(doc.get("data"), dict) or not all(
                k in doc["data"] for k in ["id", "text"]
            ):
                if not silent:
                    print(
                        f"Warning: Skipping document index {doc_index} due to missing 'data'/'id'/'text'."
                    )
                continue
            doc_id = str(doc["data"]["id"])
            original_text = doc["data"]["text"]
            if not original_text:
                if not silent:
                    print(
                        f"Warning: Skipping document {doc_id} (index {doc_index}) due to empty text."
                    )
                continue

            # --- Pass 1: Split and cache ---
            line_offsets, line_id_to_text_map, line_id_to_spans_map, lines_for_file = (
                self._split_and_cache_lines(doc_id, original_text, silent)
            )

            # --- Save line-split file ---
            # (File saving logic remains the same)
            if lines_for_file:
                output_path = os.path.join(output_dir, f"{doc_id}.txt")
                try:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(lines_for_file))
                except Exception as e:
                    if not silent:
                        print(f"Error writing file {output_path}: {e}")

            # ****************************************************************
            # *** Pass 2: Process annotations - Reverted Access Logic      ***
            # ****************************************************************

            # Check for 'predictions' key directly, as in the original code
            if "predictions" not in doc or not doc.get(
                "predictions"
            ):  # Use .get() for safety after check
                processed_doc_count += 1
                if not silent:
                    print(
                        f"--- Doc {doc_id}: No 'predictions' found or empty. Skipping annotations. ---"
                    )  # DEBUG
                continue  # Skip to next document if no predictions

            if not silent:
                print(
                    f"\n--- Debugging Annotations for Doc: {doc_id} ---"
                )  # DEBUG START DOC

            # Iterate directly over the list obtained from doc.get("predictions", [])
            annotation_counter = (
                0  # Counter for fallback IDs if 'id' is missing in result
            )
            for pred_idx, pred in enumerate(
                doc.get("predictions", [])
            ):  # pred is one item in the predictions list
                # Check if the prediction item itself has a 'result' list
                if not isinstance(pred.get("result"), list):
                    if not silent:
                        print(
                            f"  DEBUG: Skipping item in 'predictions' at index {pred_idx}, 'result' is not a list."
                        )  # DEBUG
                    continue

                # Iterate through the 'result' list within the prediction item
                for res_index, res in enumerate(
                    pred.get("result", [])
                ):  # res is one annotation result
                    if not silent:
                        print(
                            f"\n  Processing Annotation Result index {res_index} (raw): {res.get('id', 'N/A')}"
                        )  # DEBUG

                    # --- Annotation Validation (remains the same) ---
                    if not isinstance(res.get("value"), dict) or not all(
                        k in res["value"] for k in ["start", "end", "labels"]
                    ):
                        if not silent:
                            print(
                                f"    DEBUG: Malformed annotation value, skipping."
                            )  # DEBUG
                        continue
                    try:
                        original_start = int(res["value"]["start"])
                        original_end = int(res["value"]["end"])
                    except (ValueError, TypeError):
                        if not silent:
                            print(
                                f"    DEBUG: Non-integer start/end, skipping."
                            )  # DEBUG
                        continue
                    labels = res["value"]["labels"]
                    if not isinstance(labels, list) or not labels:
                        if not silent:
                            print(
                                f"    DEBUG: Invalid/empty labels list, skipping."
                            )  # DEBUG
                        continue
                    if (
                        original_start < 0
                        or original_end < original_start
                        or original_end > len(original_text)
                    ):
                        if not silent:
                            print(
                                f"    DEBUG: Invalid original span [{original_start}:{original_end}], skipping."
                            )  # DEBUG
                        continue

                    # --- If validation passes (remains the same) ---
                    res_id = res.get(
                        "id",
                        f"res_{doc_index}_{pred_idx}_{res_index}_{annotation_counter}",
                    )  # Made fallback ID more unique
                    annotation_counter += 1
                    segment_text_original = original_text[original_start:original_end]
                    if not silent:
                        print(
                            f"    Annotation OK: ID={res_id}, Orig Span=[{original_start}:{original_end}], Text='{segment_text_original[:50]}...'"
                        )

                    # --- Find Containing Line (Sentence) and Map Span (remains the same) ---
                    found_line = False
                    for line_num in range(len(line_offsets)):
                        line_s, line_e = line_offsets[line_num]
                        line_id = f"{doc_id}_{line_num}"
                        overlaps = original_start < line_e and original_end > line_s
                        if not silent:
                            print(
                                f"      Checking Line {line_num} (Span=[{line_s}:{line_e}]): Overlaps = {overlaps}"
                            )

                        if overlaps:
                            line_text_stripped = line_id_to_text_map.get(line_id)
                            token_char_spans = line_id_to_spans_map.get(line_id)
                            if line_text_stripped is None or token_char_spans is None:
                                if not silent:
                                    print(
                                        f"      DEBUG: Missing cached data for {line_id}, skipping line."
                                    )
                                continue

                            original_line_slice_for_strip_calc = original_text[
                                line_s:line_e
                            ]
                            leading_whitespace = len(
                                original_line_slice_for_strip_calc
                            ) - len(
                                original_line_slice_for_strip_calc.lstrip(" \t\n\r\f\v")
                            )
                            char_start_in_line_rel_strip = max(
                                0, original_start - line_s - leading_whitespace
                            )
                            char_end_in_line_rel_strip = min(
                                len(line_text_stripped),
                                max(0, original_end - line_s - leading_whitespace),
                            )

                            if not silent:
                                print(
                                    f"        Relative Char Span in Stripped: [{char_start_in_line_rel_strip}:{char_end_in_line_rel_strip}] (Stripped len: {len(line_text_stripped)})"
                                )

                            if (
                                char_start_in_line_rel_strip
                                >= char_end_in_line_rel_strip
                            ):
                                if not silent:
                                    print(
                                        f"        DEBUG: Invalid relative char span after strip calc, skipping line."
                                    )
                                continue

                            token_start, token_end = self._map_char_to_token_span(
                                token_char_spans,
                                char_start_in_line_rel_strip,
                                char_end_in_line_rel_strip,
                            )
                            if not silent:
                                print(
                                    f"        Mapped Token Span: [{token_start}:{token_end}]"
                                )

                            if token_start is not None and token_end is not None:
                                if not silent:
                                    print(
                                        f"        SUCCESS: Mapping successful. Adding row(s)."
                                    )
                                for label in labels:
                                    annotation_id = (
                                        f"{line_id}_{res_id}"  # Unique annotation ID
                                    )
                                    all_rows.append(
                                        {
                                            "doc_id": doc_id,
                                            "line_id": line_id,
                                            "annotation_id": annotation_id,
                                            "start": token_start,
                                            "end": token_end,
                                            "label": str(label),
                                            "text": segment_text_original,
                                        }
                                    )
                                found_line = True
                                break  # Found the line, process next annotation result
                            else:
                                if not silent:
                                    print(
                                        f"        DEBUG: _map_char_to_token_span returned None, mapping failed for this line."
                                    )

                    # --- End of loop for one line/sentence ---

                    # --- Warning if annotation not mapped (remains the same) ---
                    if not found_line and not silent:
                        print(
                            f"    WARNING: Annotation ID={res_id} (Span=[{original_start}:{original_end}]) was not successfully mapped to any line."
                        )

                # --- End of loop for annotation results (res) ---
            # --- End of loop for predictions (pred) ---

            processed_doc_count += 1  # Increment doc counter here
            if not silent and processed_doc_count % 50 == 0:
                print(f"  Processed {processed_doc_count}/{len(data)} documents...")

        # --- DataFrame Creation (remains the same) ---
        df = pd.DataFrame(all_rows)
        if not df.empty:
            cols_ordered = [
                "doc_id",
                "line_id",
                "annotation_id",
                "start",
                "end",
                "label",
                "text",
            ]
            df = df[[col for col in cols_ordered if col in df.columns]]

        print(f"Finished processing. Generated {len(df)} annotation rows.")
        return df

    # (_apply_lexical_grammar remains the same)
    def _apply_lexical_grammar(self, sentence_tokens: list[str]) -> list[str]:
        """Applies a lexical grammar to normalize tokens."""
        if self.grammar:
            print(
                "Warning: Lexical grammar application is defined but not implemented/called."
            )
            return sentence_tokens
        else:
            return sentence_tokens

