import re
import os
import string

import nltk
import pandas as pd

from typing import Callable, Optional, List, Tuple, Dict, Any, Iterator
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

import os
import pandas as pd
import nltk # Still needed for sentence tokenizer, potentially word tokenizer
from nltk.tokenize.api import TokenizerI
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional, Set

AnnotationRow = Dict[str, Any]
LineOffsets = List[Tuple[int, int]]
LineIdToTextMap = Dict[str, str] # Stores ORIGINAL stripped line text
TokenCharSpans = List[Tuple[int, int]] # List of (start_char, end_char) tuples
LineIdToFinalTokenSpansMap = Dict[str, Optional[TokenCharSpans]] # Stores FINAL token spans

class AnnotationProcessor:
    """
    Processes document annotations:
    1. Splits documents into lines (sentences).
    2. Applies token collapsing rules to each line.
    3. Saves the processed lines (with collapsed tokens) to files.
    4. Maps original character-based annotation spans to the indices of the
       final (post-collapsing) tokens for each line.
    """

    def __init__(
        self,
        tokenizer: TokenizerI, # For word tokenization WITHIN lines BEFORE collapsing
        sentence_span_tokenizer: TokenizerI, # For sentence splitting
        rules: List[Tuple[str, str]], # Rules for collapsing
    ):
        """
        Initializes the AnnotationProcessor.

        Args:
            tokenizer: NLTK-compatible tokenizer with a span_tokenize method
                       for initial word tokenization within lines.
            sentence_span_tokenizer: NLTK-compatible tokenizer with a
                       span_tokenize method for splitting text into sentences.
            rules: A list of tuples for token collapsing, where each tuple is
                   ('input phrase', 'output_lemma').
        """
        if not hasattr(tokenizer, "span_tokenize") or not callable(
            tokenizer.span_tokenize
        ):
            raise TypeError("Word tokenizer must have a callable 'span_tokenize' method.")
        if not hasattr(sentence_span_tokenizer, "span_tokenize") or not callable(
            sentence_span_tokenizer.span_tokenize
        ):
            raise TypeError(
                "Sentence span tokenizer must have a callable 'span_tokenize' method."
            )

        self.tokenizer = tokenizer
        self.sentence_span_tokenizer = sentence_span_tokenizer
        self._preprocess_rules(rules) # Pre-process rules on init
        print(f"AnnotationProcessor initialized (using collapsing rules: {len(self.rule_map)} unique phrases).")

    def _preprocess_rules(self, rules: List[Tuple[str, str]]):
        """Pre-processes the collapsing rules for efficient lookup."""
        self.rule_map: Dict[Tuple[str, ...], str] = {}
        processed_keys: List[Tuple[str, ...]] = []

        if not rules or not isinstance(rules, list):
            print("Warning: No valid rules provided during initialization.")
            self.sorted_rule_keys = []
            return

        for phrase, lemma in rules:
            if not isinstance(phrase, str) or not isinstance(lemma, str) or not phrase:
                # print(f"Warning: Skipping invalid rule during init: ({phrase!r}, {lemma!r})")
                continue
            # Tokenize phrase based on simple whitespace split, lowercase
            phrase_tokens = tuple(phrase.lower().split())
            if not phrase_tokens:
                #  print(f"Warning: Skipping rule with empty phrase tokens during init: ({phrase!r}, {lemma!r})")
                 continue
            self.rule_map[phrase_tokens] = lemma
            processed_keys.append(phrase_tokens)

        # Sort keys by length (number of tokens) descending
        self.sorted_rule_keys = sorted(list(set(processed_keys)), key=len, reverse=True)
        # print(f"Preprocessed {len(self.sorted_rule_keys)} unique rule keys.") # Optional debug

    def _apply_collapsing_rules_and_get_tokens(
        self,
        line_text_stripped: str, # Original stripped line text
        initial_token_spans: TokenCharSpans
    ) -> Tuple[List[str], TokenCharSpans]:
        """
        Applies pre-processed collapsing rules to initial tokens and their spans.

        Returns:
            Tuple[List[str], TokenCharSpans]:
                - final_tokens: List of processed tokens (lemmas or original words).
                - final_token_char_spans: List of character spans corresponding
                  to the final_tokens, relative to line_text_stripped.
        """
        if not self.rule_map or not initial_token_spans:
            # No rules or no initial tokens, return original tokens/spans
            original_tokens = [line_text_stripped[s:e] for s,e in initial_token_spans]
            return original_tokens, initial_token_spans

        # Extract initial tokens (lowercase for matching, original case for output)
        initial_tokens_lower = [line_text_stripped[s:e].lower() for s, e in initial_token_spans]
        initial_tokens_original_case = [line_text_stripped[s:e] for s, e in initial_token_spans]


        final_tokens: List[str] = []
        final_token_char_spans: TokenCharSpans = []
        idx = 0
        num_initial_tokens = len(initial_tokens_lower)

        while idx < num_initial_tokens:
            match_found = False
            for rule_key in self.sorted_rule_keys:
                rule_len = len(rule_key)
                # Check if rule fits and matches the lowercase token slice
                if idx + rule_len <= num_initial_tokens and \
                   tuple(initial_tokens_lower[idx : idx + rule_len]) == rule_key:

                    # Match found! Get the lemma
                    lemma = self.rule_map[rule_key]
                    final_tokens.append(lemma)

                    # Combine the spans of the matched original tokens
                    start_char = initial_token_spans[idx][0] # Start char of first token
                    end_char = initial_token_spans[idx + rule_len - 1][1] # End char of last token
                    final_token_char_spans.append((start_char, end_char))

                    # Advance index past the matched tokens
                    idx += rule_len
                    match_found = True
                    break # Found longest match for this position

            if not match_found:
                # No rule matched, keep the original token (with original case) and its span
                final_tokens.append(initial_tokens_original_case[idx])
                final_token_char_spans.append(initial_token_spans[idx])
                idx += 1

        return final_tokens, final_token_char_spans

    # --- _map_char_to_token_span remains unchanged from previous correct version ---
    # It correctly maps character spans relative to a line to token indices
    # using the provided list of token character spans (which will now be
    # the final_token_char_spans).
    def _map_char_to_token_span(
        self,
        token_char_spans: TokenCharSpans, # These are the FINAL spans after collapsing
        char_start: int, # Relative to stripped line text
        char_end: int,   # Relative to stripped line text
    ) -> Tuple[Optional[int], Optional[int]]:
        """Maps character span relative to stripped line text to token span."""
        if not token_char_spans:
            return None, None
        start_token_idx = -1
        end_token_idx_inclusive = -1

        # Find start token index
        for i, (tok_s, tok_e) in enumerate(token_char_spans):
            if (tok_s <= char_start < tok_e) or (tok_s == char_start):
                start_token_idx = i; break
            elif i > 0 and tok_s > char_start and token_char_spans[i-1][1] == char_start:
                 start_token_idx = i; break
            elif i == 0 and tok_s > char_start:
                 start_token_idx = 0; break

        # If start found, find end token index (working backwards)
        if start_token_idx != -1:
            for i in range(len(token_char_spans) - 1, start_token_idx - 1, -1):
                tok_s, tok_e = token_char_spans[i]
                if char_end > tok_s: # Annotation ends after this token starts
                    end_token_idx_inclusive = i; break

        # Check if a valid span was found
        if (start_token_idx != -1 and end_token_idx_inclusive != -1 and
            start_token_idx <= end_token_idx_inclusive):
            return start_token_idx, end_token_idx_inclusive + 1 # Exclusive end
        else:
            return None, None # Mapping failed


    def _split_and_cache_lines(
        self, doc_id: str, original_text: str, silent: bool = False
    ) -> Tuple[LineOffsets, LineIdToTextMap, LineIdToFinalTokenSpansMap, List[str]]:
        """
        Splits document into lines/sentences, applies collapsing rules to tokens
        within each line, and caches necessary data. Saves processed lines
        for output file.
        """
        line_offsets: LineOffsets = []
        line_id_to_text_map: LineIdToTextMap = {} # Stores ORIGINAL stripped text
        line_id_to_final_spans_map: LineIdToFinalTokenSpansMap = {} # Stores FINAL token spans
        lines_for_file: List[str] = [] # Stores FINAL joined tokens for saving

        try:
            original_line_spans: List[Tuple[int, int]] = list(
                self.sentence_span_tokenizer.span_tokenize(original_text)
            )
            for line_num, (orig_line_s, orig_line_e) in enumerate(original_line_spans):
                orig_line_s = max(0, min(orig_line_s, len(original_text)))
                orig_line_e = max(orig_line_s, min(orig_line_e, len(original_text)))
                if orig_line_s >= orig_line_e: continue

                original_line_slice = original_text[orig_line_s:orig_line_e]
                line_text_stripped = original_line_slice.strip() # Process stripped text

                if line_text_stripped:
                    line_id = f"{doc_id}_{line_num}"
                    line_offsets.append((orig_line_s, orig_line_e))
                    # Store ORIGINAL stripped text for relative char calculations later
                    line_id_to_text_map[line_id] = line_text_stripped

                    final_tokens: List[str] = []
                    final_token_spans: Optional[TokenCharSpans] = None
                    try:
                        # 1. Get initial word token spans for the stripped line
                        initial_token_spans = list(
                            self.tokenizer.span_tokenize(line_text_stripped)
                        )
                        valid_initial_spans = [
                            (max(0, s), max(s, e)) for s, e in initial_token_spans if s <= e
                        ]

                        # 2. Apply collapsing rules to get final tokens and their spans
                        final_tokens, final_token_spans = self._apply_collapsing_rules_and_get_tokens(
                            line_text_stripped,
                            valid_initial_spans
                        )

                        # 3. Store the final token spans (relative to stripped line)
                        line_id_to_final_spans_map[line_id] = final_token_spans

                        # 4. Store the space-joined final tokens for saving to file
                        lines_for_file.append(" ".join(final_tokens))

                    except Exception as e:
                        if not silent: print(f"Error tokenizing/collapsing line {line_id}: {e}")
                        line_id_to_final_spans_map[line_id] = None # Mark as error
                        # Decide how to store line for file on error? Keep original stripped? Or skip?
                        # Let's add the original stripped text as fallback for the file.
                        lines_for_file.append(line_text_stripped)


        except Exception as e:
            if not silent: print(f"Error splitting document {doc_id} into lines: {e}")
            return [], {}, {}, []

        return line_offsets, line_id_to_text_map, line_id_to_final_spans_map, lines_for_file

    def process_data(
        self,
        data: List[Dict[str, Any]],
        output_dir: str = "../data/documents", # Default or configure as needed
        silent: bool = True,
    ) -> pd.DataFrame:
        """
        Processes documents, applies token collapsing per line, saves processed
        lines, and maps annotations to final token spans.
        """
        all_rows: List[AnnotationRow] = []
        processed_doc_count = 0

        # --- Directory Creation ---
        if not os.path.exists(output_dir):
            try: os.makedirs(output_dir); print(f"Created output dir: {output_dir}")
            except OSError as e: print(f"CRITICAL Error creating dir {output_dir}: {e}"); return pd.DataFrame()

        print(f"Processing {len(data)} documents...")
        for doc_index, doc in enumerate(data):
            # --- Document Validation ---
            if not isinstance(doc.get("data"), dict) or not all(k in doc["data"] for k in ["id", "text"]):
                if not silent: print(f"Warning: Skipping doc index {doc_index}, invalid format.")
                continue
            doc_id = str(doc["data"]["id"])
            original_text = doc["data"]["text"]
            if not original_text:
                if not silent: print(f"Warning: Skipping doc {doc_id}, empty text.")
                continue

            # --- Pass 1: Split, Apply Rules per Line, Cache ---
            line_offsets, line_id_to_text_map, line_id_to_final_spans_map, lines_for_file = (
                self._split_and_cache_lines(doc_id, original_text, silent)
            )

            # --- Save Processed line-split file ---
            if lines_for_file:
                 output_path = os.path.join(output_dir, f"{doc_id}.txt")
                 try:
                     with open(output_path, "w", encoding="utf-8") as f:
                         f.write("\n".join(lines_for_file)) # Write processed lines
                 except Exception as e:
                     if not silent: print(f"Error writing file {output_path}: {e}")

            # --- Pass 2: Process annotations ---
            if "predictions" not in doc or not doc.get("predictions"):
                processed_doc_count += 1
                # No annotations to process for this doc
                continue

            # --- Annotation Loop (largely unchanged, uses final spans) ---
            annotation_counter = 0
            for pred_idx, pred in enumerate(doc.get("predictions", [])):
                if not isinstance(pred.get("result"), list): continue

                for res_index, res in enumerate(pred.get("result", [])):
                    # --- Annotation Validation (as before) ---
                    if not isinstance(res.get("value"), dict) or not all(
                        k in res["value"] for k in ["start", "end", "labels"]): continue
                    try:
                        original_start = int(res["value"]["start"])
                        original_end = int(res["value"]["end"])
                    except (ValueError, TypeError): continue
                    labels = res["value"]["labels"]
                    if not isinstance(labels, list) or not labels: continue
                    if (original_start < 0 or original_end < original_start or
                        original_end > len(original_text)): continue # Check against original text length

                    res_id = res.get("id", f"res_{doc_index}_{pred_idx}_{res_index}_{annotation_counter}")
                    annotation_counter += 1
                    segment_text_original = original_text[original_start:original_end]

                    # --- Find Containing Line and Map Span ---
                    found_line = False
                    for line_num in range(len(line_offsets)):
                        line_s, line_e = line_offsets[line_num]
                        line_id = f"{doc_id}_{line_num}"
                        overlaps = original_start < line_e and original_end > line_s

                        if overlaps:
                            # Get ORIGINAL stripped text for relative calculations
                            original_line_text_stripped = line_id_to_text_map.get(line_id)
                            # Get FINAL token spans (post-collapsing) for mapping
                            final_token_spans = line_id_to_final_spans_map.get(line_id)

                            if original_line_text_stripped is None or final_token_spans is None:
                                continue # Problem with this line's cache

                            # Calculate char span relative to the ORIGINAL STRIPPED line text
                            original_line_slice_for_strip_calc = original_text[line_s:line_e]
                            leading_whitespace = len(original_line_slice_for_strip_calc) - len(original_line_slice_for_strip_calc.lstrip(" \t\n\r\f\v"))
                            char_start_in_line_rel_strip = max(0, original_start - line_s - leading_whitespace)
                            char_end_in_line_rel_strip = min(len(original_line_text_stripped), max(0, original_end - line_s - leading_whitespace))

                            if char_start_in_line_rel_strip >= char_end_in_line_rel_strip: continue

                            # Map relative char span to FINAL token span indices
                            token_start, token_end = self._map_char_to_token_span(
                                final_token_spans, # Use the final spans
                                char_start_in_line_rel_strip,
                                char_end_in_line_rel_strip,
                            )

                            if token_start is not None and token_end is not None:
                                for label in labels:
                                    annotation_id = f"{line_id}_{res_id}"
                                    all_rows.append({
                                        "doc_id": doc_id, "line_id": line_id,
                                        "annotation_id": annotation_id,
                                        "start": token_start, "end": token_end, # These are final token indices
                                        "label": str(label),
                                        "text": segment_text_original, # Keep original text span
                                    })
                                found_line = True
                                break # Process next annotation
                    # End loop lines
                    # if not found_line and not silent: print(...) # Optional warning
                # End loop result items
            # End loop prediction items

            processed_doc_count += 1
            if not silent and processed_doc_count % 100 == 0:
                 print(f"  Processed {processed_doc_count}/{len(data)} documents...")

        # --- DataFrame Creation ---
        df = pd.DataFrame(all_rows)
        if not df.empty:
            cols_ordered = ["doc_id", "line_id", "annotation_id", "start", "end", "label", "text"]
            df = df[[col for col in cols_ordered if col in df.columns]]

        print(f"Finished processing. Generated {len(df)} annotation rows.")
        return df