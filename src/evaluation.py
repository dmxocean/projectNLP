import numpy as np
from typing import Dict, List, Tuple, Set, Any
from sklearn.metrics import precision_recall_fscore_support, classification_report
from collections import defaultdict


def compute_metrics(true_labels: List[List[str]], pred_labels: List[List[str]], is_bio: bool = False) -> Dict:
    """
    Compute precision, recall, and F1-score for the HMM model predictions considering the BIO tagging implementation

    Parameters:
        true_labels (List[List[str]]): Ground truth labels for each sequence
        pred_labels (List[List[str]]): Predicted labels for each sequence
        is_bio (bool): Whether the labels use BIO tagging

    Returns:
        Dict: Dictionary with precision, recall, and F1-score for each label
    """
    # Flatten the lists of labels
    y_true = []  # True labels
    y_pred = []  # Predicted labels

    for true_seq, pred_seq in zip(true_labels, pred_labels):  # Extend the lists with the true sentences and the predicted sentences
        y_true.extend(true_seq)
        y_pred.extend(pred_seq)

    if is_bio:  # Define labels based on tagging scheme or standard tagging
        labels = ["B-NEG", "I-NEG", "B-NSCO", "I-NSCO", "B-UNC", "I-UNC", "B-USCO", "I-USCO", "O"]
    else:
        labels = ["NEG", "NSCO", "UNC", "USCO", "O"]

    present_labels = list(set(y_true) | set(y_pred))  # Get the ground truth and predicted labels in the data
    eval_labels = [label for label in labels if label in present_labels]  # Filter to only include labels present in the data

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=eval_labels, average=None, zero_division=0)  # Compute metrics

    # Compute macro metrics unweighted mean for each label
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=eval_labels, average="macro", zero_division=0)
    # Compute weighted metrics
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=eval_labels, average="weighted", zero_division=0)

    metrics = {
        "class_metrics": {
            label: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for label, p, r, f, s in zip(eval_labels, precision, recall, f1, support)  # Compute metrics for each label
        },
        "macro_avg": {"precision": float(macro_precision), "recall": float(macro_recall), "f1": float(macro_f1)},
        "weighted_avg": {"precision": float(weighted_precision), "recall": float(weighted_recall), "f1": float(weighted_f1)},
    }

    return metrics


def print_classification_report(true_labels: List[List[str]], pred_labels: List[List[str]], is_bio: bool = False) -> None:
    """
    Classification report for the HMM model predictions

    Parameters:
        true_labels (List[List[str]]): Ground truth labels for each sequence
        pred_labels (List[List[str]]): Predicted labels for each sequence
        is_bio (bool): Whether the labels use BIO tagging
    """
    y_true = []
    y_pred = []

    for true_seq, pred_seq in zip(true_labels, pred_labels):
        y_true.extend(true_seq)
        y_pred.extend(pred_seq)

    if is_bio:
        labels = ["B-NEG", "I-NEG", "B-NSCO", "I-NSCO", "B-UNC", "I-UNC", "B-USCO", "I-USCO", "O"]
        # Filter to only include labels present in the data
        present_labels = list(set(y_true) | set(y_pred))
        labels = [label for label in labels if label in present_labels]
    else:
        labels = ["NEG", "NSCO", "UNC", "USCO", "O"]

    print(classification_report(y_true, y_pred, labels=labels, zero_division=0)) # Zero division to avoid errors


def analyze_by_language(true_labels: List[List[str]], pred_labels: List[List[str]], token_languages: List[List[str]], is_bio: bool = False) -> Dict:
    """
    Analyze performance by language

    Parameters:
        true_labels (List[List[str]]): Ground truth labels for each sequence
        pred_labels (List[List[str]]): Predicted labels for each sequence
        token_languages (List[List[str]]): Language of each token
        is_bio (bool): Whether the labels use BIO tagging scheme

    Returns:
        Dict: Performance metrics broken down by language
    """
    # Separate by language
    es_true = [] # Spanish true labels
    es_pred = []
    ca_true = [] # Catalan true labels
    ca_pred = []

    for true_seq, pred_seq, lang_seq in zip(true_labels, pred_labels, token_languages): # Iterate through the sequences by language
        for true, pred, lang in zip(true_seq, pred_seq, lang_seq):
            if lang == "es":
                es_true.append(true)
                es_pred.append(pred)
            else:  # Catalan
                ca_true.append(true)
                ca_pred.append(pred)

    es_metrics = compute_metrics([es_true], [es_pred], is_bio)
    ca_metrics = compute_metrics([ca_true], [ca_pred], is_bio)

    return {"spanish": es_metrics, "catalan": ca_metrics}


def convert_bio_to_entity_spans(bio_labels: List[str], tokens: List[Any]) -> List[Tuple[str, int, int]]:
    """
    Convert BIO labels to entity spans (entities with start and end indices)

    This function processes a list of BIO labels and identifies contiguous spans of entities
    Each span is represented as a tuple containing the (entity_type, start_index, end_index)

    Parameters:
        bio_labels (List[str]): A list of BIO labels (e.g., "B-NEG", "I-NEG", "O")
        tokens (List[Any]): A list of tokens corresponding to the BIO labels
                            (Possiblel ussage for further processing)

    Returns:
        List[Tuple[str, int, int]]: A list of entity spans, where each span is represented 
                                    as a tuple (entity_type, start_index, end_index)
    """
    entities = []  # List to store identified entity spans
    current_entity = None  # Tracks the current entity type
    start_idx = -1  # Tracks the start index of an entity (No entity has started yet required for BIO)

    for i, label in enumerate(bio_labels): # Iterate through the BIO labels
        if label.startswith("B-"):
            if current_entity is not None: # End any current entity
                entities.append((current_entity, start_idx, i - 1))  # Add the current entity span

            # Start a new entity
            current_entity = label[2:]  # Remove B- prefix
            start_idx = i

        elif label.startswith("I-"):
            entity_type = label[2:]  # Remove I- prefix

            # Continue current entity if types match, otherwise start a new entity
            if current_entity != entity_type or start_idx == -1:
                if current_entity is not None:
                    entities.append((current_entity, start_idx, i - 1))
                current_entity = entity_type
                start_idx = i

        elif label == "O": # End any current entity
            if current_entity is not None:
                entities.append((current_entity, start_idx, i - 1))
                current_entity = None # Reset current entity
                start_idx = -1 # Reset the start index

    # Add the last entity if there is one
    if current_entity is not None:
        entities.append((current_entity, start_idx, len(bio_labels) - 1))

    return entities


def get_entity_based_metrics(true_labels: List[List[str]], pred_labels: List[List[str]], is_bio: bool = False) -> Dict:
    """
    Compute entity-based metrics (treating each contiguous chunk as one entity)

    Parameters:
        true_labels (List[List[str]]): Ground truth labels for each sequence
        pred_labels (List[List[str]]): Predicted labels for each sequence
        is_bio (bool): Whether the labels use BIO tagging scheme

    Returns:
        Dict: Dictionary with entity-based precision, recall, and F1-score
    """

    def extract_entities(labels: List[str]) -> List[Tuple[str, int, int]]:
        entities = []  # List to store extracted entities

        if is_bio:
            entities = convert_bio_to_entity_spans(labels, [None] * len(labels))  # Use BIO-specific function
        else:
            current_entity = None  # Tracks the current entity type
            start_idx = 0  # Tracks the start index of an entity (No need for BIO consideration)

            for i, label in enumerate(labels):
                if label == "O":  # End current entity
                    if current_entity:
                        entities.append((current_entity, start_idx, i - 1))  # Add entity span
                        current_entity = None  # Reset current entity
                else:
                    if current_entity != label:  # Start a new entity if label changes
                        if current_entity:
                            entities.append((current_entity, start_idx, i - 1))  # Add previous entity span
                        current_entity = label  # Update current entity type
                        start_idx = i  # Update start index

            if current_entity:  # Add the last entity if it exists
                entities.append((current_entity, start_idx, len(labels) - 1))  # Add final entity span

        return entities  # Return the list of extracted entities

    true_entities = []  # True entities
    pred_entities = []  # Predicted entities

    for true_seq, pred_seq in zip(true_labels, pred_labels):  # Process each sequence
        true_entities.extend(extract_entities(true_seq))  
        pred_entities.extend(extract_entities(pred_seq)) 

    entity_types = ["NEG", "NSCO", "UNC", "USCO"] if is_bio else ["NEG", "NSCO", "UNC", "USCO"]

    correct_by_type = {entity_type: 0 for entity_type in entity_types}  # Counters for correct predictions
    pred_by_type = {entity_type: 0 for entity_type in entity_types}  # For predicted entities
    true_by_type = {entity_type: 0 for entity_type in entity_types}  # For true entities

    for entity_type, start, end in pred_entities:
        if entity_type in pred_by_type:
            pred_by_type[entity_type] += 1  # Increment predicted count for the entity type

    for entity_type, start, end in true_entities:
        if entity_type in true_by_type:
            true_by_type[entity_type] += 1  # Increment true count for the entity type
            if (entity_type, start, end) in pred_entities:  # Check for exact matches
                correct_by_type[entity_type] += 1 # True based on exact match in the predicted entities !!!

    entity_metrics = {}  # Store metrics for each entity type

    for entity_type in entity_types:
        precision = correct_by_type[entity_type] / max(1, pred_by_type[entity_type])  # Calculate precision
        recall = correct_by_type[entity_type] / max(1, true_by_type[entity_type])  # Calculate recall
        f1 = 2 * precision * recall / max(1e-10, precision + recall)  # Calculate F1-score

        entity_metrics[entity_type] = {
            "precision": precision,  
            "recall": recall,
            "f1": f1,  # F1-score
            "support": true_by_type[entity_type],  # Number of true entities for this type
        }

    # Calculate macro-averaged metrics unweighted for each label (regardless class imbalance)
    macro_precision = sum(metrics["precision"] for metrics in entity_metrics.values()) / len(entity_types)  # Macro precision
    macro_recall = sum(metrics["recall"] for metrics in entity_metrics.values()) / len(entity_types)  # Macro recall
    macro_f1 = sum(metrics["f1"] for metrics in entity_metrics.values()) / len(entity_types)  # Macro F1-score

    entity_metrics["macro_avg"] = {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1}  # Add macro averages

    return entity_metrics  # Return the computed metrics


def evaluate_scope_detection(true_labels: List[List[str]], pred_labels: List[List[str]], observations: List[List[Any]], is_bio: bool = False) -> Dict:
    """
    Evaluate negation/uncertainty scope detection more comprehensively

    Parameters:
        true_labels (List[List[str]]): Ground truth labels for each sequence
        pred_labels (List[List[str]]): Predicted labels for each sequence
        observations (List[List[Any]]): Observation sequences
        is_bio (bool): Whether the labels use BIO tagging scheme

    Returns:
        Dict: Scope detection metrics
    """

    def extract_scopes(labels: List[str], is_bio: bool = False) -> List[Tuple[str, int, int]]:
        if is_bio:
            return convert_bio_to_entity_spans(labels, [None] * len(labels))  # Use BIO-specific function
        else:
            scopes = []  # List to store extracted scopes
            current_scope = None  # Tracks the current scope type
            start_idx = -1  # Tracks the start index of the current scope

            for i, label in enumerate(labels):
                if label != "O":  # Start or continue a scope
                    if current_scope is None or current_scope != label: 
                        if current_scope is not None:
                            scopes.append((current_scope, start_idx, i - 1))  # End previous scope
                        current_scope = label  # Update current scope type
                        start_idx = i  # Update start index
                elif current_scope is not None:  # End current scope
                    scopes.append((current_scope, start_idx, i - 1))  # Add scope span
                    current_scope = None  # Reset current scope
                    start_idx = -1  # Reset start index

            if current_scope is not None:  # Add final scope if it exists
                scopes.append((current_scope, start_idx, len(labels) - 1))  # Add final scope span

            return scopes

    def calculate_overlap(true_scope: Tuple[str, int, int], pred_scope: Tuple[str, int, int]) -> float:
        true_type, true_start, true_end = true_scope
        pred_type, pred_start, pred_end = pred_scope

        if true_type != pred_type:  # Types must match
            return 0.0

        overlap_start = max(true_start, pred_start) 
        overlap_end = min(true_end, pred_end)  
        overlap_length = max(0, overlap_end - overlap_start + 1)  # Calculate overlap length

        true_length = true_end - true_start + 1  # Length of the true scope
        pred_length = pred_end - pred_start + 1  # Length of the predicted scope

        # Calculate union length (total length of both scopes minus overlap)
        union_length = true_length + pred_length - overlap_length 
        return overlap_length / union_length if union_length > 0 else 0.0  # Jaccard similarity !!!

    scope_metrics = {
        "NEG": {"exact": 0, "partial": 0, "missed": 0, "false_positive": 0},
        "NSCO": {"exact": 0, "partial": 0, "missed": 0, "false_positive": 0},
        "UNC": {"exact": 0, "partial": 0, "missed": 0, "false_positive": 0},
        "USCO": {"exact": 0, "partial": 0, "missed": 0, "false_positive": 0},
    }

    for seq_idx, (true_seq, pred_seq) in enumerate(zip(true_labels, pred_labels)):  # Process each sequence
        true_scopes = extract_scopes(true_seq, is_bio) 
        pred_scopes = extract_scopes(pred_seq, is_bio) 

        matched_pred_scopes = set()  # Track matched predicted scopes

        for true_scope in true_scopes:
            true_type = true_scope[0]
            best_match = None  # Track best matching predicted scope
            best_overlap = 0.0  # Track best overlap score

            for i, pred_scope in enumerate(pred_scopes):
                if i in matched_pred_scopes:  # Skip already matched scopes
                    continue

                overlap = calculate_overlap(true_scope, pred_scope)  # Calculate overlap
                if overlap > best_overlap:  # Update best match if overlap improves !!!
                    best_overlap = overlap
                    best_match = i

            if best_match is not None and best_overlap >= 0.7:
                scope_metrics[true_type]["exact"] += 1
                matched_pred_scopes.add(best_match)  # Mark predicted scope as matched
            elif best_match is not None and best_overlap > 0:  # Partial match
                scope_metrics[true_type]["partial"] += 1
                matched_pred_scopes.add(best_match)  # Mark predicted scope as matched
            else:  # Missed scope
                scope_metrics[true_type]["missed"] += 1

        for i, pred_scope in enumerate(pred_scopes):  # Count false positives
            if i not in matched_pred_scopes:
                pred_type = pred_scope[0]
                scope_metrics[pred_type]["false_positive"] += 1  # Increment false positive count

    result = {}

    for scope_type, metrics in scope_metrics.items():
        true_positives = metrics["exact"] + metrics["partial"] * 0.5  # Partial matches count as half
        precision = true_positives / max(1, true_positives + metrics["false_positive"])  # Calculate precision
        recall = true_positives / max(1, true_positives + metrics["missed"])  # Calculate recall
        f1 = 2 * precision * recall / max(1e-10, precision + recall)  # Calculate F1-score

        result[scope_type] = {
            "precision": precision, 
            "recall": recall, 
            "f1": f1, 
            "exact_matches": metrics["exact"], 
            "partial_matches": metrics["partial"], 
            "missed": metrics["missed"], 
            "false_positives": metrics["false_positive"], 
        }

    macro_precision = sum(m["precision"] for m in result.values()) / len(result)
    macro_recall = sum(m["recall"] for m in result.values()) / len(result)
    macro_f1 = sum(m["f1"] for m in result.values()) / len(result)

    result["macro_avg"] = {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1}  # Add macro averages

    return result  # Return the computed metrics


def convert_bio_to_standard(bio_labels: List[str]) -> List[str]:
    """
    Converts BIO labels back to standard labels (removes B-/I- prefix).

    Parameters:
        bio_labels (List[str]): List of BIO labels

    Returns:
        List[str]: List of standard labels
    """
    standard_labels = []
    for label in bio_labels:
        if label.startswith("B-") or label.startswith("I-"):
            standard_labels.append(label[2:])
        else:  # Handles "O"
            standard_labels.append(label)
    return standard_labels


def save_metrics(metrics: Dict, output_file: str) -> None:
    """
    Save evaluation metrics to a JSON file.

    Parameters:
        metrics (Dict): The metrics to save
        output_file (str): Path to save the JSON file
    """
    import json

    # Convert numpy values to Python types for JSON serialization
    def convert_to_python_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()} # Convert dict values to Python types (recursively)
        elif isinstance(obj, list):
            return [convert_to_python_types(i) for i in obj]
        else:
            return obj

    metrics = convert_to_python_types(metrics)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
