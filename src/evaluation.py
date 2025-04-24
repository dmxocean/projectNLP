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
            pass # TODO Implement BIO tagging entity extraction
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