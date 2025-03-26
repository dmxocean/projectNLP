import pandas as pd

def _get_entities_as_set(df: pd.DataFrame) -> set:
    """
    Converts rows of the DataFrame into a set of tuples for efficient comparison.
    Each tuple represents an entity: (doc_id, line_number, start, end, label).
    """
    entities = set()
    for _, row in df.iterrows():
        entities.add((
            str(row['line_number']),
            str(row['label'])
        ))
    return entities

# --- Function to Calculate Entity Matching Accuracy ---

def calculate_entity_accuracy(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> float:
    """
    Calculates the overall entity matching accuracy using the Jaccard Index method.
    Accuracy = TP / (TP + FP + FN)

    An exact match requires identical 'doc_id', 'line_number', 'start', 'end', and 'label'.

    Args:
        df_true: DataFrame with ground truth annotations.
        df_pred: DataFrame with model predictions (or scrambled data).

    Returns:
        The accuracy score (float between 0.0 and 1.0).
    """
    true_entities = _get_entities_as_set(df_true)
    pred_entities = _get_entities_as_set(df_pred)

    tp = len(true_entities.intersection(pred_entities))
    fp = len(pred_entities - true_entities)
    fn = len(true_entities - pred_entities)

    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")

    denominator = tp + fp + fn

    if denominator == 0:
        # If there are no true entities and no predicted entities, accuracy is 100%
        return 1.0
    else:
        accuracy = tp / denominator
        return accuracy