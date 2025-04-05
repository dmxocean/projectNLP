from typing import Optional, Dict, Tuple, List
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

def _get_entities_as_set(df: pd.DataFrame) -> set:
    """
    Converts rows of the DataFrame into a set of tuples for efficient comparison.
    Each tuple represents an entity: (doc_id, line_number, start, end, label).
    """
    entities = set()
    for _, row in df.iterrows():
        entities.add((
            str(row['line_id']),
            str(row['start']),
            str(row['end']),
            str(row['label'])
        ))
    return entities

# --- Function to Calculate Entity Matching Accuracy ---

def calculate_entity_accuracy(df_true: pd.DataFrame, df_pred: pd.DataFrame, verbose: Optional[int] = 0) -> float:
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

    if verbose:
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

def _get_location_label_map(df: pd.DataFrame) -> Dict[Tuple[str, str, str, str], str]:
    """
    Creates a dictionary mapping location tuples to labels for quick lookup.
    Location tuple: (line_id, start, end).
    Assumes one unique label per location in the input df.
    If duplicates exist, the last one encountered will be stored.
    """
    location_map = {}
    # Ensure columns are strings for consistent key creation
    df_str = df[['line_id', 'start', 'end', 'label']].astype(str)
    for _, row in df_str.iterrows():
        location_key = (
            row['line_id'],
            row['start'],
            row['end']
        )
        location_map[location_key] = row['label']
    return location_map


def plot_entity_label_confusion_matrix(
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'Blues',
    values_format: str = 'd',
    normalize: Optional[str] = None,
    **kwargs # Pass extra args to ConfusionMatrixDisplay.plot
    ) -> Optional[plt.Axes]:
    """
    Generates and plots a confusion matrix for entity labels based on
    entities matched by location (line_id, start, end).

    It compares the predicted label vs the true label ONLY for entities where
    the prediction correctly identified the location span defined in the
    ground truth.

    Args:
        df_true: DataFrame with ground truth annotations.
                 Requires columns: 'line_id', 'start', 'end', 'label'.
        df_pred: DataFrame with model predictions.
                 Requires columns: 'line_id', 'start', 'end', 'label'.
        ax: Matplotlib Axes object to plot on. If None, a new figure/axes is created.
        figsize: Figure size if a new figure is created.
        cmap: Colormap for the confusion matrix plot.
        values_format: Format specifier for values in the matrix cells.
        normalize: Normalization strategy ('true', 'pred', 'all', or None).
        **kwargs: Additional keyword arguments passed to
                  sklearn.metrics.ConfusionMatrixDisplay.plot().

    Returns:
        The matplotlib Axes object containing the plot, or None if no
        location-matched entities are found.
    """
    if not all(col in df_true.columns for col in ['line_id', 'start', 'end', 'label']):
        raise ValueError("df_true missing required columns: 'line_id', 'start', 'end', 'label'")
    if not all(col in df_pred.columns for col in ['line_id', 'start', 'end', 'label']):
        raise ValueError("df_pred missing required columns: 'line_id', 'start', 'end', 'label'")

    # Create a lookup map for predicted labels based on location
    pred_location_map = _get_location_label_map(df_pred)

    y_true_labels: List[str] = []
    y_pred_labels: List[str] = []

    # Iterate through true entities to find matches in predictions by location
    # Ensure columns used for keys are strings
    df_true_str = df_true[['line_id', 'start', 'end', 'label']].astype(str)
    for _, row_true in df_true_str.iterrows():
        true_location_key = (
            row_true['line_id'],
            row_true['start'],
            row_true['end']
        )
        true_label = row_true['label']

        # Check if this exact location was predicted
        predicted_label = pred_location_map.get(true_location_key)

        if predicted_label is not None:
            # Found a location match! Compare labels.
            y_true_labels.append(true_label)
            y_pred_labels.append(predicted_label)

    if not y_true_labels:
        print("Warning: No entities with matching locations found between df_true and df_pred. Cannot generate confusion matrix.")
        return None

    # Get all unique labels present in either true or predicted matched entities
    all_labels = sorted(list(set(y_true_labels) | set(y_pred_labels)))

    # Calculate the confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=all_labels)

    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        # If using a provided ax, ensure we don't create a new figure
        fig = ax.get_figure()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)

    disp.plot(ax=ax, cmap=cmap, values_format=values_format, **kwargs)

    ax.set_title('Entity Label Confusion Matrix (Location Matched)')
    plt.xticks(rotation=45, ha='right') # Improve label readability
    plt.tight_layout() # Adjust layout

    return ax