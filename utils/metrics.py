from typing import Optional, Tuple, List
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


def calculate_entity_accuracy(
    df_true: pd.DataFrame, df_pred: pd.DataFrame, verbose: Optional[int] = 0
) -> float:
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
        return 1.0
    else:
        accuracy = tp / denominator
        return accuracy


# --- Definition of _get_entities_as_set (Ensure string conversion) ---
def _get_entities_as_set(df: pd.DataFrame) -> set:
    """
    Converts rows of the DataFrame into a set of tuples for efficient comparison.
    Each tuple represents an entity: (line_id, start, end, label).
    Ensures elements within the tuple are strings for consistent hashing.
    """
    entities = set()
    # Select relevant columns and ensure they are strings
    df_str = df[["line_id", "start", "end", "label"]].astype(str)
    for _, row in df_str.iterrows():
        entities.add(
            (
                row["line_id"],
                row["start"],
                row["end"],
                row["label"],
            )
        )
    return entities


NULL_LABEL = "Null"  # Represents the ground truth state of "no entity" / Background
FN_LABEL = "__FN__"  # Represents a False Negative (a true entity was missed)


def plot_full_confusion_matrix(  # Renamed for clarity
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    values_format: str = "d",
    hide_zeros: bool = False,
    **kwargs,  # Pass extra args to ConfusionMatrixDisplay.plot
) -> Optional[plt.Axes]:
    """
    Generates and plots a confusion matrix focusing on errors, using 'Null'
    to represent the background/non-entity state in the ground truth.

    - Diagonal (True=X, Pred=X): True Positives for label X.
    - Off-Diagonal (True=X, Pred=Y): Classification errors (model predicted Y, true was X).
    - 'Null' Row (True=Null, Pred=X): False Positives (model predicted X where no entity existed).
    - '__FN__' Column (True=X, Pred=__FN__): False Negatives (model missed a true entity X).

    Args:
        df_true: DataFrame with ground truth annotations. Requires columns:
                 'line_id', 'start', 'end', 'label'.
        df_pred: DataFrame with model predictions. Requires columns:
                 'line_id', 'start', 'end', 'label'.
        ax: Matplotlib Axes object to plot on. If None, a new figure/axes is created.
        figsize: Figure size if a new figure is created.
        cmap: Colormap for the confusion matrix plot.
        values_format: Format specifier for values in the matrix cells.
        hide_zeros: If True, cells with zero values are not displayed.
        **kwargs: Additional keyword arguments passed to
                  sklearn.metrics.ConfusionMatrixDisplay.plot().

    Returns:
        The matplotlib Axes object containing the plot, or None if no
        true or predicted entities exist.
    """
    if not all(col in df_true.columns for col in ["line_id", "start", "end", "label"]):
        raise ValueError(
            "df_true missing required columns: 'line_id', 'start', 'end', 'label'"
        )
    if not all(col in df_pred.columns for col in ["line_id", "start", "end", "label"]):
        raise ValueError(
            "df_pred missing required columns: 'line_id', 'start', 'end', 'label'"
        )

    true_entities = _get_entities_as_set(df_true)
    pred_entities = _get_entities_as_set(df_pred)

    tp_set = true_entities.intersection(pred_entities)
    fn_set = (
        true_entities - pred_entities
    )  # True entities not found exactly in predictions
    fp_set = (
        pred_entities - true_entities
    )  # Predicted entities not found exactly in true

    y_true: List[str] = []
    y_pred: List[str] = []

    # 1. Add True Positives
    for entity_tuple in tp_set:
        true_label = entity_tuple[-1]  # Label is the last element
        y_true.append(true_label)
        y_pred.append(true_label)

    # 2. Add False Negatives (Missed True Entities)
    for entity_tuple in fn_set:
        true_label = entity_tuple[-1]
        y_true.append(true_label)
        y_pred.append(FN_LABEL)  # This true entity was missed (predicted as FN)

    # 3. Add False Positives (Spurious Predictions)
    for entity_tuple in fp_set:
        pred_label = entity_tuple[-1]
        # If the model predicts 'Null', we ignore it based on user request.
        # This shouldn't happen if 'Null' isn't a possible model output label.
        if pred_label == NULL_LABEL:
            continue
        y_true.append(
            NULL_LABEL
        )  # This prediction corresponds to no true entity (true is Null)
        y_pred.append(pred_label)

    if not y_true:
        print(
            "Warning: No entities found to plot (check input dataframes and matching logic). Cannot generate confusion matrix."
        )
        return None

    # Determine all possible real labels involved
    # Use labels present in the actual data points being plotted
    involved_true_labels = set(y for y in y_true if y != NULL_LABEL)
    involved_pred_labels = set(y for y in y_pred if y != FN_LABEL)
    # Also consider labels that might only appear in TPs (less common edge case)
    all_real_labels_in_data = sorted(list(involved_true_labels | involved_pred_labels))

    # Define the full set of labels for the matrix axes calculation
    # Order: Real labels, then Null row-label, then FN column-label
    matrix_labels = all_real_labels_in_data + [NULL_LABEL, FN_LABEL]

    # Calculate the confusion matrix using the full set of labels
    cm = confusion_matrix(y_true, y_pred, labels=matrix_labels)

    # --- Prepare Matrix for Display ---
    # We want rows for Real Labels + Null, and columns for Real Labels + FN
    null_idx = matrix_labels.index(NULL_LABEL)
    fn_idx = matrix_labels.index(FN_LABEL)

    # Rows to display: All rows up to and including the Null row
    display_rows_indices = list(range(null_idx + 1))
    # Columns to display: All columns for real labels + the FN column
    display_cols_indices = list(range(null_idx)) + [
        fn_idx
    ]  # Indices of real labels + FN index

    # Extract the submatrix for display
    display_cm = cm[np.ix_(display_rows_indices, display_cols_indices)]

    # Define labels for display axes
    display_labels_rows = all_real_labels_in_data + [NULL_LABEL]
    display_labels_cols = all_real_labels_in_data + [FN_LABEL]

    # --- Plotting ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    disp = ConfusionMatrixDisplay(
        confusion_matrix=display_cm, display_labels=display_labels_cols
    )  # X-axis labels

    # Modify plot arguments if hiding zeros
    plot_kwargs = kwargs.copy()
    # Simple way to hide zero text is just not to display text if value is 0
    text_kw = plot_kwargs.get("text_kw", {})
    text_kw[
        "visible"
    ] = not hide_zeros  # Hide all text if hide_zeros=True (simplification)
    # Or more complex: iterate text objects after plotting if needed.

    disp.plot(ax=ax, cmap=cmap, values_format=values_format, **plot_kwargs)

    # Manually set row labels (y-axis) and potentially adjust ticks
    ax.set_yticks(np.arange(len(display_labels_rows)))
    ax.set_yticklabels(display_labels_rows)

    ax.set_ylabel("True Label / Type ('Null' row = FPs)")
    ax.set_xlabel("Predicted Label / Type ('__FN__' col = FNs)")

    ax.set_title("Error Confusion Matrix (Null=FP / __FN__=FN)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    return ax

