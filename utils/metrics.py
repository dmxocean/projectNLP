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


TRUE_NULL_LABEL = "Null (FP Background)"
INTERNAL_FN_LABEL = "__FN__"
DISPLAY_FN_AS_NULL_LABEL = "Null (Missed / FN)"


def plot_full_confusion_matrix( # Renamed for clarity
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
    for both False Positives (True='Null') and False Negatives (Predicted='Null').

    Interpretation:
    - Diagonal (True=X, Pred=X): True Positives for label X.
    - Off-Diagonal (True=X, Pred=Y): Classification errors.
    - 'Null (FP Background)' Row (True='Null (FP Background)', Pred=X):
        False Positives - model predicted X where no true entity existed.
    - 'Null (Missed / FN)' Column (True=X, Pred='Null (Missed / FN)'):
        False Negatives - model missed a true entity X.

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
        The matplotlib Axes object containing the plot, or None if errors occur.
    """
    # --- Input Validation (same as before) ---
    required_cols = ["line_id", "start", "end", "label"]
    if not all(col in df_true.columns for col in required_cols):
        raise ValueError(f"df_true missing required columns: {required_cols}")
    if not all(col in df_pred.columns for col in required_cols):
        raise ValueError(f"df_pred missing required columns: {required_cols}")

    try: # Added try-except for robustness
        # --- Set Operations (same as before) ---
        true_entities = _get_entities_as_set(df_true)
        pred_entities = _get_entities_as_set(df_pred)

        tp_set = true_entities.intersection(pred_entities)
        fn_set = true_entities - pred_entities # Represents missed true entities
        fp_set = pred_entities - true_entities # Represents spurious predictions

        # --- Populate y_true, y_pred (same as before, using internal labels) ---
        y_true: List[str] = []
        y_pred: List[str] = []

        for entity_tuple in tp_set: # True Positives
            label = entity_tuple[-1]
            y_true.append(label); y_pred.append(label)

        for entity_tuple in fn_set: # False Negatives
            true_label = entity_tuple[-1]
            y_true.append(true_label)
            y_pred.append(INTERNAL_FN_LABEL) # Use internal FN label

        for entity_tuple in fp_set: # False Positives
            pred_label = entity_tuple[-1]
            # Skip if model predicts the special Null label (shouldn't happen with this logic)
            if pred_label == TRUE_NULL_LABEL or pred_label == DISPLAY_FN_AS_NULL_LABEL: continue
            y_true.append(TRUE_NULL_LABEL) # True state is background/Null
            y_pred.append(pred_label)

        if not y_true:
            print("Warning: No entities found to plot. Cannot generate confusion matrix.")
            return None

        # --- Determine Labels for Calculation and Display ---
        involved_true_labels = set(y for y in y_true if y != TRUE_NULL_LABEL)
        involved_pred_labels = set(y for y in y_pred if y != INTERNAL_FN_LABEL)
        all_real_labels_in_data = sorted(list(involved_true_labels | involved_pred_labels))

        # Labels for internal calculation matrix
        matrix_labels_calc = all_real_labels_in_data + [TRUE_NULL_LABEL, INTERNAL_FN_LABEL]

        # Calculate the full confusion matrix using internal labels
        cm = confusion_matrix(y_true, y_pred, labels=matrix_labels_calc)

        # --- Prepare Matrix and Labels for Display ---
        true_null_idx = matrix_labels_calc.index(TRUE_NULL_LABEL)
        internal_fn_idx = matrix_labels_calc.index(INTERNAL_FN_LABEL)

        # Rows to display: Real Labels + True Null Row
        display_rows_indices = list(range(true_null_idx + 1))
        # Columns to display: Real Labels + FN Column (which we display as 'Null')
        display_cols_indices = list(range(true_null_idx)) + [internal_fn_idx]

        # Extract the submatrix for display
        display_cm = cm[np.ix_(display_rows_indices, display_cols_indices)]

        # Define labels for display axes, using user-friendly 'Null' variations
        display_labels_rows = all_real_labels_in_data + [TRUE_NULL_LABEL]
        # *** CHANGE HERE: Use display label for the FN column ***
        display_labels_cols = all_real_labels_in_data + [DISPLAY_FN_AS_NULL_LABEL]

        # --- Plotting (same core logic) ---
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        disp = ConfusionMatrixDisplay(
            confusion_matrix=display_cm, display_labels=display_labels_cols # Labels for X-axis
        )

        plot_kwargs = kwargs.copy()
        text_kw = plot_kwargs.get("text_kw", {})
        # Visibility check depends on values_format != None, simplified here
        if hide_zeros:
             plot_kwargs['im_kw'] = plot_kwargs.get('im_kw', {})
             # A simple way: plot and then hide text nodes if value is 0
             # This requires plotting first, then iterating ax.texts

        # Plot first
        disp.plot(ax=ax, cmap=cmap, values_format=values_format, **plot_kwargs)

        # Post-plot adjustments
        # Manually set row labels (y-axis)
        ax.set_yticks(np.arange(len(display_labels_rows)))
        ax.set_yticklabels(display_labels_rows)

        # Manually hide zeros if requested
        if hide_zeros and values_format: # only makes sense if values are displayed
            for text_obj in ax.texts:
                try:
                    # Attempt to convert text to number used in values_format
                    # This is approximate, format 'd' -> int, others -> float
                    value = float(text_obj.get_text()) if values_format != 'd' else int(text_obj.get_text())
                    if value == 0:
                        text_obj.set_visible(False)
                except ValueError:
                    continue # Ignore text that cannot be converted (e.g., '-')

        # Update Axis Labels and Title for clarity
        ax.set_ylabel(f"True Label ('{TRUE_NULL_LABEL}' row = FPs)")
        ax.set_xlabel(f"Predicted Label ('{DISPLAY_FN_AS_NULL_LABEL}' col = FNs)")
        ax.set_title("Error Confusion Matrix (using 'Null')")

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        # Consider calling tight_layout() outside if fig/ax are passed in
        # fig.tight_layout()

        return ax

    except Exception as e:
        print(f"An error occurred during confusion matrix plotting: {e}")
        return None # Return None on error


def _row_to_entity_tuple(row: pd.Series) -> Tuple[str, str, str, str]:
    """Converts a DataFrame row to the standard entity tuple for set comparison."""
    return (
        str(row["line_id"]),
        str(row["start"]),
        str(row["end"]),
        str(row["label"]),
    )

def get_specific_errors(
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame,
    target_label: str,
    error_type: str = 'FP'
) -> pd.DataFrame:
    """
    Identifies specific examples of a given error type (FP, FN, TP) for a target label.

    Args:
        df_true: DataFrame with ground truth annotations.
                 Requires columns: 'line_id', 'start', 'end', 'label'.
        df_pred: DataFrame with model predictions.
                 Requires columns: 'line_id', 'start', 'end', 'label'.
                 May contain other columns (like 'text' or 'doc_id') which will
                 be preserved in the output.
        target_label: The specific entity label to analyze errors for (e.g., 'NEG').
        error_type: The type of error to retrieve ('FP', 'FN', or 'TP').
                    Defaults to 'FP'.

    Returns:
        A pandas DataFrame containing the rows from the relevant input DataFrame
        (df_pred for FP, df_true for FN, df_pred for TP) that correspond to
        the specified error type and target label. Returns an empty DataFrame
        if no such errors are found or if input is invalid.
    """
    # Validate inputs
    required_cols = ["line_id", "start", "end", "label"]
    if not all(col in df_true.columns for col in required_cols):
        raise ValueError(f"df_true missing required columns: {required_cols}")
    if not all(col in df_pred.columns for col in required_cols):
        raise ValueError(f"df_pred missing required columns: {required_cols}")
    if error_type not in ['FP', 'FN', 'TP']:
        raise ValueError("error_type must be one of 'FP', 'FN', 'TP'")

    try:
        # 1. Get Entity Sets
        true_entities: Set[Tuple[str, ...]] = _get_entities_as_set(df_true)
        pred_entities: Set[Tuple[str, ...]] = _get_entities_as_set(df_pred)

        # 2. Calculate Error Sets based on type
        error_set: Set[Tuple[str, ...]]
        source_df: pd.DataFrame
        source_df_name: str

        if error_type == 'FP':
            error_set = pred_entities - true_entities
            source_df = df_pred
            source_df_name = "Predictions (df_pred)"
        elif error_type == 'FN':
            error_set = true_entities - pred_entities
            source_df = df_true
            source_df_name = "Ground Truth (df_true)"
        elif error_type == 'TP':
            error_set = true_entities.intersection(pred_entities)
            source_df = df_pred # Or df_true, result rows identical for required cols
            source_df_name = "Predictions/Truth (df_pred)"
        else:
             # Should not happen due to initial check, but keeps linters happy
             return pd.DataFrame()

        if not error_set:
            # print(f"No {error_type} found in total.") # Optional message
            return pd.DataFrame() # Return empty if no errors of this type exist at all

        # 3. Filter Source DataFrame
        # Find rows in the source_df where the corresponding entity tuple
        # is in the calculated error_set AND the label matches the target_label.

        # Apply the row-to-tuple conversion and check conditions
        mask = source_df.apply(
            lambda row: _row_to_entity_tuple(row) in error_set and str(row['label']) == target_label,
            axis=1
        )

        error_examples_df = source_df[mask].copy() # Use .copy() to avoid SettingWithCopyWarning later

        print(f"Found {len(error_examples_df)} examples of {error_type} for label '{target_label}' in {source_df_name}.")

        return error_examples_df

    except Exception as e:
        print(f"An error occurred during error analysis: {e}")
        return pd.DataFrame() # Return empty DataFrame on error