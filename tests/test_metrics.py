import sys
sys.path.append('..')

from utils.metrics import calculate_entity_accuracy, _get_entities_as_set

import pandas as pd
import pytest
from pytest import approx # For comparing floating point numbers
import random # Used in scramble_labels

# --- Function Definitions (Included for self-containment) ---

# Scramble function (assuming full swap map)
def scramble_labels(df: pd.DataFrame, percentage_to_change: float = 0.25, random_seed: int = None) -> pd.DataFrame:
    """
    Creates a copy of the DataFrame and randomly changes a percentage of the labels
    according to predefined swap rules (NEG<->UNC, NSCO<->USCO).
    """
    if not 0.0 <= percentage_to_change <= 1.0:
        raise ValueError("percentage_to_change must be between 0.0 and 1.0")

    df_scrambled = df.copy()
    num_rows = len(df_scrambled)
    num_rows_to_change = int(num_rows * percentage_to_change)

    if num_rows_to_change == 0 and percentage_to_change > 0:
        print("Warning: Percentage too low relative to data size to change any rows.")
    elif num_rows_to_change == 0 and percentage_to_change == 0:
         pass

    if num_rows_to_change > 0:
        label_swap_map = {
            'NEG': 'UNC', 'NSCO': 'USCO', 'UNC': 'NEG', 'USCO': 'NSCO'
        }
        if num_rows_to_change > num_rows:
             num_rows_to_change = num_rows
        indices_to_change = df_scrambled.sample(n=num_rows_to_change, random_state=random_seed).index
        for idx in indices_to_change:
            current_label = df_scrambled.loc[idx, 'label']
            new_label = label_swap_map.get(current_label, current_label)
            df_scrambled.loc[idx, 'label'] = new_label
    return df_scrambled

# --- Test Fixture ---

@pytest.fixture
def complex_df_true() -> pd.DataFrame:
    """Provides a more complex DataFrame for robust testing."""
    # Includes duplicates for set testing, and all swappable types
    data = {
        # Add columns ignored by accuracy func but needed for context/scramble
        'doc_id': [f'doc{i//5}' for i in range(12)],
        '# start': range(0, 120, 10),
        '# end': range(5, 125, 10),
        # Columns used by the accuracy function
        'line_number': ['L1', 'L1', 'L1', 'L2', 'L2', 'L3', 'L3', 'L4', 'L4', 'L4', 'L5', 'L5'],
        'label':       ['NEG', 'NEG', 'NSCO','UNC', 'NSCO','NEG', 'USCO','NEG', 'UNC', 'NSCO', 'USCO', 'OTHER']
    }
    return pd.DataFrame(data)

# --- Test Cases ---

def test_accuracy_no_scramble(complex_df_true):
    """Accuracy should be 1.0 when percentage_to_change is 0."""
    df_pred = scramble_labels(complex_df_true, percentage_to_change=0.0)
    # df_pred should be identical to df_true, sets will match perfectly
    # TP = number of unique (line, label) pairs, FP = 0, FN = 0
    accuracy = calculate_entity_accuracy(complex_df_true, df_pred)
    assert accuracy == approx(1.0)

def test_accuracy_full_scramble(complex_df_true):
    """Test accuracy when all possible labels are swapped (100% scramble)."""
    df_pred = scramble_labels(complex_df_true, percentage_to_change=1.0)

    # Manually calculate expected accuracy based on (line_number, label) sets
    true_set = {('L1', 'NEG'), ('L1', 'NSCO'), ('L2', 'UNC'), ('L2', 'NSCO'),
                ('L3', 'NEG'), ('L3', 'USCO'), ('L4', 'NEG'), ('L4', 'UNC'),
                ('L4', 'NSCO'), ('L5', 'USCO'), ('L5', 'OTHER')}

    # Expected labels after 100% scramble:
    # L1: UNC, USCO
    # L2: NEG, USCO
    # L3: UNC, NSCO
    # L4: UNC, NEG, USCO
    # L5: NSCO, OTHER (OTHER doesn't change)
    expected_pred_set = {('L1', 'UNC'), ('L1', 'USCO'), ('L2', 'NEG'), ('L2', 'USCO'),
                         ('L3', 'UNC'), ('L3', 'NSCO'), ('L4', 'UNC'), ('L4', 'NEG'),
                         ('L4', 'USCO'), ('L5', 'NSCO'), ('L5', 'OTHER')}

    # Calculate TP, FP, FN based on these sets
    tp = len(true_set.intersection(expected_pred_set)) # Should be only ('L5', 'OTHER')
    fp = len(expected_pred_set - true_set)
    fn = len(true_set - expected_pred_set)

    expected_accuracy = tp / (tp + fp + fn) # 1 / (1 + 10 + 10) = 1 / 21

    # Calculate accuracy using the function
    actual_accuracy = calculate_entity_accuracy(complex_df_true, df_pred)
    assert actual_accuracy == approx(expected_accuracy)

def test_accuracy_partial_scramble(complex_df_true):
    """Test accuracy for a partial scramble with a fixed seed."""
    seed = 42
    percentage = 0.5 # Change 50% of 12 rows = 6 rows
    df_pred = scramble_labels(complex_df_true, percentage_to_change=percentage, random_seed=seed)

    # --- Manually determine the expected outcome for this specific seed/percentage ---
    # 1. Find which rows are sampled with seed 42, n=6
    sampled_indices = complex_df_true.sample(n=6, random_state=seed).index.tolist()
    # print(f"Indices sampled for partial test: {sampled_indices}") # For debugging
    # Example output (depends on pandas version/env): e.g., [11, 2, 6, 8, 9, 1]

    # 2. Create the expected df_pred by applying swaps ONLY to sampled rows
    df_expected_pred = complex_df_true.copy()
    swap_map = {'NEG': 'UNC', 'NSCO': 'USCO', 'UNC': 'NEG', 'USCO': 'NSCO'}
    for idx in sampled_indices:
         current_label = df_expected_pred.loc[idx, 'label']
         new_label = swap_map.get(current_label, current_label)
         df_expected_pred.loc[idx, 'label'] = new_label

    # 3. Calculate expected TP, FP, FN using the _get_entities_as_set logic
    true_set = _get_entities_as_set(complex_df_true)
    expected_pred_set = _get_entities_as_set(df_expected_pred)

    exp_tp = len(true_set.intersection(expected_pred_set))
    exp_fp = len(expected_pred_set - true_set)
    exp_fn = len(true_set - expected_pred_set)
    exp_denominator = exp_tp + exp_fp + exp_fn
    expected_accuracy = 1.0 if exp_denominator == 0 else exp_tp / exp_denominator
    # --- End of manual calculation ---

    # Calculate accuracy using the function
    actual_accuracy = calculate_entity_accuracy(complex_df_true, df_pred)
    assert actual_accuracy == approx(expected_accuracy)

def test_accuracy_empty():
    """Accuracy should be 1.0 if both true and pred are empty."""
    df_true = pd.DataFrame(columns=['line_number', 'label', '# start', '# end', 'doc_id'])
    # Scrambling an empty df returns an empty df
    df_pred = scramble_labels(df_true, percentage_to_change=0.5)
    accuracy = calculate_entity_accuracy(df_true, df_pred)
    assert accuracy == approx(1.0)