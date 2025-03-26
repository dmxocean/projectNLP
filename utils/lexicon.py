import os
import pandas as pd

def load_lexicon(lexicon_type, lang="ALL"):
    """
    Load lexicon files based on type and language

    Parameters:
        lexicon_type (str): "negation" or "uncertainty"
        lang (str): Language code "es", "ca", or "ALL"

    Returns:
        pandas.DataFrame: Combined lexicon DataFrame
    """

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Get base path relative to script location
    path = os.path.join(base_path, "data", "lexicons", lexicon_type)  # Full path to lexicon dir

    # Map language codes to possible file suffixes
    lang_mapping = {
        # "es": ["es", "esp"], # TODO The idea is to only have "es" as the language code
        # "ca": ["ca", "cat"], # TODO The idea is to only have "cat" as the language code
        "es": ["es"],
        "ca": ["ca"], 
        "ALL": ["ALL"]
    }

    # Load specific language file if not ALL
    if lang != "ALL":
        for lang_variant in lang_mapping[lang]:
            file_path = os.path.join(path, f"{lexicon_type}_{lang_variant}.csv")
            if os.path.exists(file_path):
                return pd.read_csv(file_path)  # TODO Return first match for now

    # Load combined ALL file
    all_file_path = os.path.join(path, f"{lexicon_type}_ALL.csv")
    if os.path.exists(all_file_path):
        return pd.read_csv(all_file_path)

    return pd.DataFrame(columns=["term", "freq", "language", "POS"])  # Empty fallback


# Cache to store loaded lexicons
_lexicon_cache = {}

def get_lexicon(lexicon_type, lang="ALL"):
    """
    Get lexicon and cache it for future use

    Parameters:
        lexicon_type (str): "negation" or "uncertainty"
        lang (str): Language code - "es", "ca", or "ALL"

    Returns:
        pandas.DataFrame: Cached lexicon DataFrame
    """
    cache_key = f"{lexicon_type}_{lang}"
    if cache_key not in _lexicon_cache:
        _lexicon_cache[cache_key] = load_lexicon(lexicon_type, lang) 
    return _lexicon_cache[cache_key]


def organize_by_POS(lexicon_df):
    """
    Organize lexicon terms by POS categories

    Parameters:
        lexicon_df (pandas.DataFrame): Lexicon DataFrame with Term and POS columns

    Returns:
        dict: POS categories as keys, lists of terms as values
    """

    
    organized = { # Initialize POS categories
        "adverb": [], 
        "verb": [], 
        "noun": [], 
        "adjective": [],
        "preposition": [],
        "prefix": [],
        "phrase": [],
        "other": []
    }

    # Categorize each term by POS
    for _, row in lexicon_df.iterrows():
        pos = row["POS"] if "POS" in row else "other"  # Default to "other" if POS missing
        term = row["term"]
        if pos in organized:
            organized[pos].append(term.lower())
        else:
            organized["other"].append(term.lower())  # Unknown POS goes to "other"

    return organized


def is_uncertainty(word, lang="ALL"):
    """
    Check if a word is in the uncertainty lexicon

    Parameters:
        word (str): Word to check
        lang (str): Language code "es", "ca", or "ALL"

    Returns:
        bool: True if word is in uncertainty lexicon
    """

    word = word.lower()
    lexicon_df = get_lexicon("uncertainty", lang)  # Load lexicon

    
    if any(term.lower() == word for term in lexicon_df["term"]): # Direct match check
        return True

    if any(word in term.lower().split() for term in lexicon_df["term"]): # Multi-word term check
        return True

    lexicon = organize_by_POS(lexicon_df) # Check by POS categories
    for category in ["adverb", "verb", "noun", "adjective", "phrase"]:
        if word in lexicon[category]:
            return True

    if any(word.startswith(prefix) for prefix in lexicon["prefix"]): # Prefix check
        return True

    return False


def is_negation(word, lang="ALL"):
    """
    Check if a word is in the negation lexicon

    Parameters:
        word (str): Word to check
        lang (str): Language code "es", "ca", or "ALL"

    Returns:
        bool: True if word is in negation lexicon
    """

    word = word.lower()
    lexicon_df = get_lexicon("negation", lang)  # Load lexicon

    # Direct match check
    if any(term.lower() == word for term in lexicon_df["term"]):
        return True

    # Multi-word term check
    if any(word in term.lower().split() for term in lexicon_df["term"]):
        return True

    # Check by POS categories
    lexicon = organize_by_POS(lexicon_df)
    for category in ["adverb", "verb", "noun", "adjective", "phrase"]: # TODO Add more categories if needed
        if word in lexicon[category]:
            return True

    # Prefix check
    if any(word.startswith(prefix) for prefix in lexicon["prefix"]):
        return True

    return False


def get_all_negation(lang="ALL"):
    """
    Get all negation terms for a language

    Returns:
        list: List of negation terms
    """
    lexicon_df = get_lexicon("negation", lang)
    return lexicon_df["term"].tolist()


def get_all_uncertainty(lang="ALL"):
    """
    Get all uncertainty terms for a language

    Returns:
        list: List of uncertainty terms
    """
    lexicon_df = get_lexicon("uncertainty", lang)
    return lexicon_df["term"].tolist()