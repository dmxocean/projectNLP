import os
import re
import sys
import spacy
import subprocess
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


# Global variables for spaCy models
nlp_es = None
nlp_ca = None


def load_spacy_models():
    """
    Load spaCy models for Spanish and Catalan if not already loaded
        
    Returns:
        tuple: (Spanish model, Catalan model)
    """
    global nlp_es, nlp_ca
    
    if nlp_es is None or nlp_ca is None:
        print("Loading spaCy models...")
        
        try: # Try loading existing models
            nlp_es = spacy.load("es_core_news_sm")
            nlp_ca = spacy.load("ca_core_news_sm")
            print("Both Spanish and Catalan models loaded successfully")
        except IOError as e:
            print(f"Models not found: {e}")
            print("Attempting to download spaCy models...")
            
            try: # Download models using subprocess
                print("Downloading Spanish model...")
                subprocess.check_call([sys.executable, "-m", "spacy", "download", "es_core_news_sm"])
                
                print("Downloading Catalan model...")
                subprocess.check_call([sys.executable, "-m", "spacy", "download", "ca_core_news_sm"])
                
                print("Models downloaded successfully. Loading models...")
                
                try: # Try loading the downloaded models
                    nlp_es = spacy.load("es_core_news_sm")
                    nlp_ca = spacy.load("ca_core_news_sm")
                    print("Both Spanish and Catalan models loaded successfully")
                except Exception as e:
                    print(f"Error loading downloaded models: {e}")
                    print("Please install the models manually with:")
                    print("python -m spacy download es_core_news_sm")
                    print("python -m spacy download ca_core_news_sm")
            
            except Exception as e:
                print(f"Error downloading models: {e}")
                print("Please install the models manually with:")
                print("python -m spacy download es_core_news_sm")
                print("python -m spacy download ca_core_news_sm")
    
    return nlp_es, nlp_ca


def detect_language(text, auto=False, interactive=False, default_lang="es"):
    """
    Language detection using manual input or spaCy models
    
    This function detects whether a term is Spanish (es), Catalan (ca) or both
    
    It uses a multi-tiered approach:
        - First tries to find the term in existing classifications
        - If not found, can use automatic detection with spaCy
        - Can also prompt for user input if interactive mode is enabled
        - Falls back to a default language as a last resort

    Parameters:
        text (str): The text to detect
        auto (bool): Whether to use automatic detection with spaCy models
        interactive (bool): Whether to allow interactive user input
        default_lang (str): Default language to use if other methods fail
        
    Returns:
        str: "ca" for Catalan, "es" for Spanish, "both" for words in both languages
    """
    text = text.lower().strip()
    
    try: # Try by checking existing classifications
        from words import negation, uncertainty, allclassifications
        if text in allclassifications:
            return allclassifications[text]
    except ImportError: # If words.py doesn't exist, continue with other methods
        pass
    
    if auto: # Try spaCy-based detection
        try:
            global nlp_es, nlp_ca # Load spaCy models
            nlp_es, nlp_ca = load_spacy_models()
            
            if nlp_es is None or nlp_ca is None:
                return default_lang
            
            doc_es = nlp_es(text) # Process with both models
            doc_ca = nlp_ca(text)
            
            es_recognized = sum(1 for token in doc_es if not token.is_oov) # Count recognized tokens
            ca_recognized = sum(1 for token in doc_ca if not token.is_oov)
            
            if ca_recognized > es_recognized: # If more Catalan tokens recognized, return "ca"
                return "ca"
            else: # If equal or more Spanish tokens recognized, return "es"
                return "es"
        except Exception as e:
            print(f"Error in automatic language detection: {e}") # Fall through to next method instead of returning
    
    if interactive: # Final try by interactive input if enabled
        print(f"\nClassifying: '{text}'")
        while True:
            lang = input("Enter language (es/ca/both): ").lower()
            if lang in ["es", "ca", "both"]:
                print(f"'{text}': '{lang}',")
                return lang
            else:
                print("Invalid input. Please enter 'es', 'ca', or 'both'")
    
    return default_lang # Final fallback to default language


def process_lexicons_with_language(neg_lexicon, unc_lexicon, force_rebuild=False, append_new=True):
    """
    Process lexicons with language detection and classification
    
    Processes negation and uncertainty lexicons
    - Ensure each term has proper language classification
    - Either loads existing language classifications from a file
    - Either creates new ones through interactive or automatic detection
    
    For terms in both languages, separate entries are created in the lexicons (duplicate terms)
    
    Parameters:
        neg_lexicon (DataFrame): Negation lexicon with at minimum a "term" column
        unc_lexicon (DataFrame): Uncertainty lexicon with at minimum a "term" column
        force_rebuild (bool): Whether to force rebuild the words.py file even if it exists
        append_new (bool): Whether to append new words to existing classifications
        
    Returns:
        tuple: (Updated negation lexicon DataFrame, Updated uncertainty lexicon DataFrame)
              Both returned DataFrames will have a "language" column added/updated
    """
    
    current_dir = os.path.dirname(os.path.abspath(__file__)) # Get the current directory
    words_path = os.path.join(current_dir, "words.py")
    
    # Check lang classifications saved and not forcing rebuild
    words_exist = os.path.exists(words_path) and not force_rebuild
    
    if words_exist:
        print("Loading existing language classifications from utils")
        try:
            with open(words_path, "r") as f: # Read the file directly to get all entries including duplicates
                content = f.read()
            
             # Extract pairs using regex for both dictionaries
            neg_entries = re.findall(r"negation\s*=\s*{.*?}", content, re.DOTALL)  # Allow newlines 
            unc_entries = re.findall(r"uncertainty\s*=\s*{.*?}", content, re.DOTALL) 
            
            neg_term_langs = []
            if neg_entries:
                pairs = re.findall(r"[\"']([^\"']+)[\"'](?:\s*):(?:\s*)[\"']([^\"']+)[\"']", neg_entries[0])
                neg_term_langs = pairs  # Keep original case for writing back
            
            unc_term_langs = []
            if unc_entries:
                pairs = re.findall(r"[\"']([^\"']+)[\"'](?:\s*):(?:\s*)[\"']([^\"']+)[\"']", unc_entries[0])
                unc_term_langs = pairs  # Keep original case for writing back
                # [\"']([^\"']+)[\"'] matches a string in either double or single quotes, capturing the content inside those quotes (term)
                # (?:\s*):(?:\s*) matches a colon with optional whitespace on either side (non captured group, just for matching)
                # [\"']([^\"']+)[\"'] matches another quoted string (language code)
            
            print(f"Loaded:")
            print(f"    - {len(neg_term_langs)} negation terms")
            print(f"    - {len(unc_term_langs)} uncertainty terms")
            
            neg_term_lower_to_langs = {}  # Case-insensitive mapping for negation terms
            for term, lang in neg_term_langs:
                term_lower = term.lower()
                if term_lower not in neg_term_lower_to_langs:
                    neg_term_lower_to_langs[term_lower] = []
                neg_term_lower_to_langs[term_lower].append(lang)
            
            unc_term_lower_to_langs = {}  # Case-insensitive mapping for uncertainty terms
            for term, lang in unc_term_langs:
                term_lower = term.lower()
                if term_lower not in unc_term_lower_to_langs:
                    unc_term_lower_to_langs[term_lower] = []
                unc_term_lower_to_langs[term_lower].append(lang)
            
            new_neg_words = [] # Keep track of new negation words
            for term in neg_lexicon["term"]:
                term_lower = term.lower()
                if term_lower not in neg_term_lower_to_langs:
                    new_neg_words.append(term)
            
            new_unc_words = [] # Keep track of new uncertain words
            for term in unc_lexicon["term"]:
                term_lower = term.lower()
                if term_lower not in unc_term_lower_to_langs:
                    new_unc_words.append(term)
            
            print("Found:")
            print(f"    - {len(new_neg_words)} new negation terms")
            print(f"    - {len(new_unc_words)} new uncertainty terms")
            
            print("Applying existing classifications to lexicons...")
            
            neg_lexicon_expanded = [] # Process negation terms
            for idx, row in neg_lexicon.iterrows():
                term = row["term"]
                term_lower = term.lower()
                
                if term_lower in neg_term_lower_to_langs:
                    langs = neg_term_lower_to_langs[term_lower]  # Get all language classifications for this term
                    for lang in langs:  # Create separate entry for each language to preserve duplicates
                        row_data = row.to_dict()
                        row_data["language"] = lang
                        neg_lexicon_expanded.append(row_data)
                else:
                    lang = detect_language(term, auto=True, interactive=False)  # Auto-detect for new terms
                    row_data = row.to_dict()
                    row_data["language"] = lang
                    neg_lexicon_expanded.append(row_data)
            
            neg_lexicon = pd.DataFrame(neg_lexicon_expanded)  # DataFrame with expanded entries
            
            unc_lexicon_expanded = [] # Process uncertainty terms
            for idx, row in unc_lexicon.iterrows():
                term = row["term"]
                term_lower = term.lower()
                
                if term_lower in unc_term_lower_to_langs:
                    langs = unc_term_lower_to_langs[term_lower]  # Get classifications
                    for lang in langs:  # Create entry for each language
                        row_data = row.to_dict()
                        row_data["language"] = lang
                        unc_lexicon_expanded.append(row_data)
                else:
                    lang = detect_language(term, auto=True, interactive=False)  # Auto-detect
                    row_data = row.to_dict()
                    row_data["language"] = lang
                    unc_lexicon_expanded.append(row_data)
            
            unc_lexicon = pd.DataFrame(unc_lexicon_expanded)  # DataFrame with expanded entries
            
            # DEBUG - Check if we preserved duplicates
            term_counts = neg_lexicon["term"].value_counts()
            print(f"Terms appearing multiple times in negation lexicon: {sum(term_counts > 1)}")
            term_counts = unc_lexicon["term"].value_counts()
            print(f"Terms appearing multiple times in uncertainty lexicon: {sum(term_counts > 1)}")
            
            if append_new and (new_neg_words or new_unc_words): # New words detected and append enabled
                new_neg_classifications = [] # Interactive classification for new words
                for i, term in enumerate(new_neg_words):
                    print(f"\nNew negation term {i+1}/{len(new_neg_words)}")
                    lang = detect_language(term, auto=False, interactive=True)
                    
                    if lang == "both":
                        new_neg_classifications.append((term, "es"))  # Add Spanish entry
                        new_neg_classifications.append((term, "ca"))  # Add Catalan entry
                    else:
                        new_neg_classifications.append((term, lang))
                
                new_unc_classifications = []
                for i, term in enumerate(new_unc_words):
                    print(f"\nNew uncertainty term {i+1}/{len(new_unc_words)}")
                    lang = detect_language(term, auto=False, interactive=True)
                    
                    if lang == "both":
                        new_unc_classifications.append((term, "es"))  # Add Spanish entry
                        new_unc_classifications.append((term, "ca"))  # Add Catalan entry
                    else:
                        new_unc_classifications.append((term, lang))
                
                # Update words.py file
                with open(words_path, "w") as f:
                    f.write("# Manual language classifications\n\n")
                    
                    f.write("negation = {\n")  # Negation dictionary
                    for term, lang in neg_term_langs:  # Write existing classifications
                        f.write(f"    \"{term}\": \"{lang}\",\n")
                    for term, lang in new_neg_classifications:  # Write new classifications
                        f.write(f"    \"{term}\": \"{lang}\",\n")
                    f.write("}\n\n")
                    
                    f.write("uncertainty = {\n")  # Uncertainty dictionary
                    for term, lang in unc_term_langs:  # Write existing classifications
                        f.write(f"    \"{term}\": \"{lang}\",\n")
                    for term, lang in new_unc_classifications:  # Write new classifications
                        f.write(f"    \"{term}\": \"{lang}\",\n")
                    f.write("}\n\n")
                    
                    f.write("# Combined dictionary for faster lookups\n")
                    f.write("allclassifications = {**negation, **uncertainty}\n")
                
                print(f"\nUpdated classifications saved to {words_path}")
                
                # Check terms with multiple languages
                terms_with_both_neg = sum(1 for term, langs in neg_term_lower_to_langs.items() if "es" in langs and "ca" in langs)
                terms_with_both_unc = sum(1 for term, langs in unc_term_lower_to_langs.items() if "es" in langs and "ca" in langs)
                print(f"Words in both languages - Negation: {terms_with_both_neg}")
                print(f"Words in both languages - Uncertainty: {terms_with_both_unc}")
            
            print("Existing classifications applied successfully!")
            
        except Exception as e:
            print(f"Error processing words.py: {e}")
            import traceback
            traceback.print_exc()
            words_exist = False  # If error, proceed with manual classification
    
    else:
        print("No existing classifications found. Starting manual classification.") # First run setup
        
        print(f"Detecting languages for {len(neg_lexicon['term'])} negation terms...")
        neg_term_langs = [] # Process negation terms
        for i, term in enumerate(neg_lexicon["term"]):
            print(f"\nProgress: {i+1}/{len(neg_lexicon['term'])}")
            lang = detect_language(term, auto=False, interactive=True)
            
            if lang == "both":
                neg_term_langs.append((term, "es"))  # Add Spanish entry
                neg_term_langs.append((term, "ca"))  # Add Catalan entry
            else:
                neg_term_langs.append((term, lang))
        
        neg_term_lower_to_langs = {}  # Case-insensitive mapping
        for term, lang in neg_term_langs:
            term_lower = term.lower()
            if term_lower not in neg_term_lower_to_langs:
                neg_term_lower_to_langs[term_lower] = []
            neg_term_lower_to_langs[term_lower].append(lang)
        
        neg_lexicon_expanded = []  # Create entries preserving language duplicates
        for idx, row in neg_lexicon.iterrows():
            term = row["term"]
            term_lower = term.lower()
            if term_lower in neg_term_lower_to_langs:
                for lang in neg_term_lower_to_langs[term_lower]:
                    row_data = row.to_dict()
                    row_data["language"] = lang
                    neg_lexicon_expanded.append(row_data)
        
        neg_lexicon = pd.DataFrame(neg_lexicon_expanded)  # DataFrame with expanded entries
        
        print(f"Detecting languages for {len(unc_lexicon['term'])} uncertainty terms...")
        unc_term_langs = [] # Process uncertainty terms
        for i, term in enumerate(unc_lexicon["term"]):
            print(f"\nProgress: {i+1}/{len(unc_lexicon['term'])}")
            lang = detect_language(term, auto=False, interactive=True)
            
            if lang == "both":
                unc_term_langs.append((term, "es"))  # Add Spanish entry
                unc_term_langs.append((term, "ca"))  # Add Catalan entry
            else:
                unc_term_langs.append((term, lang))
        
        unc_term_lower_to_langs = {}  # Case-insensitive mapping
        for term, lang in unc_term_langs:
            term_lower = term.lower()
            if term_lower not in unc_term_lower_to_langs:
                unc_term_lower_to_langs[term_lower] = []
            unc_term_lower_to_langs[term_lower].append(lang)
        
        unc_lexicon_expanded = []  # Create entries preserving language duplicates 
        for idx, row in unc_lexicon.iterrows():
            term = row["term"]
            term_lower = term.lower()
            if term_lower in unc_term_lower_to_langs:
                for lang in unc_term_lower_to_langs[term_lower]:
                    row_data = row.to_dict()
                    row_data["language"] = lang
                    unc_lexicon_expanded.append(row_data)
        
        unc_lexicon = pd.DataFrame(unc_lexicon_expanded)  # DataFrame with expanded entries
        
        # DEBUG - Check if we preserved duplicates
        term_counts = neg_lexicon["term"].value_counts()
        print(f"Terms appearing multiple times in negation lexicon: {sum(term_counts > 1)}")
        term_counts = unc_lexicon["term"].value_counts()
        print(f"Terms appearing multiple times in uncertainty lexicon: {sum(term_counts > 1)}")
        
        # Save classifications to file
        with open(words_path, "w") as f:
            f.write("negation = {\n")  # Negation dictionary
            for term, lang in neg_term_langs:
                f.write(f"    \"{term}\": \"{lang}\",\n")
            f.write("}\n\n")
            
            f.write("uncertainty = {\n")  # Uncertainty dictionary
            for term, lang in unc_term_langs:
                f.write(f"    \"{term}\": \"{lang}\",\n")
            f.write("}\n\n")
            
            f.write("allclassifications = {**negation, **uncertainty}\n")  # Combined lookup
        
        print(f"\nClassifications saved to {words_path}")
        
        # Check terms with multiple languages
        terms_with_both_neg = sum(1 for term, langs in neg_term_lower_to_langs.items() if "es" in langs and "ca" in langs)
        terms_with_both_unc = sum(1 for term, langs in unc_term_lower_to_langs.items() if "es" in langs and "ca" in langs)
        print(f"Words in both languages - Negation: {terms_with_both_neg}")
        print(f"Words in both languages - Uncertainty: {terms_with_both_unc}")
    
    return neg_lexicon, unc_lexicon