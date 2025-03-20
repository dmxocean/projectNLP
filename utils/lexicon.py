import csv
import os

# Cargar el CSV en un diccionario
uncertainty_lexicon_ca = {"adverb": [], "verb": [], "noun": [], "prefix": [], "group": []}

# Function to load the CSV file into a dictionary
def load_uncertainty_lexicon(file_name, lexicon_dir):
    lexicon = {"adverb": [], "verb": [], "noun": [], "prefix": [], "group": []}

    # Define the path to the file in the specific directory
    file_path = os.path.join('..','data', 'lexicons', lexicon_dir, file_name)
    
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            category = row['category']
            term = row['term']
            if category in lexicon:
                lexicon[category].append(term)
    
    return lexicon

# Load both CSV files into dictionaries
uncertainty_lexicon_ca = load_uncertainty_lexicon('uncertainty_cat.csv', 'uncertainty')
uncertainty_lexicon_es = load_uncertainty_lexicon('uncertainty_esp.csv', 'uncertainty')
negation_lexicon_ca = load_uncertainty_lexicon('negation_cat.csv', 'negation')
negation_lexicon_es = load_uncertainty_lexicon('negation_esp.csv', 'negation')

def is_uncertainty(word, language='ca'):
    word = word.lower()
    
    # Select the appropriate lexicon based on language
    if language == 'ca':
        lexicon = uncertainty_lexicon_ca
    else:
        lexicon = uncertainty_lexicon_es
    
    # Check if the word is in adverbs, nouns, groups or verbs 
    if word in lexicon["adverb"]:
        return True
    if word in lexicon["noun"]:
        return True
    if word in lexicon["group"]:
        return True
    if word in lexicon["verb"]:
        return True
    
    # Check if the word starts with a prefix 
    if any(word.startswith(prefix) for prefix in lexicon["prefix"]):
        return True

    
    return False

def is_negation(word, language='ca'):
    word = word.lower()
    
    # Select the appropriate lexicon based on language
    if language == 'ca':
        lexicon = negation_lexicon_ca
    else:
        lexicon = negation_lexicon_es
    
    # Check if the word is in adverbs, nouns or verbs
    if word in lexicon["adverb"]:
        return True
    if word in lexicon["noun"]:
        return True
    if word in lexicon["verb"]:
        return True
    
    # Check if the word starts with a prefix
    if any(word.startswith(prefix) for prefix in lexicon["prefix"]):
        return True
    
    return False
