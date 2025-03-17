import unicodedata
import re


### Change this to eliminate more symbols if necessary
def normalize(word):
    # Remove accents and diacritics
    word = ''.join(c for c in unicodedata.normalize('NFD', word) if unicodedata.category(c) != 'Mn')
    # Convert to lowercase and remove special characters
    word = re.sub(r'[^\w\s]', '', word.lower())
    return word


negation_lexicon_es = {
    "adverbs": ["no", "sin", "nunca", "tampoco", "ni", "excepto"],
    "verbs": ["neg", "rechaz", "ausen", "falt", "carec", "desaparec", "retir", "ced", 
              "desestim", "nieg", "descart", "imped", "imposib",  "retir"],
    "nouns": ["inestabilidad", "negatividad", "negativo", "neg:", "neg;", "suspendido", "atípico", 
              "indetectable", "inespecífico", "imposibilidad", "irregular", "ningun"],
    "prefixes": ["in", "des", "a", "anti", "contra", "ex"]
} #  desorientado, afebril asintomático??, exfumador

def is_negation_es(word):
    word = word.lower()
    return (word in negation_lexicon_es["adverbs"] or
            any(word.startswith(prefix) for prefix in negation_lexicon_es["verbs"]) or # Roots of verbs
            word in negation_lexicon_es["nouns"] or
            any(word.startswith(prefix) for prefix in negation_lexicon_es["prefixes"]))

negation_lexicon_ca = {
    "adverbs": ["no", "sense", "mai", "tampoc", "ni", "excepte"],
    "verbs": ["neg", "rebutj", "absen", "falt", "manc", "desapareix", "retir", "imped", 
              "descart", "ced", "deneg", "impossib"],
    "nouns": ["inestabilitat", "negativitat", "negatiu", "suspès", "retirada", "atípic", 
              "indetectable", "inespecífic", "impossibilitat", "irregular", "cap"],
    "prefixes": ["in", "des", "a", "anti", "contra", "ex"]
}

def is_negation_ca(word):
    word = word.lower()
    return (word in negation_lexicon_ca["adverbs"] or
            any(word.startswith(prefix) for prefix in negation_lexicon_es["verbs"]) or
            word in negation_lexicon_ca["nouns"] or
            any(word.startswith(prefix) for prefix in negation_lexicon_ca["prefixes"]))

