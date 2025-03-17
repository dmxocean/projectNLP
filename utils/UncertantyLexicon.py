import unicodedata
import re

def normalize(word):
    # Remove accents and diacritics
    word = ''.join(c for c in unicodedata.normalize('NFD', word) if unicodedata.category(c) != 'Mn')
    # Convert to lowercase and remove special characters
    word = re.sub(r'[^\w\s]', '', word.lower())
    return word

# Spanish uncertainty lexicon
uncertainty_lexicon_es = {
    "adverbs": ["quizas", "tal vez", "posiblemente", "probablemente", "aproximadamente", "eventualmente", "aparentemente", "dudosamente", "al parecer", "poco"],
    "verbs": ["sospech", "consider", "parec", "pod", "supon", "cre", "estim", "piens", "evalu", "analiz",
               "sugier", "interpret", "orient", "plantea", "indiqu", "valor", "impresiona", "desconoc"],
    "nouns": ["incertidumbre", "duda", "probabilidad", "posibilidad", "hipotesis", "presuncion", "evaluacion", "posible", "probable", "indeterminado", "indeterminación",
              "sospecha", "impresion"]
}

def is_uncertainty_es(word):
    word = normalize(word)
    return (word in uncertainty_lexicon_es["adverbs"] or
            any(word.startswith(prefix) for prefix in uncertainty_lexicon_es["verbs"]) or
            word in uncertainty_lexicon_es["nouns"] or
            any(word.startswith(prefix) for prefix in uncertainty_lexicon_es["prefixes"]))

# Catalan uncertainty lexicon
uncertainty_lexicon_ca = {
     "adverbs": ["potser", "tal vegada", "possiblement", "probablement", "aproximadament", "eventualment", 
                "aparentment", "dubtosament", "al sembla", "poc"],
    "verbs": ["sospit", "consider", "sembl", "pod", "supon", "creu", "estim", "pense", "avalu", "analitz", 
               "suggereix", "interpreta", "orient", "planteja", "indiqu", "valor", "impressiona", "desconeix"],
    "nouns": ["incertesa", "dubte", "probabilitat", "possibilitat", "hipòtesi", "pressumpció", "avaluació", 
              "possible", "probable", "indeterminat", "indeterminació", "sospita", "impressió"]
}

def is_uncertainty_ca(word):
    word = normalize(word)
    return (word in uncertainty_lexicon_ca["adverbs"] or
            any(word.startswith(prefix) for prefix in uncertainty_lexicon_ca["verbs"]) or
            word in uncertainty_lexicon_ca["nouns"])