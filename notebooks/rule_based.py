#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import re
import sys
import pickle
import pandas as pd
import spacy
from tqdm import tqdm

import nltk


# Load previously calculated lexicons and annotations

# In[3]:


path_annotations = "../data/annotations"
path_lexicons = "../data/lexicons"
path_predictions = "../data/predictions"

os.makedirs(path_predictions, exist_ok=True)


# Load the training data

# In[4]:


df_train = pickle.load(open(os.path.join(path_annotations, "df_train.pkl"), "rb"))
print(f"Loaded {len(df_train)} annotations")


# In[5]:


df_train


# In[6]:


from nltk.tokenize import word_tokenize


# In[7]:


import nltk
from typing import List, Dict

# Assume 'grammar' is the nltk.CFG object created successfully in the previous steps
# If not, you need to run the code that builds the lemma_grammar_dict,
# converts it to cfg_string, and then calls nltk.CFG.fromstring(cfg_string)

# --- Simple Parsing (Mapping) Function ---

def parse_tokens_with_lexical_grammar(grammar: nltk.CFG, sentence_tokens: list[str]) -> list[str]:
    """
    Applies a purely lexical NLTK grammar to map tokens in a sentence.

    This function iterates through the grammar's lexical rules (LEMMA -> 'word')
    to build a word-to-lemma lookup map. It then iterates through the input
    tokens, replacing any known token (case-insensitive) with its corresponding
    lemma symbol from the grammar. Tokens not found in the grammar are
    returned unchanged.

    This is NOT syntactic parsing but rather a form of lexical normalization or tagging.

    Args:
        grammar: An nltk.CFG object containing primarily lexical rules.
                 It must have been successfully created (not None).
        sentence_tokens: A list of strings representing the tokenized sentence.

    Returns:
        A list of strings where known tokens are replaced by their lemma
        symbols from the grammar. Returns the original list if grammar is
        invalid or sentence is empty.
    """
    if not isinstance(grammar, nltk.CFG) or not sentence_tokens:
        return sentence_tokens # Return original if grammar invalid or no tokens

    word_to_lemma_map: dict[str, str] = {}
    try:
        for production in grammar.productions():
            # Check if it's a lexical rule like: LEMMA -> 'word'
            if production.is_lexical() and isinstance(production.rhs()[0], str):
                word = production.rhs()[0] # The terminal word from CFG (should be lowercase)
                lemma = production.lhs().symbol() # The non-terminal lemma string (e.g., '_negativo')
                word_to_lemma_map[word] = lemma
    except Exception as e:
        print(f"Error processing grammar productions: {e}")
        return sentence_tokens # Return original tokens on error

    if not word_to_lemma_map:
        print("Warning: No lexical rules found in the grammar to build a map.")
        # Fall through to map_tokens_to_lemmas, which will just return original tokens

    # 2. Map input tokens using the created map
    mapped_output = []
    for token in sentence_tokens:
        # Lookup the lowercase version of the token in the map
        # If not found, default to the original token itself
        lemma = word_to_lemma_map.get(token.lower(), token)
        mapped_output.append(lemma)

    return mapped_output


# In[8]:


def preprocess_text(text):
    """
    Preprocess the text for rule-based analysis
    """
    if not text or pd.isna(text):
        return ""
        
    text = text.lower().strip()
    
    return text.strip()


def tokenize_text(text):
    """Split text into tokens (words, punctuation)"""
    if not text:
        return []
    # Simple tokenization
    return word_tokenize(text)


def find_cues(text, cue_lexicon):
    """
    Find all instances of cues in the text
    
    Parameters:
        text list[str]: Tokenized text to search for cues
        cue_lexicon (DataFrame): Lexicon containing cues
        
    Returns:
        list: List of dictionaries with cue information
    """
    if not text:
        return []

    cues = []
    
    lemmas = set(cue_lexicon["term"].tolist())

    # Find each cue term in the text
    for term in lemmas:
        indices = []
        indices = [i for i, x in enumerate(text) if x == term]
        for index in indices:
            start = index
            end = index + 1
            cues.append({
                "start": start,
                "end": end,
                "text": " ".join(text[start:end]),
                "token": text[start],
                "lemma": term
            })
    
    # Sort cues by position in text
    cues = sorted(cues, key=lambda x: x["start"])
    return cues


# **Document Processing Function**
# 
# The `process_document()` function generates predictions for a single document:
# 
# For each line of text in the document:
# - Preprocess the text (lowercase, extract relevant section)
# - Find negation cues by matching terms from the lexicon
# - For each negation cue:
#    - Create a NEG prediction
#    - Determine the scope affected by this negation
#    - Create an NSCO prediction for the scope
# - Find uncertainty cues by matching terms from the lexicon
# - For each uncertainty cue:
#    - Create a UNC prediction
#    - Determine the scope affected by this uncertainty
#    - Create a USCO prediction for the scope
# 
# This approach creates predictions based only on text pattern matching, without using existing labels

# In[9]:


# load lemmatizer
import pickle
import os
import nltk

path_lexicons = "../data/lexicons"

lemma_grammar_dict = pickle.load(open(os.path.join(path_lexicons, "lemma_grammar_dict.pkl"), "rb"))

def add_lemma(grammar_dict: dict[str, set[str]], input_word: str, target_lemma: str):
    """Adds lowercase input word mapping."""
    if not target_lemma or not input_word: return # Skip empty
    # Store input word in lowercase for case-insensitive lookup later
    grammar_dict[target_lemma].add(input_word.lower())

def convert_dict_to_cfg_string(grammar_dict: dict[str, set[str]]) -> str:
    """Converts dict to CFG string."""
    cfg_rules = []
    for target_lemma in sorted(grammar_dict.keys()):
        # Input words are already lowercase in the set
        input_words = sorted(list(grammar_dict[target_lemma]))
        # Use repr() for proper quoting in CFG string format
        productions = " | ".join(repr(word) for word in input_words)
        # Ensure target_lemma is valid (basic check)
        safe_target = target_lemma.replace('<','').replace('>','') # Use content if <> included
        if not safe_target.replace('_','').isalnum() or not safe_target[0].isalpha():
             print(f"Warning: Target '{target_lemma}' converted to '{safe_target}' may still be invalid Nonterminal.")
        cfg_rules.append(f"{safe_target} -> {productions}")
    return "\n".join(cfg_rules)



# Convert the dictionary to a CFG string
cfg_string = convert_dict_to_cfg_string(lemma_grammar_dict)

grammar = nltk.CFG.fromstring(cfg_string)


# In[10]:


import nltk
from typing import List, Dict

# Assume 'grammar' is the nltk.CFG object created successfully in the previous steps
# If not, you need to run the code that builds the lemma_grammar_dict,
# converts it to cfg_string, and then calls nltk.CFG.fromstring(cfg_string)

# --- Simple Parsing (Mapping) Function ---

def parse_tokens_with_lexical_grammar(grammar: nltk.CFG, sentence_tokens: list[str]) -> list[str]:
    """
    Applies a purely lexical NLTK grammar to map tokens in a sentence.

    This function iterates through the grammar's lexical rules (LEMMA -> 'word')
    to build a word-to-lemma lookup map. It then iterates through the input
    tokens, replacing any known token (case-insensitive) with its corresponding
    lemma symbol from the grammar. Tokens not found in the grammar are
    returned unchanged.

    This is NOT syntactic parsing but rather a form of lexical normalization or tagging.

    Args:
        grammar: An nltk.CFG object containing primarily lexical rules.
                 It must have been successfully created (not None).
        sentence_tokens: A list of strings representing the tokenized sentence.

    Returns:
        A list of strings where known tokens are replaced by their lemma
        symbols from the grammar. Returns the original list if grammar is
        invalid or sentence is empty.
    """
    if not isinstance(grammar, nltk.CFG) or not sentence_tokens:
        return sentence_tokens # Return original if grammar invalid or no tokens

    word_to_lemma_map: dict[str, str] = {}
    try:
        for production in grammar.productions():
            # Check if it's a lexical rule like: LEMMA -> 'word'
            if production.is_lexical() and isinstance(production.rhs()[0], str):
                word = production.rhs()[0] # The terminal word from CFG (should be lowercase)
                lemma = production.lhs().symbol() # The non-terminal lemma string (e.g., '_negativo')
                word_to_lemma_map[word] = lemma
    except Exception as e:
        print(f"Error processing grammar productions: {e}")
        return sentence_tokens # Return original tokens on error

    if not word_to_lemma_map:
        print("Warning: No lexical rules found in the grammar to build a map.")
        # Fall through to map_tokens_to_lemmas, which will just return original tokens

    # 2. Map input tokens using the created map
    mapped_output = []
    for token in sentence_tokens:
        lemma = word_to_lemma_map.get(token.lower(), token)
        mapped_output.append(lemma)

    return mapped_output


# In[11]:


with open(os.path.join(path_lexicons, "negation/negation.csv")) as f:
    negation_lexicon = pd.read_csv(f)

with open(os.path.join(path_lexicons, "uncertainty/uncertainty.csv")) as f:
    uncertainty_lexicon = pd.read_csv(f)


# In[12]:


def process_document(doc_id, document_texts):
    """
    Process a document to detect negation and uncertainty
    
    Parameters:
        doc_id (str): Document ID
        document_texts (dict): Dictionary mapping line numbers to text
        
    Returns:
        list: List of prediction dictionaries
    """
    predictions = []
    
    cue_idx = 0
    for line_num, text in document_texts.items():
        if not text:
            continue
            
        processed_text = preprocess_text(text) # Preprocess text
        processed_text = tokenize_text(processed_text) # Tokenize text

        
        # Find negation cues
        neg_cues = find_cues(processed_text, negation_lexicon)
        
        for neg_cue in neg_cues:
            predictions.append({
                "doc_id": doc_id,
                "line_id": f"{doc_id}_{line_num}",
                "start": neg_cue["start"],
                "end": neg_cue["end"],
                "label": "NEG",
                "text": processed_text[neg_cue["start"]:neg_cue["end"]]
            }) # Add NEG prediction
            
            cues_for_scope = ["no", "sin", "sense"]
            if neg_cue["lemma"] in cues_for_scope:
                scope = {}

                # define scope as everything after the cue
                scope["start"] = neg_cue["end"]
                scope["end"] = len(processed_text)
                scope["text"] = processed_text[scope["start"]:scope["end"]]
            
                if scope["start"] < scope["end"]: # Add NSCO prediction if scope is non-empty
                    predictions.append({
                        "doc_id": doc_id,
                        "line_id": f"{doc_id}_{line_num}",
                        "start": scope["start"],
                        "end": scope["end"],
                        "label": "NSCO",
                        "text": scope["text"]
                    })

        # Find uncertainty cues
        unc_cues = find_cues(processed_text, uncertainty_lexicon)
        
        # Process each uncertainty cue
        for cue_idx, unc_cue in enumerate(unc_cues):
            # Add UNC prediction
            predictions.append({
                "doc_id": doc_id,
                "line_id": f"{doc_id}_{line_num}",
                "start": unc_cue["start"],
                "end": unc_cue["end"],
                "label": "UNC",
                "text": processed_text[unc_cue["start"]:unc_cue["end"]]
            })
    
    return predictions


# **Text Extraction Process**
# 
# The document text extraction code works as follows:
# 
# - Create an empty dictionary `doc_texts` to store all document texts
# - For each unique document ID in our dataframe:
#    - Filter the dataframe to get only rows for this document
#    - Create a dictionary `texts` to store lines for this document
#    - For each unique line number in this document:
#      - Filter to get only annotations for this line
#      - Check each annotation in this line
#      - This gives us the most complete text for this line
#    - Store the line texts dictionary in our main document dictionary
# 
# This extraction preserves the document and line structure while giving us just the text content to work with

# In[13]:


documents_directory = "../data/documents"

# loop over the documents
all_predictions = []

re_doc = re.compile(r"(\d+)\.txt$")

for document_name in tqdm(os.listdir(documents_directory)):
    match_obj = re_doc.match(document_name)
    if match_obj:
        doc_id = match_obj.group(1)

        # Load the document
        with open(os.path.join(documents_directory, f"{doc_id}.txt"), "r") as file:
            document_texts = {i: line.strip() for i, line in enumerate(file.readlines())}
        
        pred = process_document(doc_id, document_texts)
        all_predictions.extend(pred)


# In[14]:


df_train_pred = pd.DataFrame(all_predictions)
print(f"Generated {len(df_train_pred)} predictions")

print("Prediction distribution:")
print(df_train_pred["label"].value_counts())

# Save predictions to file
output_file = os.path.join(path_predictions, "df_train_predictions.pkl")
pickle.dump(df_train_pred, open(output_file, "wb"))
print(f"Saved predictions to {output_file}")


# In[15]:


df_train_pred


# In[16]:


df_train_filtered = df_train[(df_train["label"] == "NEG") | (df_train["label"] == "UNC")]
df_train_pred_filtered = df_train_pred[(df_train_pred["label"] == "NEG") | (df_train_pred["label"] == "UNC")]


# In[17]:


print(len(df_train_filtered), len(df_train_pred_filtered))


# ## Accuracy Calculation
# 
# First we calculate using the "interval", as it is done with the annotations in the training and test data.

# In[19]:


import sys
sys.path.append("..")

from utils.metrics import calculate_entity_accuracy


# All labels:

# In[20]:


accuracy = calculate_entity_accuracy(df_train, df_train_pred, verbose=1)
print(f"Accuracy: {100*accuracy:.2f}%")


# Just NEG and UNC:

# In[21]:


accuracy_filtered = calculate_entity_accuracy(df_train_filtered, df_train_pred_filtered, verbose=1)
print(f"Filtered accuracy: {100*accuracy_filtered:.2f}%")


# ### Accuracy per token
# 
# With this metric we expect to get a higher value. We consider that this is a more resonable metric:
# For example, in some scopes the final period `.` is included in it, in other no. Since we define our scopes including it, in the cases were is not included we would get a false positive, despite having all of the other words predicted correctly.
# 
# We use a function that passes an annotation dataframe (either the groud thruth) to a per-token annotations. Note that this would increase the dataframe size significantly.
# 

# In[22]:


import pandas as pd

def expand_annotations(df):
    """
    Expands annotations in a pandas DataFrame to per-token annotations.

    Args:
        df (pd.DataFrame): Input DataFrame with 'start' and 'end' columns.

    Returns:
        pd.DataFrame: DataFrame with expanded annotations.
    """

    new_rows = []
    for index, row in df.iterrows():
        start = row['start']
        end = row['end']
        
        # If the interval is just one token, keep it as is
        if start == end:
            new_rows.append(row.to_dict())
        else:
            # Expand the interval into individual tokens
            for i in range(start, end):
                new_row = row.copy()
                new_row['start'] = i
                new_row['end'] = i + 1
                new_rows.append(new_row.to_dict())
    
    return pd.DataFrame(new_rows)


# In[23]:


expanded_train = expand_annotations(df_train)
expanded_pred = expand_annotations(df_train_pred)

accuracy_expanded = calculate_entity_accuracy(expanded_train, expanded_pred, verbose=1)
print(f"Expanded accuracy: {100*accuracy_expanded:.2f}%")

