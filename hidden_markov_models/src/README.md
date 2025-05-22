# Comprehensive Analysis of Negation and Uncertainty Detection in Multilingual Medical Texts

## Table of Contents

1. [Introduction](#introduction)
2. [Data Format and Structure](#data-format-and-structure)
3. [Preprocessing Pipeline](#preprocessing-pipeline)
   - [Language Detection](#language-detection)
   - [Text Cleaning and Normalization](#text-cleaning-and-normalization)
   - [Tokenization and Feature Extraction](#tokenization-and-feature-extraction)
   - [Token-to-Annotation Mapping](#token-to-annotation-mapping)
   - [BIO Tagging Scheme](#bio-tagging-scheme)
   - [Observation Sequence Creation](#observation-sequence-creation)
   - [Vocabulary Building](#vocabulary-building)
4. [Hidden Markov Models](#hidden-markov-models)
   - [Basic HMM Principles](#basic-hmm-principles)
   - [Baseline HMM Implementation](#baseline-hmm-implementation)
   - [BIO+POS Enhanced HMM](#biopos-enhanced-hmm)
   - [Second-Order HMM with BIO+POS](#second-order-hmm-with-biopos)
5. [Evaluation Metrics](#evaluation-metrics)
   - [Token-Level Evaluation](#token-level-evaluation)
   - [Entity-Level Evaluation](#entity-level-evaluation)
   - [Scope Detection Evaluation](#scope-detection-evaluation)
   - [Language-Specific Analysis](#language-specific-analysis)
6. [Confusion Matrix Analysis](#confusion-matrix-analysis)
   - [Understanding Confusion Matrices](#understanding-confusion-matrices)
   - [Analyzing Tag Transitions](#analyzing-tag-transitions)
   - [Common Error Patterns](#common-error-patterns)
7. [Model Comparison](#model-comparison)
   - [Baseline vs. BIO+POS](#baseline-vs-biopos)
   - [BIO+POS vs. Second-Order](#biopos-vs-second-order)
   - [Performance on Entity Types](#performance-on-entity-types)
   - [Language-Specific Performance](#language-specific-performance)
8. [Code Implementation Details](#code-implementation-details)
   - [HMM Base Class](#hmm-base-class)
   - [Viterbi Algorithm](#viterbi-algorithm)
   - [Model Training Process](#model-training-process)
   - [Smoothing Strategies](#smoothing-strategies)
9. [Conclusion and Future Work](#conclusion-and-future-work)
10. [References](#references)

## Introduction

This document provides a comprehensive analysis of a project implementing Hidden Markov Models (HMMs) for the task of negation and uncertainty detection in multilingual Spanish and Catalan medical texts. In clinical documentation, understanding whether a condition is negated, uncertain, or affirmed is critical for accurate information extraction and decision support.

The core task is to identify four entity types within the text:

- **NEG**: Negation markers – words or phrases that express negation (e.g., "no", "sin", "ausencia de")
- **NSCO**: Scope of negation – the linguistic span affected by a negation marker
- **UNC**: Uncertainty markers – words or phrases that express uncertainty (e.g., "posible", "sospecha de")
- **USCO**: Scope of uncertainty – the linguistic span affected by an uncertainty marker

For example, in the sentence "El paciente no presenta fiebre" (The patient does not present fever):
- "no" would be labeled as NEG (negation marker)
- "presenta fiebre" would be labeled as NSCO (scope of negation)

The project implements three progressively sophisticated models:

1. **Baseline HMM**: A standard first-order HMM using only word tokens as observations
2. **BIO+POS Enhanced HMM**: A first-order HMM incorporating Begin-Inside-Outside tagging and Part-of-Speech features
3. **Second-Order HMM with BIO+POS**: An advanced trigram HMM that captures longer dependencies while using BIO tagging and POS features

The implementation spans multiple files:
- `preprocessing.ipynb`: Data preparation pipeline
- `hmm.py`: Core HMM model implementations
- `evaluation.py`: Comprehensive evaluation metrics
- Model-specific notebooks: `hmm_baseline.ipynb`, `hmm_pos_bio.ipynb`, `hmm_second_order.ipynb`

## Data Format and Structure

### Raw Data Format

The raw data consists of multilingual medical texts in Spanish and Catalan, with character-level annotations for negation markers, negation scopes, uncertainty markers, and uncertainty scopes. The data is stored in JSON format with the following structure:

```json
{
  "data": {
    "text": "El paciente no presenta fiebre ni dolor abdominal."
  },
  "predictions": [
    {
      "result": [
        {
          "value": {
            "start": 12,
            "end": 14,
            "text": "no",
            "labels": ["NEG"]
          }
        },
        {
          "value": {
            "start": 15,
            "end": 40,
            "text": "presenta fiebre ni dolor",
            "labels": ["NSCO"]
          }
        }
      ]
    }
  ]
}
```

Each document contains:
- The raw text in the `data.text` field
- Annotations in the `predictions.result` array
- Each annotation has character-level `start` and `end` positions, the annotated `text`, and `labels` indicating the entity type

### Annotation Types

The annotations follow these key patterns:

1. **Negation markers (NEG)** are typically short words or phrases that express negation
2. **Negation scopes (NSCO)** are the linguistic spans affected by the negation
3. **Uncertainty markers (UNC)** express doubt or possibility
4. **Uncertainty scopes (USCO)** are the linguistic spans affected by the uncertainty

Multiple annotations can overlap, necessitating careful processing during tokenization and label assignment.

## Preprocessing Pipeline

The preprocessing pipeline transforms raw JSON medical text into structured sequences suitable for HMM training and evaluation. This section details each step of this complex process.

### Language Detection

A critical feature of this dataset is that it contains mixed-language documents, with text in both Spanish and Catalan. Accurate processing requires identifying the language of each sentence for proper tokenization and normalization.

#### Sentence-Level Language Detection

Language detection is performed at the sentence level rather than the document level, using the `langdetect` library:

```python
def detect_sentence_language(text: str) -> str:
    """
    Detect the language of a given sentence
    
    Args:
        text: Text to detect language from
        
    Returns:
        str: Language code ('es' for Spanish, 'ca' for Catalan)
    """
    try:
        # Try to detect language using langdetect
        lang = detect(text)
        # Map to our language codes
        if lang == 'ca':
            return 'ca'
        elif lang == 'es':
            return 'es'
        else:
            # Default to Spanish for other cases
            return 'es'
    except LangDetectException:
        # Default to Spanish if detection fails
        return 'es'
```

This approach is crucial because:

1. Documents may contain sentences in both languages
2. Each language requires specific tokenization and POS tagging models
3. Language-specific cleaning rules need to be applied

The processing pipeline segments documents into sentences using spaCy's sentence boundary detection, then processes each sentence with the appropriate language-specific models.

#### Language Statistics

During preprocessing, the system tracks language distribution:

```python
# Track languages found at document level
doc_languages = set()

# In the processing loop
sent_lang = detect_sentence_language(sent_text)
doc_languages.add(sent_lang)
```

This information helps analyze language-specific model performance in later evaluation stages.

### Text Cleaning and Normalization

Medical texts contain numerous abbreviations, measurements, and domain-specific notation that require careful normalization while preserving important clinical information.

#### Language-Specific Cleaning

The cleaning process applies language-specific rules:

```python
def clean_text_by_language(text: str, language: str, es_replacements: Dict[str, str] = None, 
                           ca_replacements: Dict[str, str] = None) -> str:
    # Default Spanish replacements
    default_es_replacements = {
        # Common medical abbreviations
        "d.": "de",
        "dr.": "doctor",
        "dra.": "doctora",
        "t.a.": "tensión arterial",
        "tto.": "tratamiento",
        # Contractions
        "a el": "al",
        "d el": "del",
    }
    
    # Default Catalan replacements
    default_ca_replacements = {
        # Common medical abbreviations
        "dr.": "doctor",
        "dra.": "doctora",
        "hosp.": "hospital",
        # More Catalan-specific replacements...
    }
    
    # Apply language-specific replacements...
```

#### Handling Apostrophes in Catalan

Catalan has specific apostrophe patterns that require special handling:

```python
# Handle Catalan apostrophes
apostrophe_replacements = {"d'": "de ", "l'": "el ", "s'": "se ", "n'": "en ", 
                          "m'": "me ", "t'": "te "}

for orig, repl in apostrophe_replacements.items():
    temp_cleaned = temp_cleaned.replace(orig, repl)
```

#### Preserving Clinical Measurements

To preserve critical medical information, the cleaning process identifies and protects various numerical patterns:

```python
# Patterns to preserve
patterns = [
    # Decimal numbers (both . and , as decimal separators)
    (r"\b\d+[\.,]\d+\b", "DECIMAL_"),
    # Dates in various formats
    (r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b", "DATE_"),
    # Times
    (r"\b\d{1,2}:\d{1,2}(?::\d{1,2})?\b", "TIME_"),
    # Measurements with units (including temperature)
    (r"\b\d+(?:[\.,]\d+)?[\s-]*(?:mg|kg|g|ml|l|cm|mm|mmHg)\b", "MEASURE_"),
    (r"\b\d+(?:[\.,]\d+)?[\s-]*°[CF]\b", "TEMP_"),
    # Percentages
    (r"\b\d+(?:[\.,]\d+)?[\s-]*%\b", "PERCENT_"),
    # Range expressions (common in dosing)
    (r"\b\d+-\d+\b", "RANGE_"),
    # Blood pressure values
    (r"\b\d+\/\d+\b", "BP_"),
    # Lab values with units
    (r"\b\d+(?:[\.,]\d+)?[\s-]*(?:g\/dl|mg\/dl|mmol\/l|µg|ng\/ml|ui\/l|u\/ml)\b", "LAB_"),
]
```

The cleaning process:
1. Replaces these patterns with unique placeholders
2. Performs general cleaning (removing punctuation, normalizing whitespace)
3. Restores the original patterns from the placeholders

This approach ensures that critical medical information is preserved intact while standardizing the rest of the text for model training.

### Tokenization and Feature Extraction

Tokenization is performed using language-specific spaCy models:

```python
# Load spaCy models for Spanish and Catalan
nlp_es = spacy.load("es_core_news_sm")
nlp_ca = spacy.load("ca_core_news_sm")

# In the processing loop
nlp = nlp_ca if sent_lang == "ca" else nlp_es  # Select appropriate language model
spacy_sent = nlp(sent_text)  # Process with correct language model
```

#### Token Feature Extraction

For each token, the system extracts a rich set of features:

```python
token_info = {
    "text": token.text,            # Original text
    "cleaned_text": cleaned_text.lower(),  # Normalized text
    "pos": token.pos_,             # Part-of-Speech tag
    "lemma": token.lemma_,         # Lemmatized form
    "start": abs_token_start,      # Character position (start)
    "end": abs_token_end,          # Character position (end)
    "label": label,                # Assigned label from annotations
    "language": sent_lang,         # Language of the sentence
}
```

These features enable:
1. Mapping tokens to character-level annotations
2. Creating different observation formats for different models
3. Analyzing language-specific patterns and performance

The Part-of-Speech (POS) tags are particularly important for the enhanced models, as they provide grammatical information about each token's function.

### Token-to-Annotation Mapping

A crucial step is mapping character-level annotations to token-level labels. This is handled by the `get_token_label` function:

```python
def get_token_label(token_span: Dict, annotations: List[Dict], use_bio: bool = False) -> str:
    """
    Determines the label for a token span based on character-level annotations
    
    Parameters:
        token_span (Dict): Token info with 'start' and 'end' character positions
        annotations (List[Dict]): List of annotation spans from the raw data
        use_bio (bool): Whether to use BIO tagging scheme
        
    Returns:
        str: Label for the token (e.g., 'NEG', 'O', 'B-NEG', 'I-UNC')
    """
    token_start = token_span["start"]
    token_end = token_span["end"]
    
    overlaps = []  # Store overlapping annotations with priority
    priority = {"UNC": 4, "USCO": 3, "NEG": 2, "NSCO": 1}
    
    # Find overlapping annotations
    for annotation in annotations:
        if "value" not in annotation or "labels" not in annotation["value"] or not annotation["value"]["labels"]:
            continue
            
        anno_start = annotation["value"]["start"]
        anno_end = annotation["value"]["end"]
        label = annotation["value"]["labels"][0]
        
        if label not in priority:  # Ignore irrelevant labels if any
            continue
            
        # Calculate overlap fraction relative to the token length
        overlap_start = max(token_start, anno_start)
        overlap_end = min(token_end, anno_end)
        overlap_length = max(0, overlap_end - overlap_start)
        token_length = max(1, token_end - token_start)  # Avoid division by zero
        
        # Define overlap threshold (e.g., > 50% of token must overlap)
        if overlap_length / token_length > 0.5:
            overlaps.append({
                "label": label, 
                "start": anno_start, 
                "end": anno_end, 
                "priority": priority[label]
            })
    
    # No overlapping annotations
    if not overlaps:
        return "O"
        
    # Find the highest priority overlapping annotation
    best_overlap = max(overlaps, key=lambda x: x["priority"])
    best_label = best_overlap["label"]
    
    # Standard labeling (not BIO)
    if not use_bio:
        return best_label
    else:
        # Determine B- or I- tag for BIO scheme
        # Allow slight tolerance for tokenization differences vs annotation boundaries
        is_beginning = (token_start <= best_overlap["start"] < token_end or 
                        abs(token_start - best_overlap["start"]) <= 1)
        
        if is_beginning:
            return f"B-{best_label}"
        else:
            # Ensure the token is actually inside the span (not just overlapping the end slightly)
            token_midpoint = token_start + (token_end - token_start) / 2
            if best_overlap["start"] <= token_midpoint < best_overlap["end"]:
                return f"I-{best_label}"
            else:
                # Should not happen often if overlap > 0.5, but as a fallback
                return f"B-{best_label}"
```

This function handles several important edge cases:

1. **Multiple overlapping annotations**: Resolved using a priority system (UNC > USCO > NEG > NSCO)
2. **Partial overlaps**: Using a 50% threshold to determine if a token should receive a label
3. **Tokenization mismatches**: Allowing slight tolerance for differences between tokenization and annotation boundaries
4. **BIO scheme determination**: Identifying beginning vs. inside positions for BIO tagging

### BIO Tagging Scheme

The BIO (Begin-Inside-Outside) tagging scheme is crucial for capturing entity boundaries more precisely.

#### BIO Tags Explained

The BIO scheme expands the five standard tags (NEG, NSCO, UNC, USCO, O) into nine BIO-prefixed tags:

- **B-NEG**: Beginning of a negation marker
- **I-NEG**: Inside (continuation) of a negation marker
- **B-NSCO**: Beginning of a negation scope
- **I-NSCO**: Inside of a negation scope
- **B-UNC**: Beginning of an uncertainty marker
- **I-UNC**: Inside of an uncertainty marker
- **B-USCO**: Beginning of an uncertainty scope
- **I-USCO**: Inside of an uncertainty scope
- **O**: Outside any entity (unchanged)

#### Importance of BIO Tagging

BIO tagging provides several advantages over standard labeling:

1. **Entity Boundary Precision**: Explicit marking of entity beginnings and continuations
2. **Multi-token Entity Handling**: Better handling of multi-token entities like "no parece" (does not seem)
3. **Scope Delineation**: More accurate detection of scope boundaries
4. **Transition Modeling**: Enables the HMM to learn meaningful transition patterns (e.g., B-NEG → I-NEG is highly likely, while B-NEG → B-NEG is unlikely)

#### BIO Scheme Implementation

The BIO tagging is implemented in two places:

1. **During preprocessing**: When mapping annotations to tokens

   ```python
   if use_bio:
       # Define state space with BIO tags
       state_space = {"O", "B-NEG", "I-NEG", "B-NSCO", "I-NSCO", "B-UNC", "I-UNC", "B-USCO", "I-USCO"}
   else:
       # Standard state space
       state_space = {"O", "NEG", "NSCO", "UNC", "USCO"}
   ```

2. **In the `get_token_label` function**: When determining if a token is at the beginning or inside an entity

   ```python
   # For BIO tagging
   if is_beginning:
       return f"B-{best_label}"
   else:
       # Token is inside an entity
       return f"I-{best_label}"
   ```

The successful implementation of BIO tagging has a significant impact on the models' ability to identify entity boundaries correctly, especially for longer entities and scopes.

### Observation Sequence Creation

The preprocessing pipeline creates different observation formats for each model variant:

#### 1. Baseline Model Observations

For the baseline model, observations are simply the cleaned token text:

```python
# Simple word-only observations for baseline
if not include_pos:
    observation = token_info["cleaned_text"]
    doc_observations.append(observation)
    doc_states.append(token_info["label"])
    vocab.add(observation)
```

Example of a baseline observation sequence:
```
["el", "paciente", "no", "presenta", "fiebre"]
```

With corresponding states:
```
["O", "O", "NEG", "NSCO", "NSCO"]
```

#### 2. POS Model Observations

For the POS-enhanced model, observations are tuples of (token, POS tag):

```python
# Word+POS tuple observations
if include_pos:
    observation = (token_info["cleaned_text"], token_info["pos"])
    doc_observations.append(observation)
    doc_states.append(token_info["label"])
    vocab.add(observation)
    pos_tags.add(token_info["pos"])
```

Example of a POS model observation sequence:
```
[("el", "DET"), ("paciente", "NOUN"), ("no", "ADV"), ("presenta", "VERB"), ("fiebre", "NOUN")]
```

With corresponding states (standard labeling):
```
["O", "O", "NEG", "NSCO", "NSCO"]
```

#### 3. BIO+POS Model Observations

For the BIO+POS model, observations are the same as for the POS model, but the states use BIO tagging:

Example observation sequence (same as POS model):
```
[("el", "DET"), ("paciente", "NOUN"), ("no", "ADV"), ("presenta", "VERB"), ("fiebre", "NOUN")]
```

With corresponding BIO-tagged states:
```
["O", "O", "B-NEG", "B-NSCO", "I-NSCO"]
```

### Vocabulary Building

Each model requires building appropriate vocabulary sets:

```python
# Set of unique observations
vocab = set()  

# Set of unique POS tags encountered
pos_tags = set()  

# In the processing loop
vocab.add(observation)
if include_pos:
    pos_tags.add(token_info["pos"])

# For BIO+POS models, extract words and POS tags separately
self.words = sorted(list({word for word, _ in vocabulary}))
self.pos_tags = sorted(list({pos for _, pos in vocabulary}))
```

The vocabulary size and composition differ significantly between models:
- The baseline model vocabulary contains only word tokens
- The POS and BIO+POS models have tuple-based vocabularies (word, POS)
- Additional mappings for words and POS tags are created for backoff strategies

The final preprocessed data structure for each model contains:

```python
{
    "sequences": sequences,        # Detailed sequence info
    "observations": observations,  # Observation sequences for HMM
    "states": states,              # State sequences for HMM
    "vocabulary": vocab,           # Unique observations
    "pos_tags": pos_tags,          # Unique POS tags (if applicable)
    "state_space": state_space,    # Possible states
    "doc_languages": doc_languages,  # Languages encountered
}
```

## Hidden Markov Models

This section provides a detailed explanation of the different Hidden Markov Model (HMM) implementations used in this project.

### Basic HMM Principles

Hidden Markov Models are probabilistic sequence models that represent a sequence of observations as being generated by transitioning through a sequence of hidden states.

#### Core HMM Components

1. **States**: The hidden states in the model (in this case, entity labels)
2. **Observations**: The visible outputs (tokens or token+POS tuples)
3. **Parameters**:
   - **Initial probabilities**: P(s₁) - probability of starting in state s
   - **Transition probabilities**: P(sₜ|sₜ₋₁) - probability of moving from one state to another
   - **Emission probabilities**: P(oₜ|sₜ) - probability of outputting observation o from state s

#### HMM Assumptions

1. **Markov property**: The current state depends only on the previous state(s)
2. **Output independence**: The current observation depends only on the current state
3. **Stationarity**: Transition and emission probabilities don't change over time

### Baseline HMM Implementation

The baseline model is a standard first-order HMM that uses only word tokens as observations.

#### State Space

The baseline model uses five states:
- `NEG`: Negation markers
- `NSCO`: Scope of negation
- `UNC`: Uncertainty markers
- `USCO`: Scope of uncertainty
- `O`: Outside (not part of any entity)

#### Model Initialization

```python
class HMMBaseline(BaseHMM):
    def __init__(self, state_space: Set[str], vocabulary: Set[str], smoothing: float = 0.01):
        """
        Initialize the HMM model with state space and vocabulary
        
        Parameters:
            state_space (Set[str]): Set of possible states
            vocabulary (Set[str]): Set of possible observations
            smoothing (float): Laplace smoothing parameter
        """
        super().__init__(state_space, vocabulary, smoothing)
```

#### Training Process

The training process involves counting co-occurrences in the training data:

```python
def train(self, observations: List[List[Any]], states: List[List[str]]) -> None:
    """
    Train the HMM by counting transitions and emissions
    """
    n_states = len(self.state_space)
    n_obs = len(self.vocabulary)
    
    # Initialize counts with smoothing
    initial_counts = np.ones(n_states) * self.smoothing
    transition_counts = np.ones((n_states, n_states)) * self.smoothing
    emission_counts = np.ones((n_states, n_obs)) * self.smoothing
    
    # Count occurrences
    for obs_seq, state_seq in zip(observations, states):
        # Count initial states
        if state_seq:
            initial_counts[self.state_to_idx[state_seq[0]]] += 1
        
        # Count transitions and emissions
        for i in range(len(state_seq)):
            state_idx = self.state_to_idx[state_seq[i]]
            
            # Emission
            if i < len(obs_seq) and obs_seq[i] in self.obs_to_idx:
                obs_idx = self.obs_to_idx[obs_seq[i]]
                emission_counts[state_idx, obs_idx] += 1
            
            # Transition (if not last)
            if i < len(state_seq) - 1:
                next_state_idx = self.state_to_idx[state_seq[i+1]]
                transition_counts[state_idx, next_state_idx] += 1
    
    # Normalize to get probabilities
    self.initial_probs = initial_counts / np.sum(initial_counts)
    self.transition_probs = transition_counts / np.sum(transition_counts, axis=1, keepdims=True)
    self.emission_probs = emission_counts / np.sum(emission_counts, axis=1, keepdims=True)
```

#### Viterbi Decoding

The Viterbi algorithm finds the most likely sequence of states given a sequence of observations:

```python
def viterbi(self, observations: List[Any]) -> List[str]:
    """
    Implement the Viterbi algorithm to find the most likely sequence of states
    """
    n_states = len(self.state_space)
    T = len(observations)
    
    # Initialize Viterbi matrix and backpointers
    V = np.zeros((T, n_states))
    backpointers = np.zeros((T, n_states), dtype=int)
    
    # Initialize first step
    for s in range(n_states):
        if observations[0] in self.obs_to_idx:
            obs_idx = self.obs_to_idx[observations[0]]
            V[0, s] = np.log(self.initial_probs[s]) + np.log(self.emission_probs[s, obs_idx])
        else:
            # Observation OOV - use smoothing
            V[0, s] = np.log(self.initial_probs[s]) + np.log(self.smoothing)
    
    # Forward pass
    for t in range(1, T):
        for s in range(n_states):
            # Find the most likely previous state
            probs = V[t-1, :] + np.log(self.transition_probs[:, s])
            backpointers[t, s] = np.argmax(probs)
            max_prob = probs[backpointers[t, s]]
            
            # Add emission probability
            if observations[t] in self.obs_to_idx:
                obs_idx = self.obs_to_idx[observations[t]]
                V[t, s] = max_prob + np.log(self.emission_probs[s, obs_idx])
            else:
                # Unknown observation - use smoothing
                V[t, s] = max_prob + np.log(self.smoothing)
    
    # Backward pass to find the best path
    best_path = np.zeros(T, dtype=int)
    best_path[T-1] = np.argmax(V[T-1, :])
    
    for t in range(T-2, -1, -1):
        best_path[t] = backpointers[t+1, best_path[t+1]]
    
    return [self.state_space[idx] for idx in best_path]
```

#### Handling Out-of-Vocabulary Words

The baseline model uses a simple Laplace smoothing strategy for out-of-vocabulary (OOV) words:

```python
# If observation not in vocabulary, use a small probability
V[t, s] = max_prob + np.log(self.smoothing)
```

#### Limitations of the Baseline Model

1. **Entity Boundary Detection**: No explicit representation of entity beginnings and continuations
2. **Linguistic Information**: No use of POS tags or other linguistic features
3. **Contextual Information**: Limited to first-order dependencies

### BIO+POS Enhanced HMM

The BIO+POS model extends the baseline with two key enhancements:

1. **BIO Tagging**: Begin-Inside-Outside scheme for better entity boundary detection
2. **POS Information**: Part-of-Speech tags to capture linguistic patterns

#### Enhanced State Space

The BIO+POS model uses nine states:
- `B-NEG`, `I-NEG`: Beginning and continuation of negation markers
- `B-NSCO`, `I-NSCO`: Beginning and continuation of negation scopes
- `B-UNC`, `I-UNC`: Beginning and continuation of uncertainty markers
- `B-USCO`, `I-USCO`: Beginning and continuation of uncertainty scopes
- `O`: Outside (not part of any entity)

#### Handling Tuple Observations

The model uses (word, POS) tuples as observations:

```python
class HMMBIOPOS(BaseHMM):
    def __init__(self, state_space: Set[str], vocabulary: Set[Tuple[str, str]], smoothing: float = 0.01):
        """
        Initialize the HMM model with state space and vocabulary
        
        Parameters:
            state_space (Set[str]): Set of possible states (with BIO prefixes)
            vocabulary (Set[Tuple[str, str]]): Set of possible observations (word, POS)
            smoothing (float): Laplace smoothing parameter
        """
        super().__init__(state_space, vocabulary, smoothing)
        
        # Extract words and POS tags separately for backoff
        self.words = sorted(list({word for word, _ in vocabulary}))
        self.pos_tags = sorted(list({pos for _, pos in vocabulary}))
        
        self.word_to_idx = {word: idx for idx, word in enumerate(self.words)}
        self.pos_to_idx = {pos: idx for idx, pos in enumerate(self.pos_tags)}
        
        # Backoff parameters
        self.word_emission_probs = None
        self.pos_emission_probs = None
```

#### Backoff Strategy for Unknown Words/POS

A sophisticated backoff strategy handles unseen (word, POS) combinations:

```python
def get_emission_prob(self, state_idx: int, obs: Tuple[str, str]) -> float:
    """
    Get emission probability with backoff for unknown observations
    """
    word, pos = obs
    
    # If full observation exists in vocabulary
    if obs in self.obs_to_idx:
        obs_idx = self.obs_to_idx[obs]
        return self.emission_probs[state_idx, obs_idx]
    
    # Backoff strategy - Combine word and POS probabilities
    word_prob = self.smoothing
    pos_prob = self.smoothing
    
    if word in self.word_to_idx:
        word_idx = self.word_to_idx[word]
        word_prob = self.word_emission_probs[state_idx, word_idx]
    
    if pos in self.pos_to_idx:
        pos_idx = self.pos_to_idx[pos]
        pos_prob = self.pos_emission_probs[state_idx, pos_idx]
    
    # Geometric mean of word and POS probabilities (with more weight to word)
    return (word_prob ** 0.7) * (pos_prob ** 0.3)
```

This approach:
1. First checks if the exact (word, POS) pair is known
2. If not, backs off to separate word and POS probabilities
3. Combines them with a weighted geometric mean
4. Falls back to smoothing for completely unknown words/POS tags

#### BIO-Specific Transition Constraints

The BIO+POS model enforces additional constraints on BIO tag transitions:

```python
# Additional BIO-specific transition constraints
for i, state in enumerate(self.state_space):
    if state.startswith("B-"):
        entity_type = state[2:]  # Extract entity type (e.g., "NEG", "NSCO")
        
        # Increase probability of B-X -> I-X transitions
        i_state = f"I-{entity_type}"
        if i_state in self.state_to_idx:
            i_idx = self.state_to_idx[i_state]
            # Increase P(I-X | B-X) without normalization
            # We don't normalize to maintain the overall distribution
            self.transition_probs[i, i_idx] *= 5  # Add stronger bias because of BIO pairings
            
            # Normalize the row to ensure it sums to 1
            self.transition_probs[i, :] /= np.sum(self.transition_probs[i, :])
```

This enforces linguistically valid sequences, such as:
- B-NEG → I-NEG (high probability)
- I-NEG → I-NEG (high probability)
- B-NEG → B-NSCO (lower probability)

#### Modified Viterbi Algorithm

The BIO+POS model uses a modified Viterbi algorithm that incorporates the backoff strategy:

```python
def viterbi(self, observations: List[Tuple[str, str]]) -> List[str]:
    """
    Implement the Viterbi algorithm with backoff
    """
    # Similar structure to baseline, but using get_emission_prob
    # for emission probabilities with backoff
    emission_prob = self.get_emission_prob(s, observations[t])
    V[t, s] = max_prob + np.log(emission_prob)
```

### Second-Order HMM with BIO+POS

The second-order HMM extends the BIO+POS model by capturing trigram state dependencies (i.e., each state depends on the two previous states rather than just one).

#### Capturing Longer Dependencies

The key difference in the second-order model is in the transition probabilities:
- First-order: P(sₜ|sₜ₋₁) - current state depends on one previous state
- Second-order: P(sₜ|sₜ₋₁,sₜ₋₂) - current state depends on two previous states

This enables the model to capture longer-range dependencies, which is particularly useful for entity scopes that span multiple tokens.

#### Trigram Transition Parameters

The second-order model uses a 3D transition probability matrix:

```python
class HMMSecondOrder(HMMBIOPOS):
    def __init__(self, state_space: Set[str], vocabulary: Set[Tuple[str, str]], smoothing: float = 0.01):
        """
        Initialize the second-order HMM model
        """
        super().__init__(state_space, vocabulary, smoothing)
        
        # Add artificial start state for initialization
        self.START = "<START>"
        self.extended_state_space = [self.START] + self.state_space
        
        # Extended mapping for the artificial start state
        self.extended_state_to_idx = {state: idx for idx, state in enumerate(self.extended_state_space)}
        
        # HMM parameters for second-order model
        self.initial_bigram_probs = None  # P(s_2 | s_1)
        self.transition_probs = None  # P(s_t | s_{t-1}, s_{t-2}) - will be 3D
```

#### Counting Trigram Transitions

Training the second-order model involves counting trigram transitions:

```python
# For second-order transitions P(s_t | s_{t-1}, s_{t-2})
transition_counts = np.ones((n_states, n_states, n_states)) * self.smoothing

# Trigram transitions and remaining emissions
for i in range(2, len(state_seq)):
    try:
        # Get indices of the trigram
        idx_t_2 = self.state_to_idx[state_seq[i-2]]
        idx_t_1 = self.state_to_idx[state_seq[i-1]]
        idx_t = self.state_to_idx[state_seq[i]]
        
        # Increment trigram count
        transition_counts[idx_t_2, idx_t_1, idx_t] += 1
        
        # Emission for current state
        # ... (similar to first-order model)
    except KeyError:
        continue
```

#### Modified Viterbi Algorithm for Second-Order Models

The Viterbi algorithm for second-order HMMs is significantly more complex:

```python
def viterbi(self, observations: List[Tuple[str, str]]) -> List[str]:
    """
    Modified Viterbi for second-order HMM decoding
    """
    if not observations:
        return []
    
    n_states = len(self.state_space)
    T = len(observations)
    
    if T == 1:
        # Special case for single observation
        # ...
    
    # Initialize DP table for Viterbi - now 3D
    V = np.zeros((T, n_states, n_states))  # MAX probability of being in states s1, s2 at times t-1, t
    
    # Initialize backpointers
    bp = np.zeros((T, n_states, n_states), dtype=int)  # Previous state that led to s1, s2
    
    # Base case: t = 1
    for s0 in range(n_states):
        for s1 in range(n_states):
            prob_s0 = self.initial_probs[s0]
            prob_s1_given_s0 = self.initial_bigram_probs[s0, s1]
            
            # Emissions with backoff
            emission_prob_s0 = self.get_emission_prob(s0, observations[0])
            emission_prob_s1 = self.get_emission_prob(s1, observations[1])
            
            # Combined probability
            V[1, s0, s1] = (np.log(prob_s0) + np.log(prob_s1_given_s0) + 
                           np.log(emission_prob_s0) + np.log(emission_prob_s1))
    
    # Forward pass
    for t in range(2, T):
        for s1 in range(n_states):
            for s2 in range(n_states):
                max_prob = float('-inf')
                max_s0 = 0
                
                # Find the previous state s0 that maximizes the probability
                for s0 in range(n_states):
                    prev_prob = V[t-1, s0, s1]
                    trans_prob = self.transition_probs[s0, s1, s2]
                    
                    prob = prev_prob + np.log(trans_prob)
                    
                    if prob > max_prob:
                        max_prob = prob
                        max_s0 = s0
                
                # Emission for s2 with backoff
                emission_prob = self.get_emission_prob(s2, observations[t])
                
                # Update DP table and backpointer
                V[t, s1, s2] = max_prob + np.log(emission_prob)
                bp[t, s1, s2] = max_s0
    
    # Backward pass to find the best path
    path = np.zeros(T, dtype=int)
    
    # Find the best pair of final states
    max_prob = float('-inf')
    for s1 in range(n_states):
        for s2 in range(n_states):
            if V[T-1, s1, s2] > max_prob:
                max_prob = V[T-1, s1, s2]
                path[T-2] = s1
                path[T-1] = s2
    
    # Trace back the best path
    for t in range(T-3, -1, -1):
        path[t] = bp[t+2, path[t+1], path[t+2]]
    
    # Convert indices back to states
    return [self.state_space[idx] for idx in path]
```

Key differences from the first-order Viterbi:
1. The DP table is 3D instead of 2D: `V[t, s1, s2]` represents the probability of the most likely path ending with states s1 and s2 at times t-1 and t
2. The backpointers now point to the best state two steps back
3. The backward pass is more complex, as it needs to find the best pair of final states

#### Enhanced BIO Transition Constraints

The second-order model applies even more sophisticated BIO transition constraints:

```python
# Additional BIO-specific transition constraints
for i, state1 in enumerate(self.state_space):
    if state1.startswith("B-"):
        entity_type = state1[2:]  # Extract entity type
        i_state = f"I-{entity_type}"
        
        if i_state in self.state_to_idx:
            i_idx = self.state_to_idx[i_state]
            for j, state2 in enumerate(self.state_space):
                if state2 == i_state:
                    # Boost B-X → I-X → I-X transitions
                    transition_counts[i, i_idx, i_idx] += 5  # Add stronger bias
```

This enforces linguistically valid trigram sequences, such as:
- B-NEG → I-NEG → I-NEG (high probability)
- I-NSCO → I-NSCO → I-NSCO (high probability)
- B-NEG → I-NEG → O (medium probability)

## Evaluation Metrics

This section details the comprehensive evaluation framework used to assess model performance, with metrics at multiple levels of analysis.

### Token-Level Evaluation

Token-level evaluation assesses the models' performance on a per-token basis, without considering entity boundaries.

#### Token-Level Metrics

The primary token-level metrics are precision, recall, and F1-score:

```python
def compute_metrics(true_labels: List[List[str]], pred_labels: List[List[str]], is_bio: bool = False) -> Dict:
    """
    Compute precision, recall, and F1-score for the HMM model predictions
    """
    # Flatten the lists of labels
    y_true = []
    y_pred = []
    
    for true_seq, pred_seq in zip(true_labels, pred_labels):
        y_true.extend(true_seq)
        y_pred.extend(pred_seq)
    
    if is_bio:
        labels = ["B-NEG", "I-NEG", "B-NSCO", "I-NSCO", "B-UNC", "I-UNC", "B-USCO", "I-USCO", "O"]
    else:
        labels = ["NEG", "NSCO", "UNC", "USCO", "O"]
    
    present_labels = list(set(y_true) | set(y_pred))
    eval_labels = [label for label in labels if label in present_labels]
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=eval_labels, average=None, zero_division=0
    )
    
    # Compute macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=eval_labels, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=eval_labels, average="weighted", zero_division=0
    )
    
    # Structure results
    metrics = {
        "class_metrics": {
            label: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for label, p, r, f, s in zip(eval_labels, precision, recall, f1, support)
        },
        "macro_avg": {"precision": float(macro_precision), "recall": float(macro_recall), "f1": float(macro_f1)},
        "weighted_avg": {"precision": float(weighted_precision), "recall": float(weighted_recall), "f1": float(weighted_f1)},
    }
    
    return metrics
```

#### Token-Level Evaluation for BIO vs. Standard Labels

Token-level evaluation is handled differently depending on whether BIO tagging is used:

1. **Standard Labeling**: Each token is evaluated for its label (NEG, NSCO, UNC, USCO, O)
2. **BIO Labeling**: Each token is evaluated for its BIO tag and entity type (B-NEG, I-NEG, etc.)

This distinction is important because:
- In BIO tagging, the distinction between B-X and I-X tags is as important as the entity type
- A model might correctly identify an entity type but incorrectly predict B- vs. I- tags
- Token-level metrics for BIO models tend to be lower due to the increased granularity

#### Classification Report

A human-readable classification report is generated for token-level evaluation:

```python
def print_classification_report(true_labels: List[List[str]], pred_labels: List[List[str]], is_bio: bool = False) -> None:
    """
    Classification report for the HMM model predictions
    """
    y_true = []
    y_pred = []
    
    for true_seq, pred_seq in zip(true_labels, pred_labels):
        y_true.extend(true_seq)
        y_pred.extend(pred_seq)
    
    if is_bio:
        labels = ["B-NEG", "I-NEG", "B-NSCO", "I-NSCO", "B-UNC", "I-UNC", "B-USCO", "I-USCO", "O"]
        present_labels = list(set(y_true) | set(y_pred))
        labels = [label for label in labels if label in present_labels]
    else:
        labels = ["NEG", "NSCO", "UNC", "USCO", "O"]
    
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
```

Sample classification report for a BIO model:
```
              precision    recall  f1-score   support

       B-NEG       0.89      0.92      0.90       342
       I-NEG       0.76      0.70      0.73        98
      B-NSCO       0.83      0.78      0.80       512
      I-NSCO       0.79      0.77      0.78      1423
       B-UNC       0.87      0.83      0.85       271
       I-UNC       0.71      0.68      0.69        86
      B-USCO       0.80      0.76      0.78       389
      I-USCO       0.77      0.75      0.76      1014
           O       0.96      0.97      0.97     18421

    accuracy                           0.94     22556
   macro avg       0.82      0.80      0.81     22556
weighted avg       0.94      0.94      0.94     22556
```

### Entity-Level Evaluation

Entity-level evaluation assesses the models' performance in identifying complete entities (spans), which is more relevant for practical applications.

#### Entity Extraction

An entity is defined as a contiguous sequence of tokens with the same label (for standard tagging) or a B-X followed by zero or more I-X tags (for BIO tagging):

```python
def extract_entities(labels: List[str]) -> List[Tuple[str, int, int]]:
    """
    Extract entities from token-level labels
    """
    entities = []
    
    if is_bio:
        entities = convert_bio_to_entity_spans(labels, [None] * len(labels))
    else:
        current_entity = None
        start_idx = 0
        
        for i, label in enumerate(labels):
            if label == "O":
                # End current entity
                if current_entity:
                    entities.append((current_entity, start_idx, i - 1))
                    current_entity = None
            else:
                if current_entity != label:
                    # Start a new entity if label changes
                    if current_entity:
                        entities.append((current_entity, start_idx, i - 1))
                    current_entity = label
                    start_idx = i
        
        if current_entity:
            # Add the last entity
            entities.append((current_entity, start_idx, len(labels) - 1))
    
    return entities
```

For BIO tagging, a specialized function converts BIO tags to entity spans:

```python
def convert_bio_to_entity_spans(bio_labels: List[str], tokens: List[Any]) -> List[Tuple[str, int, int]]:
    """
    Convert BIO labels to entity spans
    """
    entities = []
    current_entity = None
    start_idx = -1
    
    for i, label in enumerate(bio_labels):
        if label.startswith("B-"):
            # End any current entity
            if current_entity is not None:
                entities.append((current_entity, start_idx, i - 1))
                
            # Start a new entity
            current_entity = label[2:]
            start_idx = i
            
        elif label.startswith("I-"):
            entity_type = label[2:]
            
            # Continue current entity if types match, otherwise start a new entity
            if current_entity != entity_type or start_idx == -1:
                if current_entity is not None:
                    entities.append((current_entity, start_idx, i - 1))
                current_entity = entity_type
                start_idx = i
                
        elif label == "O":
            # End any current entity
            if current_entity is not None:
                entities.append((current_entity, start_idx, i - 1))
                current_entity = None
                start_idx = -1
    
    # Add the last entity if there is one
    if current_entity is not None:
        entities.append((current_entity, start_idx, len(bio_labels) - 1))
    
    return entities
```

#### Entity-Level Metrics Calculation

Entity-level metrics are calculated by comparing the extracted entity spans:

```python
def get_entity_based_metrics(true_labels: List[List[str]], pred_labels: List[List[str]], is_bio: bool = False) -> Dict:
    """
    Compute entity-based metrics (treating each contiguous chunk as one entity)
    """
    true_entities = []
    pred_entities = []
    
    for true_seq, pred_seq in zip(true_labels, pred_labels):
        true_entities.extend(extract_entities(true_seq))
        pred_entities.extend(extract_entities(pred_seq))
    
    entity_types = ["NEG", "NSCO", "UNC", "USCO"]
    
    correct_by_type = {entity_type: 0 for entity_type in entity_types}
    pred_by_type = {entity_type: 0 for entity_type in entity_types}
    true_by_type = {entity_type: 0 for entity_type in entity_types}
    
    # Count predicted entities
    for entity_type, start, end in pred_entities:
        if entity_type in pred_by_type:
            pred_by_type[entity_type] += 1
    
    # Count true entities and exact matches
    for entity_type, start, end in true_entities:
        if entity_type in true_by_type:
            true_by_type[entity_type] += 1
            if (entity_type, start, end) in pred_entities:
                correct_by_type[entity_type] += 1
    
    # Calculate metrics for each entity type
    entity_metrics = {}
    
    for entity_type in entity_types:
        precision = correct_by_type[entity_type] / max(1, pred_by_type[entity_type])
        recall = correct_by_type[entity_type] / max(1, true_by_type[entity_type])
        f1 = 2 * precision * recall / max(1e-10, precision + recall)
        
        entity_metrics[entity_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": true_by_type[entity_type],
        }
    
    # Calculate macro-averaged metrics
    macro_precision = sum(metrics["precision"] for metrics in entity_metrics.values()) / len(entity_types)
    macro_recall = sum(metrics["recall"] for metrics in entity_metrics.values()) / len(entity_types)
    macro_f1 = sum(metrics["f1"] for metrics in entity_metrics.values()) / len(entity_types)
    
    entity_metrics["macro_avg"] = {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1}
    
    return entity_metrics
```

An entity is considered correctly predicted only if:
1. The entity type matches (e.g., NEG)
2. The entity boundaries (start and end positions) match exactly

This is a stricter evaluation than token-level metrics and better reflects the models' practical utility.

### Scope Detection Evaluation

Scope detection evaluation specifically focuses on how well models identify the scope of negation (NSCO) and uncertainty (USCO), which is particularly important for medical text analysis.

#### Scope Detection Metrics

Unlike entity-level evaluation, scope detection metrics include partial matches:

```python
def evaluate_scope_detection(true_labels: List[List[str]], pred_labels: List[List[str]], observations: List[List[Any]], is_bio: bool = False) -> Dict:
    """
    Evaluate negation/uncertainty scope detection more comprehensively
    """
    def calculate_overlap(true_scope: Tuple[str, int, int], pred_scope: Tuple[str, int, int]) -> float:
        """Calculate Jaccard similarity between two scopes"""
        true_type, true_start, true_end = true_scope
        pred_type, pred_start, pred_end = pred_scope
        
        if true_type != pred_type:
            return 0.0
        
        overlap_start = max(true_start, pred_start)
        overlap_end = min(true_end, pred_end)
        overlap_length = max(0, overlap_end - overlap_start + 1)
        
        true_length = true_end - true_start + 1
        pred_length = pred_end - pred_start + 1
        
        # Calculate Jaccard similarity (intersection / union)
        union_length = true_length + pred_length - overlap_length
        return overlap_length / union_length if union_length > 0 else 0.0
    
    scope_metrics = {
        "NEG": {"exact": 0, "partial": 0, "missed": 0, "false_positive": 0},
        "NSCO": {"exact": 0, "partial": 0, "missed": 0, "false_positive": 0},
        "UNC": {"exact": 0, "partial": 0, "missed": 0, "false_positive": 0},
        "USCO": {"exact": 0, "partial": 0, "missed": 0, "false_positive": 0},
    }
    
    # Process each sequence
    for seq_idx, (true_seq, pred_seq) in enumerate(zip(true_labels, pred_labels)):
        true_scopes = extract_scopes(true_seq, is_bio)
        pred_scopes = extract_scopes(pred_seq, is_bio)
        
        matched_pred_scopes = set()  # Track matched predicted scopes
        
        for true_scope in true_scopes:
            true_type = true_scope[0]
            best_match = None
            best_overlap = 0.0
            
            for i, pred_scope in enumerate(pred_scopes):
                if i in matched_pred_scopes:
                    continue
                
                overlap = calculate_overlap(true_scope, pred_scope)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = i
            
            if best_match is not None and best_overlap >= 0.7:
                # Exact match (high overlap)
                scope_metrics[true_type]["exact"] += 1
                matched_pred_scopes.add(best_match)
            elif best_match is not None and best_overlap > 0:
                # Partial match
                scope_metrics[true_type]["partial"] += 1
                matched_pred_scopes.add(best_match)
            else:
                # Missed scope
                scope_metrics[true_type]["missed"] += 1
        
        # Count false positives
        for i, pred_scope in enumerate(pred_scopes):
            if i not in matched_pred_scopes:
                pred_type = pred_scope[0]
                scope_metrics[pred_type]["false_positive"] += 1
    
    # Calculate precision, recall, and F1 for each scope type
    result = {}
    
    for scope_type, metrics in scope_metrics.items():
        # Count partial matches as half
        true_positives = metrics["exact"] + metrics["partial"] * 0.5
        precision = true_positives / max(1, true_positives + metrics["false_positive"])
        recall = true_positives / max(1, true_positives + metrics["missed"])
        f1 = 2 * precision * recall / max(1e-10, precision + recall)
        
        result[scope_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_matches": metrics["exact"],
            "partial_matches": metrics["partial"],
            "missed": metrics["missed"],
            "false_positives": metrics["false_positive"],
        }
    
    # Calculate macro-averaged metrics
    macro_precision = sum(m["precision"] for m in result.values()) / len(result)
    macro_recall = sum(m["recall"] for m in result.values()) / len(result)
    macro_f1 = sum(m["f1"] for m in result.values()) / len(result)
    
    result["macro_avg"] = {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1}
    
    return result
```

Key features of scope detection evaluation:

1. **Jaccard Similarity**: Used to quantify the overlap between predicted and true scopes
2. **Exact vs. Partial Matches**: 
   - Exact match: Jaccard similarity ≥ 0.7
   - Partial match: 0 < Jaccard similarity < 0.7
3. **Partial Credit**: Partial matches count as 0.5 in precision and recall calculations
4. **Scope Types**: Separate metrics for negation markers (NEG), negation scopes (NSCO), uncertainty markers (UNC), and uncertainty scopes (USCO)

This evaluation approach is more lenient than strict entity-level evaluation but still rewards precise scope boundary detection.

### Language-Specific Analysis

Since the dataset contains both Spanish and Catalan text, language-specific analysis helps identify potential performance differences between languages.

```python
def analyze_by_language(true_labels: List[List[str]], pred_labels: List[List[str]], token_languages: List[List[str]], is_bio: bool = False) -> Dict:
    """
    Analyze performance by language
    """
    # Separate by language
    es_true = []  # Spanish true labels
    es_pred = []
    ca_true = []  # Catalan true labels
    ca_pred = []
    
    for true_seq, pred_seq, lang_seq in zip(true_labels, pred_labels, token_languages):
        for true, pred, lang in zip(true_seq, pred_seq, lang_seq):
            if lang == "es":
                es_true.append(true)
                es_pred.append(pred)
            else:  # Catalan
                ca_true.append(true)
                ca_pred.append(pred)
    
    # Calculate metrics for each language
    es_metrics = compute_metrics([es_true], [es_pred], is_bio)
    ca_metrics = compute_metrics([ca_true], [ca_pred], is_bio)
    
    return {"spanish": es_metrics, "catalan": ca_metrics}
```

This analysis helps answer important questions like:
- Does the model perform better in one language than the other?
- Are there language-specific error patterns?
- Do certain entity types have different recognition rates across languages?

The language information is derived from the preprocessing step, where language detection is performed at the sentence level.

## Confusion Matrix Analysis

Confusion matrices provide detailed insights into model performance, showing exactly which classes are being confused with each other.

### Understanding Confusion Matrices

A confusion matrix is a table showing the counts of true vs. predicted labels. For each class pair, it shows how many instances of the true class were predicted as another class.

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Flatten true states and predicted states
flat_true_states = [tag for seq in test_data["states"] for tag in seq]
flat_pred_states = [tag for seq in test_predictions for tag in seq]

# Get unique labels (sorted to ensure consistency)
labels = sorted(test_data["state_space"])

# Compute confusion matrix
cm = confusion_matrix(flat_true_states, flat_pred_states, labels=labels)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - BIO+POS Model')
plt.tight_layout()
plt.savefig(os.path.join(PATH_ROOT, "data", "results", "evaluation", "confusion_matrix_bio_pos.png"))
plt.show()
```

#### Normalizing Confusion Matrices

Normalized confusion matrices show proportions rather than raw counts, making it easier to identify patterns:

```python
# Print normalized confusion matrix (percentages)
plt.figure(figsize=(12, 10))
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix - BIO+POS Model')
plt.tight_layout()
plt.savefig(os.path.join(PATH_ROOT, "data", "results", "evaluation", "confusion_matrix_bio_pos_norm.png"))
plt.show()
```

#### Extracting Insights from Confusion Matrices

Additional analysis can extract specific insights from confusion matrices:

```python
# Add row and column totals for better analysis
row_sums = cm.sum(axis=1)
col_sums = cm.sum(axis=0)

print("\nRow Totals (True counts):")
for i, label in enumerate(labels):
    print(f"{label}: {row_sums[i]}")
    
print("\nColumn Totals (Predicted counts):")
for i, label in enumerate(labels):
    print(f"{label}: {col_sums[i]}")

# Identify most common misclassifications
misclassifications = []
for i in range(len(labels)):
    for j in range(len(labels)):
        if i != j and cm[i, j] > 0:  # Skip correct classifications and zero values
            misclassifications.append((labels[i], labels[j], cm[i, j]))

# Sort by count (descending)
misclassifications.sort(key=lambda x: x[2], reverse=True)

print("\nTop 10 Most Common Misclassifications:")
for true_label, pred_label, count in misclassifications[:10]:
    print(f"True: {true_label}, Predicted: {pred_label}, Count: {count}")
```

### Analyzing Tag Transitions

For BIO tagging models, analyzing transitions between tags provides valuable insights:

```python
# Analyze BIO tag transitions
print("\nBIO Tag Transition Analysis:")
bio_transitions = {
    "B→I correct": 0,
    "B→I incorrect": 0,
    "I→I correct": 0,
    "I→I incorrect": 0,
    "O→B correct": 0,
    "O→B incorrect": 0
}

for i in range(len(test_data["states"])):
    true_seq = test_data["states"][i]
    pred_seq = test_predictions[i]
    
    for j in range(1, len(true_seq)):  # Start from second token
        if true_seq[j-1].startswith('B-') and true_seq[j].startswith('I-'):
            if pred_seq[j].startswith('I-'):
                bio_transitions["B→I correct"] += 1
            else:
                bio_transitions["B→I incorrect"] += 1
                
        if true_seq[j-1].startswith('I-') and true_seq[j].startswith('I-'):
            if pred_seq[j].startswith('I-'):
                bio_transitions["I→I correct"] += 1
            else:
                bio_transitions["I→I incorrect"] += 1
                
        if true_seq[j-1] == 'O' and true_seq[j].startswith('B-'):
            if pred_seq[j].startswith('B-'):
                bio_transitions["O→B correct"] += 1
            else:
                bio_transitions["O→B incorrect"] += 1

for transition, count in bio_transitions.items():
    print(f"{transition}: {count}")
```

This analysis reveals how well the model handles specific transitions, which is particularly relevant for the second-order HMM.

### Common Error Patterns

For the second-order model, additional context-dependent analysis is performed:

```python
# Function to identify specific tag sequences in data
def count_tag_sequences(states, predictions):
    sequence_counts = {
        "I-X following B-X correct": 0,
        "I-X following B-X incorrect": 0,
        "I-X following I-X correct": 0,
        "I-X following I-X incorrect": 0,
        "B-X following O correct": 0,
        "B-X following O incorrect": 0
    }
    
    for i in range(len(states)):
        true_seq = states[i]
        pred_seq = predictions[i]
        
        for j in range(1, len(true_seq)):  # Start from second token
            # Check I-X following B-X
            if true_seq[j-1].startswith('B-') and true_seq[j].startswith('I-') and true_seq[j-1][2:] == true_seq[j][2:]:
                if pred_seq[j].startswith('I-') and pred_seq[j][2:] == true_seq[j][2:]:
                    sequence_counts["I-X following B-X correct"] += 1
                else:
                    sequence_counts["I-X following B-X incorrect"] += 1
                    
            # Check I-X following I-X
            if true_seq[j-1].startswith('I-') and true_seq[j].startswith('I-') and true_seq[j-1][2:] == true_seq[j][2:]:
                if pred_seq[j].startswith('I-') and pred_seq[j][2:] == true_seq[j][2:]:
                    sequence_counts["I-X following I-X correct"] += 1
                else:
                    sequence_counts["I-X following I-X incorrect"] += 1
                    
            # Check B-X following O
            if true_seq[j-1] == 'O' and true_seq[j].startswith('B-'):
                if pred_seq[j].startswith('B-') and pred_seq[j][2:] == true_seq[j][2:]:
                    sequence_counts["B-X following O correct"] += 1
                else:
                    sequence_counts["B-X following O incorrect"] += 1
                    
    return sequence_counts
```

These analyses help identify specific error patterns:

1. **Boundary Detection Errors**: Incorrect B- vs. I- predictions
2. **Entity Type Errors**: Correct boundary but wrong entity type
3. **Scope Coverage Errors**: Missing parts of the scope span
4. **False Positives**: Predicting entities where none exist
5. **False Negatives**: Missing entities entirely

## Model Comparison

This section compares the performance of the three models across multiple evaluation metrics.

### Baseline vs. BIO+POS

The first comparison is between the baseline HMM and the BIO+POS enhanced model:

```python
# Load baseline metrics
with open(os.path.join(PATH_ROOT, "data", "results", "evaluation", "evaluation_baseline.json"), 'r') as f:
    baseline_metrics = json.load(f)
    
baseline_token_f1 = baseline_metrics["token_metrics"]["macro_avg"]["f1"]
baseline_entity_f1 = baseline_metrics["entity_metrics"]["macro_avg"]["f1"]

bio_pos_token_f1 = metrics["macro_avg"]["f1"]
bio_pos_entity_f1 = entity_metrics["macro_avg"]["f1"]

# Plot comparison
labels = ['Token-Level F1', 'Entity-Level F1']
baseline_scores = [baseline_token_f1, baseline_entity_f1]
bio_pos_scores = [bio_pos_token_f1, bio_pos_entity_f1]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, baseline_scores, width, label='Baseline HMM')
ax.bar(x + width/2, bio_pos_scores, width, label='BIO+POS HMM')

ax.set_ylabel('F1 Score')
ax.set_title('Performance Comparison: Baseline vs. BIO+POS Enhanced HMM')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add values on top of bars
for i, v in enumerate(baseline_scores):
    ax.text(i - width/2, v + 0.01, f'{v:.4f}', ha='center')
for i, v in enumerate(bio_pos_scores):
    ax.text(i + width/2, v + 0.01, f'{v:.4f}', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(PATH_ROOT, "data", "results", "evaluation", "bio_pos_vs_baseline.png"))
plt.show()

# Calculate improvement
token_improvement = (bio_pos_token_f1 - baseline_token_f1) / baseline_token_f1 * 100
entity_improvement = (bio_pos_entity_f1 - baseline_entity_f1) / baseline_entity_f1 * 100

print(f"Token-level F1 improvement: {token_improvement:.2f}%")
print(f"Entity-level F1 improvement: {entity_improvement:.2f}%")
```

The key advantages of the BIO+POS model over the baseline include:

1. **Improved Entity Boundary Detection**: BIO tagging provides explicit marking of entity beginnings and continuations
2. **Better Handling of Multi-Token Entities**: The BIO scheme better handles entities that span multiple tokens
3. **Linguistic Information**: POS tags capture grammatical patterns relevant to negation and uncertainty
4. **Robust OOV Handling**: The backoff strategy for unknown words reduces the impact of out-of-vocabulary terms

### BIO+POS vs. Second-Order

The second comparison is between the first-order BIO+POS model and the second-order BIO+POS model:

```python
# Load first-order BIO+POS metrics
with open(os.path.join(PATH_ROOT, "data", "results", "evaluation", "evaluation_bio_pos.json"), 'r') as f:
    bio_pos_metrics = json.load(f)
    
# Extract macro F1 scores
bio_pos_token_f1 = bio_pos_metrics["token_metrics"]["macro_avg"]["f1"]
bio_pos_entity_f1 = bio_pos_metrics["entity_metrics"]["macro_avg"]["f1"]
bio_pos_scope_f1 = bio_pos_metrics["scope_metrics"]["macro_avg"]["f1"]

second_order_token_f1 = metrics["macro_avg"]["f1"]
second_order_entity_f1 = entity_metrics["macro_avg"]["f1"]
second_order_scope_f1 = scope_metrics["macro_avg"]["f1"]

# Plot comparison
labels = ["Token-Level F1", "Entity-Level F1", "Scope-Level F1"]
bio_pos_scores = [bio_pos_token_f1, bio_pos_entity_f1, bio_pos_scope_f1]
second_order_scores = [second_order_token_f1, second_order_entity_f1, second_order_scope_f1]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, bio_pos_scores, width, label='First-Order BIO+POS HMM')
ax.bar(x + width/2, second_order_scores, width, label='Second-Order BIO+POS HMM')

ax.set_ylabel('F1 Score')
ax.set_title('Performance Comparison: First-Order vs. Second-Order BIO+POS HMM')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add values on top of bars
for i, v in enumerate(bio_pos_scores):
    ax.text(i - width/2, v + 0.01, f'{v:.4f}', ha='center')
for i, v in enumerate(second_order_scores):
    ax.text(i + width/2, v + 0.01, f'{v:.4f}', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(PATH_ROOT, 'data', 'results', 'evaluation', 'first_vs_second_order.png'))
plt.show()

# Calculate improvement
second_vs_first_token = (second_order_token_f1 - bio_pos_token_f1) / bio_pos_token_f1 * 100
second_vs_first_entity = (second_order_entity_f1 - bio_pos_entity_f1) / bio_pos_entity_f1 * 100
second_vs_first_scope = (second_order_scope_f1 - bio_pos_scope_f1) / bio_pos_scope_f1 * 100

print(f"Second-Order vs First-Order (Token-level F1): {second_vs_first_token:.2f}% improvement")
print(f"Second-Order vs First-Order (Entity-level F1): {second_vs_first_entity:.2f}% improvement")
print(f"Second-Order vs First-Order (Scope-level F1): {second_vs_first_scope:.2f}% improvement")
```

The key advantages of the second-order model over the first-order BIO+POS model include:

1. **Longer-Range Dependencies**: Better modeling of dependencies between non-adjacent states
2. **Improved Tag Sequence Learning**: More accurate prediction of BIO tag sequences (B→I→I vs. B→I→O)
3. **Better Scope Detection**: More accurate identification of scope boundaries
4. **Context-Sensitive Decisions**: Decisions based on richer context

### Performance on Entity Types

A detailed comparison of performance on specific entity types:

```python
print("Entity-level F1 comparison by entity type:")
print("Entity Type\tFirst-Order\tSecond-Order\tImprovement")
print("-" * 60)

for entity_type in ["NEG", "NSCO", "UNC", "USCO"]:
    first_f1 = bio_pos_metrics["entity_metrics"][entity_type]["f1"]
    second_f1 = entity_metrics[entity_type]["f1"]
    improvement = (second_f1 - first_f1) / first_f1 * 100
    
    print(f"{entity_type}\t\t{first_f1:.4f}\t\t{second_f1:.4f}\t\t{improvement:+.2f}%")
```

This analysis shows how performance varies across entity types, highlighting specific strengths and weaknesses of each model.

### Language-Specific Performance

Comparing performance across Spanish and Catalan:

```python
# Extract language-specific metrics
spanish_baseline_f1 = baseline_lang_metrics["spanish"]["macro_avg"]["f1"]
catalan_baseline_f1 = baseline_lang_metrics["catalan"]["macro_avg"]["f1"]

spanish_bio_pos_f1 = bio_pos_lang_metrics["spanish"]["macro_avg"]["f1"]
catalan_bio_pos_f1 = bio_pos_lang_metrics["catalan"]["macro_avg"]["f1"]

spanish_second_order_f1 = second_order_lang_metrics["spanish"]["macro_avg"]["f1"]
catalan_second_order_f1 = second_order_lang_metrics["catalan"]["macro_avg"]["f1"]

# Plot comparison
labels = ["Spanish", "Catalan"]
baseline_scores = [spanish_baseline_f1, catalan_baseline_f1]
bio_pos_scores = [spanish_bio_pos_f1, catalan_bio_pos_f1]
second_order_scores = [spanish_second_order_f1, catalan_second_order_f1]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, baseline_scores, width, label='Baseline HMM')
ax.bar(x, bio_pos_scores, width, label='BIO+POS HMM')
ax.bar(x + width, second_order_scores, width, label='Second-Order HMM')

ax.set_ylabel('F1 Score')
ax.set_title('Language-Specific Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(PATH_ROOT, 'data', 'results', 'evaluation', 'language_comparison.png'))
plt.show()
```

This comparison reveals how model performance varies across languages, which is important for multilingual applications.

## Code Implementation Details

This section provides details on the core code components that implement the HMM models.

### HMM Base Class

All HMM implementations inherit from a common base class:

```python
class BaseHMM:
    """
    Base Hidden Markov Model implementation with common functionalities
    """
    def __init__(self, state_space: Set[str], vocabulary: Any, smoothing: float = 0.01):
        """
        Initialize the base HMM model with state space and vocabulary
        """
        self.state_space = sorted(list(state_space))
        self.vocabulary = sorted(list(vocabulary))
        self.smoothing = smoothing
        
        # Mapping from states/observations to indices
        self.state_to_idx = {state: idx for idx, state in enumerate(self.state_space)}
        self.obs_to_idx = {obs: idx for idx, obs in enumerate(self.vocabulary)}
        
        # HMM parameters (To be learned)
        self.initial_probs = None
        self.transition_probs = None
        self.emission_probs = None
    
    def train(self, observations: List[List[Any]], states: List[List[str]]) -> None:
        """
        Train the HMM by counting transitions and emissions
        Subclasses needed
        """
        raise NotImplementedError("Subclasses must implement the train method")
    
    def viterbi(self, observations: List[Any]) -> List[str]:
        """
        Implement the Viterbi algorithm to find the most likely sequence of states
        Subclasses needed
        """
        raise NotImplementedError("Subclasses must implement the viterbi method")
    
    def predict(self, observations: List[List[Any]]) -> List[List[str]]:
        """
        Predict state sequences for multiple observation sequences
        """
        return [self.viterbi(obs_seq) for obs_seq in observations]
    
    def save(self, output_file: str) -> None:
        """
        Save the trained HMM model to a file
        Subclasses needed
        """
        raise NotImplementedError("Subclasses must implement the save method")
    
    @classmethod
    def load(cls, input_file: str):
        """
        Load a trained HMM model from a file
        Subclasses needed
        """
        raise NotImplementedError("Subclasses must implement the load method")
```

This design allows for easy extension to different model variants.

### Viterbi Algorithm

The Viterbi algorithm is a dynamic programming approach for finding the most likely sequence of states in an HMM. Each model variant implements its own version:

1. **Baseline Viterbi**: Standard first-order Viterbi algorithm
2. **BIO+POS Viterbi**: First-order Viterbi with backoff for unknown observations
3. **Second-Order Viterbi**: Modified algorithm for trigram dependencies

The core of the Viterbi algorithm consists of:

1. **Initialization**: Set up the first step based on initial probabilities
2. **Forward Pass**: Fill in the DP table by considering all possible previous states
3. **Backward Pass**: Trace back to find the best path

### Model Training Process

All models are trained through a counting-based approach:

1. **Initialize Counts**: Start with small counts (smoothing) for all events
2. **Count Occurrences**: Iterate through training data, counting transitions and emissions
3. **Normalize**: Convert counts to probabilities by normalizing

For the second-order model, the training process is more complex, as it involves counting trigram transitions.

### Smoothing Strategies

Two smoothing strategies are employed:

1. **Laplace Smoothing**: Add a small value to all counts to avoid zero probabilities
   ```python
   # Initialize counts with smoothing
   initial_counts = np.ones(n_states) * self.smoothing
   transition_counts = np.ones((n_states, n_states)) * self.smoothing
   emission_counts = np.ones((n_states, n_obs)) * self.smoothing
   ```

2. **Backoff Smoothing**: For BIO+POS models, implemented as a hierarchical backoff
   ```python
   # Geometric mean of word and POS probabilities (with more weight to word)
   return (word_prob ** 0.7) * (pos_prob ** 0.3)
   ```

These strategies ensure robust performance even with unseen words or transitions.

## Conclusion and Future Work

### Key Findings

1. **BIO Tagging Effectiveness**: BIO tagging significantly improves entity boundary detection compared to standard labeling, with especially strong improvements for scope detection
2. **POS Information Value**: Part-of-Speech information provides valuable linguistic signals that improve model performance, particularly for distinguishing between entity types
3. **Second-Order Benefits**: The second-order HMM captures important contextual patterns in entity spans, leading to further improvements in both entity detection and scope detection
4. **Language-Specific Processing**: Multilingual performance varies across models, with some showing stronger improvements in one language over the other

### Performance Comparison

The models show clear progressive improvements:

1. **Baseline → BIO+POS**:
   - Token-level F1: +X% improvement
   - Entity-level F1: +Y% improvement

2. **BIO+POS → Second-Order**:
   - Token-level F1: +A% improvement
   - Entity-level F1: +B% improvement
   - Scope-level F1: +C% improvement

These improvements demonstrate the value of both the BIO+POS enhancements and the second-order dependencies.

### Future Work

1. **Neural Approaches**: Extend this work with neural sequence models like BiLSTM-CRF or transformer-based models
2. **Contextual Word Embeddings**: Incorporate pretrained multilingual embeddings (e.g., BERT, XLM-R)
3. **Joint Detection**: Explore joint detection of negation/uncertainty markers and their scopes
4. **Cross-Lingual Transfer**: Investigate cross-lingual transfer between Spanish and Catalan
5. **Real-World Deployment**: Evaluate the models in real clinical applications, such as information extraction systems

### Final Remarks

The comprehensive evaluation framework used in this project, with metrics at token, entity, and scope levels, provides a nuanced understanding of model performance. The progression from baseline to BIO+POS to second-order models demonstrates a systematic approach to improving performance on this important task in medical text processing.

## References

1. Agarwal, S., & Yu, H. (2010). Biomedical negation scope detection with conditional random fields. Journal of the American Medical Informatics Association, 17(6), 696-701.
2. Chapman, W. W., Bridewell, W., Hanbury, P., Cooper, G. F., & Buchanan, B. G. (2001). A simple algorithm for identifying negated findings and diseases in discharge summaries. Journal of Biomedical Informatics, 34(5), 301-310.
3. Ramshaw, L. A., & Marcus, M. P. (1999). Text chunking using transformation-based learning. In Natural language processing using very large corpora (pp. 157-176). Springer.
4. Jurafsky, D., & Martin, J. H. (2009). Speech and language processing: An introduction to natural language processing, computational linguistics, and speech recognition. Prentice Hall.
5. Marimon, M., Vivaldi, J., & Bel, N. (2017). Annotation of negation in the IULA Spanish Clinical Record Corpus. In Proceedings of the Workshop on Computational Semantics beyond Events and Roles (pp. 43-52).













# Scope-Level F1 Score Calculation in the Code

Looking at the provided code, the scope-level F1 scores are not calculated in the `utils/analysis.py` file you shared. Instead, they're computed in the `evaluation.py` file that you shared earlier, specifically in the `evaluate_scope_detection` function. Here's how this works:

## Step 1: Scope Detection Evaluation Function

The `evaluate_scope_detection` function in `evaluation.py` calculates scope-level metrics:

```python
def evaluate_scope_detection(true_labels, pred_labels, observations, is_bio=False):
    """
    Evaluate negation/uncertainty scope detection more comprehensively
    """
```

## Step 2: Extracting Scopes

It first extracts all scope spans from both true and predicted labels:

```python
def extract_scopes(labels, is_bio=False):
    if is_bio:
        return convert_bio_to_entity_spans(labels, [None] * len(labels))
    else:
        scopes = []
        current_scope = None
        start_idx = -1
        
        for i, label in enumerate(labels):
            if label != "O":  # Start or continue a scope
                if current_scope is None or current_scope != label:
                    if current_scope is not None:
                        scopes.append((current_scope, start_idx, i - 1))  # End previous scope
                    current_scope = label
                    start_idx = i
            elif current_scope is not None:  # End current scope
                scopes.append((current_scope, start_idx, i - 1))
                current_scope = None
                start_idx = -1
        
        if current_scope is not None:
            scopes.append((current_scope, start_idx, len(labels) - 1))
        
        return scopes
```

## Step 3: Calculating Overlap

For each true scope, it finds the best matching predicted scope using Jaccard similarity:

```python
def calculate_overlap(true_scope, pred_scope):
    true_type, true_start, true_end = true_scope
    pred_type, pred_start, pred_end = pred_scope
    
    if true_type != pred_type:  # Types must match
        return 0.0
    
    overlap_start = max(true_start, pred_start)
    overlap_end = min(true_end, pred_end)
    overlap_length = max(0, overlap_end - overlap_start + 1)
    
    true_length = true_end - true_start + 1
    pred_length = pred_end - pred_start + 1
    
    # Jaccard similarity: |intersection| / |union|
    union_length = true_length + pred_length - overlap_length
    return overlap_length / union_length if union_length > 0 else 0.0
```

## Step 4: Classifying Matches

For each scope, it classifies it into:
- Exact matches (overlap ≥ 0.7)
- Partial matches (0 < overlap < 0.7) 
- Missed scopes (no match found)
- False positives (predicted scope with no matching true scope)

## Step 5: Calculating F1 Scores

Finally, it calculates precision, recall, and F1 for each scope type (NEG, NSCO, UNC, USCO):

```python
for scope_type, metrics in scope_metrics.items():
    true_positives = metrics["exact"] + metrics["partial"] * 0.5  # Partial matches count as half
    precision = true_positives / max(1, true_positives + metrics["false_positive"])
    recall = true_positives / max(1, true_positives + metrics["missed"])
    f1 = 2 * precision * recall / max(1e-10, precision + recall)
    
    result[scope_type] = {
        "precision": precision, 
        "recall": recall, 
        "f1": f1, 
        "exact_matches": metrics["exact"], 
        "partial_matches": metrics["partial"], 
        "missed": metrics["missed"], 
        "false_positives": metrics["false_positive"], 
    }
```

## Key Points About Scope-Level F1:

1. It evaluates how well the model detects continuous spans of negation/uncertainty markers
2. A true positive can be either:
   - An exact match (weighted fully)
   - A partial match (weighted at 0.5)
3. The overlap threshold for exact matches is 0.7 (70% Jaccard similarity)
4. The F1 score balances precision and recall:
   - Precision: What percentage of predicted scope tokens are actually in scope
   - Recall: What percentage of actual scope tokens are correctly identified
5. The macro-average F1 is calculated across all scope types

This scope-level F1 is more challenging than token-level F1 because it requires the model to correctly identify the entire span of a negation or uncertainty scope.



# Understanding Token-Level and Entity-Level F1 Calculation

Let me explain how token-level and entity-level metrics are calculated in the code, similar to my explanation of scope-level metrics:

## Token-Level F1 Score Calculation

Token-level evaluation is handled by the `compute_metrics` function in `evaluation.py`:

```python
def compute_metrics(true_labels, pred_labels, is_bio=False):
    """Compute precision, recall, and F1-score for the HMM model predictions"""
    # Flatten the lists of labels
    y_true = []
    y_pred = []

    for true_seq, pred_seq in zip(true_labels, pred_labels):
        y_true.extend(true_seq)
        y_pred.extend(pred_seq)
```

### Key steps:

1. **Flattening**: The function flattens all sequences into single lists of true and predicted labels.
2. **Label Definition**: It defines the label set based on whether BIO tagging is used.
3. **Metric Calculation**: It uses scikit-learn's `precision_recall_fscore_support` to compute the metrics:
   ```python
   precision, recall, f1, support = precision_recall_fscore_support(
       y_true, y_pred, labels=eval_labels, average=None, zero_division=0
   )
   ```
4. **Aggregation**: It also computes macro and weighted averages:
   ```python
   macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
       y_true, y_pred, labels=eval_labels, average="macro", zero_division=0
   )
   ```

### Key characteristics of token-level F1:

- **Independence**: Each token is treated as an independent classification problem
- **Simplicity**: No consideration of spans or continuity
- **Lower Bar**: It's easier to achieve higher token-level F1 than entity-level or scope-level
- **Limited Context**: Doesn't evaluate the model's ability to identify complete entities

## Entity-Level F1 Score Calculation

Entity-level evaluation is handled by the `get_entity_based_metrics` function:

```python
def get_entity_based_metrics(true_labels, pred_labels, is_bio=False):
    """Compute entity-based metrics (treating each contiguous chunk as one entity)"""
```

### Key steps:

1. **Entity Extraction**: It extracts entity spans from both true and predicted labels:
   ```python
   def extract_entities(labels):
       entities = []
       
       if is_bio:
           entities = convert_bio_to_entity_spans(labels, [None] * len(labels))
       else:
           # Handle non-BIO tagging by finding continuous chunks
           current_entity = None
           start_idx = 0
           
           for i, label in enumerate(labels):
               if label == "O":  # End current entity
                   if current_entity:
                       entities.append((current_entity, start_idx, i - 1))
                       current_entity = None
               else:
                   if current_entity != label:  # New entity type
                       if current_entity:
                           entities.append((current_entity, start_idx, i - 1))
                       current_entity = label
                       start_idx = i
   ```

2. **Exact Matching**: It counts entities as correct only if they have an exact match:
   ```python
   for entity_type, start, end in true_entities:
       if entity_type in true_by_type:
           true_by_type[entity_type] += 1
           if (entity_type, start, end) in pred_entities:  # Exact match check
               correct_by_type[entity_type] += 1
   ```

3. **Metric Calculation**: It calculates precision, recall, and F1 for each entity type:
   ```python
   precision = correct_by_type[entity_type] / max(1, pred_by_type[entity_type])
   recall = correct_by_type[entity_type] / max(1, true_by_type[entity_type])
   f1 = 2 * precision * recall / max(1e-10, precision + recall)
   ```

### Key characteristics of entity-level F1:

- **Exact Matching**: An entity is only considered correct if its type, start, and end boundaries exactly match
- **Stricter Evaluation**: Much harder to achieve high scores than token-level
- **All-or-Nothing**: One token error in an entity span means the entire entity is wrong
- **No Partial Credit**: Unlike scope-level, there's no concept of partial matches
- **Boundary Awareness**: Tests the model's ability to correctly identify entity boundaries

## Comparing the Evaluation Approaches

Here's how the three evaluation approaches differ:

| Feature | Token-Level | Entity-Level | Scope-Level |
|---------|------------|--------------|-------------|
| Unit of evaluation | Individual tokens | Complete entities | Scope spans |
| Matching criteria | Exact token match | Exact entity match | Exact or partial overlap |
| Partial credit | No | No | Yes (0.5 weight for partial) |
| Difficulty | Easiest | Harder | Hardest |
| What it measures | Basic classification | Entity boundary detection | Scope coverage |
| Continuous spans | Not considered | Required | Required with overlap calculation |

These different evaluation metrics provide complementary perspectives on model performance, with token-level being the most lenient and scope-level often being the most challenging and informative for negation/uncertainty detection tasks.


# F1 Improvement Calculation Between Models

The code calculates the F1 improvement percentages between models using a specific methodology to show how each enhancement (BIO+POS and Second-Order) improves over the previous model. Let me explain how this is done:

## 1. Baseline vs. BIO+POS Comparison (in hmm_pos_bio.ipynb)

```python
try:
    with open(os.path.join(PATH_ROOT, "data", "results", "evaluation", "evaluation_baseline.json"), 'r') as f:
        baseline_metrics = json.load(f)  # Load baseline metrics from saved file
        
    # Extract F1 scores from both models
    baseline_token_f1 = baseline_metrics["token_metrics"]["macro_avg"]["f1"]
    baseline_entity_f1 = baseline_metrics["entity_metrics"]["macro_avg"]["f1"]
    
    bio_pos_token_f1 = metrics["macro_avg"]["f1"]
    bio_pos_entity_f1 = entity_metrics["macro_avg"]["f1"]
    
    # Calculate percentage improvement
    token_improvement = (bio_pos_token_f1 - baseline_token_f1) / baseline_token_f1 * 100
    entity_improvement = (bio_pos_entity_f1 - baseline_entity_f1) / baseline_entity_f1 * 100
    
    print(f"Token-level F1 improvement: {token_improvement:.2f}%")
    print(f"Entity-level F1 improvement: {entity_improvement:.2f}%")
```

The calculation is:
- **Percentage improvement** = ((new_F1 - baseline_F1) / baseline_F1) * 100

## 2. First-Order vs. Second-Order Comparison (in hmm_second_order.ipynb)

```python
try:
    with open(os.path.join(PATH_ROOT, "data", "results", "evaluation", "evaluation_bio_pos.json"), 'r') as f:
        bio_pos_metrics = json.load(f)  # Load first-order metrics from saved file
    
    # Extract F1 scores from both models
    bio_pos_token_f1 = bio_pos_metrics["token_metrics"]["macro_avg"]["f1"]
    bio_pos_entity_f1 = bio_pos_metrics["entity_metrics"]["macro_avg"]["f1"]
    bio_pos_scope_f1 = bio_pos_metrics["scope_metrics"]["macro_avg"]["f1"]
    
    second_order_token_f1 = metrics["macro_avg"]["f1"]
    second_order_entity_f1 = entity_metrics["macro_avg"]["f1"]
    second_order_scope_f1 = scope_metrics["macro_avg"]["f1"]
    
    # Calculate percentage improvement
    second_vs_first_token = (second_order_token_f1 - bio_pos_token_f1) / bio_pos_token_f1 * 100
    second_vs_first_entity = (second_order_entity_f1 - bio_pos_entity_f1) / bio_pos_entity_f1 * 100
    second_vs_first_scope = (second_order_scope_f1 - bio_pos_scope_f1) / bio_pos_scope_f1 * 100
    
    print(f"Second-Order vs First-Order (Token-level F1): {second_vs_first_token:.2f}% improvement")
    print(f"Second-Order vs First-Order (Entity-level F1): {second_vs_first_entity:.2f}% improvement")
    print(f"Second-Order vs First-Order (Scope-level F1): {second_vs_first_scope:.2f}% improvement")
```

## 3. Entity-Type Specific Improvements

The second-order notebook also calculates improvements for each specific entity type:

```python
print("\nEntity-level F1 comparison by entity type:")
print("Entity Type\tFirst-Order\tSecond-Order\tImprovement")
print("-" * 60)

for entity_type in ["NEG", "NSCO", "UNC", "USCO"]:
    first_f1 = bio_pos_metrics["entity_metrics"][entity_type]["f1"]
    second_f1 = entity_metrics[entity_type]["f1"]
    improvement = (second_f1 - first_f1) / first_f1 * 100
    
    print(f"{entity_type}\t\t{first_f1:.4f}\t\t{second_f1:.4f}\t\t{improvement:+.2f}%")
```

## 4. Token-Level Improvement Counts

In addition to percentage improvements, there's also a direct token-level comparison in the `compare_with_first_order` function:

```python
comparison = compare_with_first_order(
    test_data["states"], 
    test_predictions,  # second-order predictions
    first_order_preds,  # loaded from saved file
    test_data["observations"]
)

print("\n-- COMPARISON WITH FIRST-ORDER MODEL --")
print(f"Total improvements: {comparison['improve_count']} tokens")
print(f"Total regressions: {comparison['regress_count']} tokens")
print(f"Net improvement: {comparison['improve_count'] - comparison['regress_count']} tokens")
```

This counts:
- **Improvements**: Number of tokens where second-order correctly predicted the label but first-order didn't
- **Regressions**: Number of tokens where first-order correctly predicted the label but second-order didn't
- **Net improvement**: Improvements minus regressions

## Key Points About F1 Improvement Calculations:

1. **Relative Percentage**: All improvements are calculated as relative percentages rather than absolute differences
2. **Macro Averages**: The main comparisons use macro-averaged F1 scores (equal weight to all classes)
3. **Multi-Level Evaluation**: Improvements are calculated at token, entity, and scope levels
4. **Entity-Specific Analysis**: For the second-order model, improvements are broken down by entity type
5. **Direct Token Comparison**: The token-by-token comparison provides a direct count of improved predictions
6. **Saved Metrics**: All calculations rely on loading metrics from previously saved JSON files

This comprehensive approach to model comparison helps identify not just whether a model improves overall, but exactly where and how it improves, which is crucial for understanding the benefits of each enhancement (BIO+POS tagging and second-order dependencies).



# Entity-Level Metrics for the Baseline Model

You've identified an important point! The concept of "entities" can indeed exist even without BIO tagging, though it's handled differently. Let me explain:

## Entities in the Baseline Model (without BIO)

In the baseline model, the code still calculates "entity-level" metrics despite not using BIO tagging. Here's how this works:

1. **Entity Definition**: For the baseline model, an "entity" is defined as a contiguous sequence of the same non-"O" tag.

2. **How Entities are Extracted**: The `get_entity_based_metrics` function in `evaluation.py` includes logic to extract entities differently depending on whether BIO tagging is used:

```python
if is_bio:
    entities = convert_bio_to_entity_spans(labels, [None] * len(labels))
else:
    current_entity = None
    start_idx = 0
    
    for i, label in enumerate(labels):
        if label == "O":  # End current entity
            if current_entity:
                entities.append((current_entity, start_idx, i - 1))
                current_entity = None
        else:
            if current_entity != label:  # Start a new entity if label changes
                if current_entity:
                    entities.append((current_entity, start_idx, i - 1))
                current_entity = label
                start_idx = i
```

## Limitations of Baseline Entity Recognition

The baseline approach to entity extraction has important limitations:

1. **Cannot Distinguish Adjacent Entities**: The biggest limitation is that without BIO tagging, there's no way to distinguish between adjacent entities of the same type.

   For example, with BIO tags, you could have:
   ```
   O B-NEG I-NEG I-NEG O B-NEG I-NEG O
   ```
   (Two separate NEG entities)

   But in the baseline model, this would be:
   ```
   O NEG NEG NEG O NEG NEG O
   ```
   Which would be treated as two separate NEG entities only because there's an "O" between them.

2. **Entity Boundaries**: Without BIO tagging, entity boundaries are determined solely by transitions between different tags, which may not accurately represent linguistic entity boundaries.

## Why the Comparison is Still Valid

Despite these limitations, comparing entity-level F1 between baseline and BIO models is still meaningful:

1. The baseline approach represents how well the model can identify contiguous spans of the same tag type
2. The BIO approach represents the more linguistically accurate concept of entities with explicitly marked beginnings and insides
3. The improvement when moving to BIO tagging shows the value of the more sophisticated tagging scheme

## Summary

So the "entity-level" metrics for the baseline model are actually measuring the model's ability to identify contiguous regions of the same tag, rather than true linguistic entities with well-defined boundaries. This is one of the reasons why moving to BIO tagging typically shows improvement - it allows for more accurate entity boundary detection and the ability to distinguish adjacent entities of the same type.












Based on the provided `hmm.py` file, Laplace smoothing is used throughout all the HMM implementations with a value of 0.01. Here's a detailed breakdown of how it's used:

### Where Laplace smoothing is applied:

1. **Initial count matrices** - In all model variants, the count matrices are initialized with the smoothing value:
   ```python
   # Initialize counts with smoothing
   initial_counts = np.ones(n_states) * self.smoothing
   transition_counts = np.ones((n_states, n_states)) * self.smoothing
   emission_counts = np.ones((n_states, n_obs)) * self.smoothing
   ```

2. **Handling out-of-vocabulary (OOV) observations** - In the Viterbi algorithm:
   ```python
   # If observation not in vocabulary, use a small probability
   V[t, s] = max_prob + np.log(self.smoothing)
   ```

3. **Backoff strategy** in the POS models:
   ```python
   # Default to smoothing values for unseen word/POS combinations
   word_prob = self.smoothing
   pos_prob = self.smoothing
   ```

4. **Second-order transitions** in the more complex models:
   ```python
   # Also applied to bigram probabilities and trigram transitions
   initial_bigram_counts = np.ones((n_states, n_states)) * self.smoothing
   transition_counts = np.ones((n_states, n_states, n_states)) * self.smoothing
   ```

### Purpose of the smoothing:

1. **Prevent zero probabilities**: By adding a small value (0.01) to all counts, it ensures no probability is exactly zero, which would cause mathematical problems when calculating log probabilities in the Viterbi algorithm.

2. **Handle unseen events**: For transitions or emissions not observed in the training data, smoothing provides a small non-zero probability, making the model more robust to unseen patterns.

3. **Improve generalization**: By avoiding overfitting to the training data's exact patterns, the model can better generalize to unseen data with similar but not identical patterns.

The value of 0.01 represents a reasonable balance - small enough not to significantly alter the probabilities of frequently observed events, but large enough to give unseen events a meaningful chance of being selected when they're the only plausible option.