import numpy as np
import pickle
from typing import Dict, List, Tuple, Set, Any

class BaseHMM:
    """
    Base Hidden Markov Model implementation with common functionalities for all HMM variants
    """
    def __init__(self, state_space: Set[str], vocabulary: Any, smoothing: float = 0.01):
        """
        Initialize the base HMM model with state space and vocabulary
        
        Parameters:
            state_space (Set[str]): Set of possible states
            vocabulary (Any): Set of possible observations (Change for different HMM models)
            smoothing (float): Laplace smoothing parameter
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
        
        Parameters:
            observations (List[List[Any]]): List of observation sequences
            
        Returns:
            List[List[str]]: List of predicted state sequences
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


class HMMBaseline(BaseHMM):
    """
    Basic Hidden Markov Model implementation for sequence labeling
    """
    def __init__(self, state_space: Set[str], vocabulary: Set[str], smoothing: float = 0.01):
        """
        Initialize the HMM model with state space and vocabulary
        
        Parameters:
            state_space (Set[str]): Set of possible states
            vocabulary (Set[str]): Set of possible observations
            smoothing (float): Laplace smoothing parameter
        """
        super().__init__(state_space, vocabulary, smoothing)
    
    def train(self, observations: List[List[Any]], states: List[List[str]]) -> None:
        """
        Train the HMM by counting transitions and emissions
        
        Parameters:
            observations (List[List[Any]]): List of observation sequences
            states (List[List[str]]): List of state sequences
        """
        n_states = len(self.state_space)
        n_obs = len(self.vocabulary)
        
        # Initialize counts with smoothing
        initial_counts = np.ones(n_states) * self.smoothing # P(s_0) is the initial state
        transition_counts = np.ones((n_states, n_states)) * self.smoothing # P(s_t | s_{t-1}) is the transition
        emission_counts = np.ones((n_states, n_obs)) * self.smoothing # P(o_t | s_t) is the emission
        
        # Count occurrences
        for obs_seq, state_seq in zip(observations, states):
            if state_seq: # Initial state
                try:
                    initial_counts[self.state_to_idx[state_seq[0]]] += 1
                except KeyError:
                    # Skip if state is not in state space (unlikely)
                    pass
            
            # Transitions and emissions
            for i in range(len(state_seq)):
                try:
                    state_idx = self.state_to_idx[state_seq[i]]
                    
                    # Emission
                    if i < len(obs_seq) and obs_seq[i] in self.obs_to_idx:
                        obs_idx = self.obs_to_idx[obs_seq[i]]
                        emission_counts[state_idx, obs_idx] += 1
                    
                    # Transition (if not last)
                    if i < len(state_seq) - 1 and state_seq[i+1] in self.state_to_idx:
                        next_state_idx = self.state_to_idx[state_seq[i+1]]
                        transition_counts[state_idx, next_state_idx] += 1
                except KeyError:
                    # Skip if state or observation is not in space
                    continue
        
        # Normalize to get probabilities
        self.initial_probs = initial_counts / np.sum(initial_counts)
        self.transition_probs = transition_counts / np.sum(transition_counts, axis=1, keepdims=True)
        self.emission_probs = emission_counts / np.sum(emission_counts, axis=1, keepdims=True)
    
    def viterbi(self, observations: List[Any]) -> List[str]:
        """
        Implement the Viterbi algorithm to find the most likely sequence of states
        
        Parameters:
            observations (List[Any]): Sequence of observations
            
        Returns:
            List[str]: Most likely sequence of states
        """
        n_states = len(self.state_space) # Number of states
        T = len(observations) # Length of the observation sequence !!!
        
        V = np.zeros((T, n_states)) # Initialize Viterbi matrix and backpointers
        backpointers = np.zeros((T, n_states), dtype=int) # Backpointers to store the previous state
        
        # Initialize init probabilities
        for s in range(n_states):
            if observations[0] in self.obs_to_idx:
                obs_idx = self.obs_to_idx[observations[0]] # Index first observation
                V[0, s] = np.log(self.initial_probs[s]) + np.log(self.emission_probs[s, obs_idx])
            else:
                V[0, s] = np.log(self.initial_probs[s]) + np.log(self.smoothing) # Observation OOV Smoothing by Laplace
        
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
                    # If observation not in vocabulary, use a small probability
                    V[t, s] = max_prob + np.log(self.smoothing)
        
        # Backward pass to find the best path
        best_path = np.zeros(T, dtype=int) # Init best path
        best_path[T-1] = np.argmax(V[T-1, :]) # Last state is the one with max prob
        
        for t in range(T-2, -1, -1):
            best_path[t] = backpointers[t+1, best_path[t+1]]
        
        return [self.state_space[idx] for idx in best_path] # Convert indices back to states
    
    def save(self, output_file: str) -> None:
        """
        Save the trained HMM model to a file
        
        Parameters:
            output_file (str): Path to save the model
        """
        model_data = {
            "state_space": self.state_space,
            "vocabulary": self.vocabulary,
            "state_to_idx": self.state_to_idx,
            "obs_to_idx": self.obs_to_idx,
            "initial_probs": self.initial_probs,
            "transition_probs": self.transition_probs,
            "emission_probs": self.emission_probs,
            "smoothing": self.smoothing
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, input_file: str) -> 'HMMBaseline':
        """
        Load a trained HMM model from a file
        
        Parameters:
            input_file (str): Path to the saved model
            
        Returns:
            HMMBaseline: Loaded HMM model
        """
        with open(input_file, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            state_space=set(model_data["state_space"]),
            vocabulary=set(model_data["vocabulary"]),
            smoothing=model_data["smoothing"]
        )
        
        model.state_to_idx = model_data["state_to_idx"]
        model.obs_to_idx = model_data["obs_to_idx"]
        model.initial_probs = model_data["initial_probs"]
        model.transition_probs = model_data["transition_probs"]
        model.emission_probs = model_data["emission_probs"]
        
        return model


class HMMPOS(BaseHMM):
    """
    Enhanced Hidden Markov Model implementation that uses (word, POS) tuples as observations
    """
    def __init__(self, state_space: Set[str], vocabulary: Set[Tuple[str, str]], smoothing: float = 0.01):
        """
        Initialize the HMM model with state space and vocabulary
        
        Parameters:
            state_space (Set[str]): Set of possible states
            vocabulary (Set[Tuple[str, str]]): Set of possible observations (word, POS)
            smoothing (float): Laplace smoothing parameter
        """
        super().__init__(state_space, vocabulary, smoothing)
        
        # Extract words and POS tags separately for backoff
        self.words = sorted(list({word for word, _ in vocabulary})) # Unique words
        self.pos_tags = sorted(list({pos for _, pos in vocabulary})) # Unique POS tags
        
        self.word_to_idx = {word: idx for idx, word in enumerate(self.words)} # Mapping from words to indices
        self.pos_to_idx = {pos: idx for idx, pos in enumerate(self.pos_tags)} # Mapping from POS tags to indices
        
        # Backoff parameters
        self.word_emission_probs = None
        self.pos_emission_probs = None
    
    def train(self, observations: List[List[Tuple[str, str]]], states: List[List[str]]) -> None:
        """
        Train the HMM by counting transitions and emissions
        
        Parameters:
            observations (List[List[Tuple[str, str]]]): List of observation sequences
            states (List[List[str]]): List of state sequences
        """
        n_states = len(self.state_space)
        n_obs = len(self.vocabulary)
        n_words = len(self.words)
        n_pos = len(self.pos_tags)
        
        # Init counts with smoothing
        initial_counts = np.ones(n_states) * self.smoothing 
        transition_counts = np.ones((n_states, n_states)) * self.smoothing
        emission_counts = np.ones((n_states, n_obs)) * self.smoothing
        
        # Backoff counts
        word_emission_counts = np.ones((n_states, n_words)) * self.smoothing
        pos_emission_counts = np.ones((n_states, n_pos)) * self.smoothing
        
        # Count occurrences
        for obs_seq, state_seq in zip(observations, states):
            if state_seq:
                try:
                    initial_counts[self.state_to_idx[state_seq[0]]] += 1
                except KeyError:
                    pass # Unlikely
            
            # Transitions and emissions
            for i in range(len(state_seq)):
                try:
                    state_idx = self.state_to_idx[state_seq[i]]
                    
                    # Emission for (word, POS) tuple
                    if i < len(obs_seq):
                        obs = obs_seq[i]
                        word, pos = obs
                        
                        if obs in self.obs_to_idx: # Full tuple emission
                            obs_idx = self.obs_to_idx[obs]
                            emission_counts[state_idx, obs_idx] += 1
                        
                        if word in self.word_to_idx: # Backoff to word only
                            word_idx = self.word_to_idx[word]
                            word_emission_counts[state_idx, word_idx] += 1
                        
                        if pos in self.pos_to_idx: # Backoff to POS only
                            pos_idx = self.pos_to_idx[pos]
                            pos_emission_counts[state_idx, pos_idx] += 1
                    
                    if i < len(state_seq) - 1 and state_seq[i+1] in self.state_to_idx:
                        next_state_idx = self.state_to_idx[state_seq[i+1]]
                        transition_counts[state_idx, next_state_idx] += 1
                except KeyError:
                    continue
        
        self.initial_probs = initial_counts / np.sum(initial_counts)
        self.transition_probs = transition_counts / np.sum(transition_counts, axis=1, keepdims=True)
        self.emission_probs = emission_counts / np.sum(emission_counts, axis=1, keepdims=True)
        
        self.word_emission_probs = word_emission_counts / np.sum(word_emission_counts, axis=1, keepdims=True)
        self.pos_emission_probs = pos_emission_counts / np.sum(pos_emission_counts, axis=1, keepdims=True)
    
    def get_emission_prob(self, state_idx: int, obs: Tuple[str, str]) -> float:
        """
        Get emission probability with backoff for unknown observations
        
        Parameters:
            state_idx (int): Index of the state
            obs (Tuple[str, str]): Observation tuple (word, POS)
            
        Returns:
            float: Emission probability
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
    
    def viterbi(self, observations: List[Tuple[str, str]]) -> List[str]:
        """
        Implement the Viterbi algorithm to find the most likely sequence of states
        
        Parameters:
            observations (List[Tuple[str, str]]): Sequence of observations
            
        Returns:
            List[str]: Most likely sequence of states
        """
        n_states = len(self.state_space)
        T = len(observations)
        
        V = np.zeros((T, n_states))
        backpointers = np.zeros((T, n_states), dtype=int)
        
        for s in range(n_states):
            # Get emission probability with backoff
            emission_prob = self.get_emission_prob(s, observations[0])
            V[0, s] = np.log(self.initial_probs[s]) + np.log(emission_prob)
        
        # Forward pass
        for t in range(1, T):
            for s in range(n_states):
                # Find the most likely previous state
                probs = V[t-1, :] + np.log(self.transition_probs[:, s])
                backpointers[t, s] = np.argmax(probs)
                max_prob = probs[backpointers[t, s]]
                
                # Add emission probability with backoff
                emission_prob = self.get_emission_prob(s, observations[t])
                V[t, s] = max_prob + np.log(emission_prob)
        
        # Backward pass to find the best path
        best_path = np.zeros(T, dtype=int)
        best_path[T-1] = np.argmax(V[T-1, :])
        
        for t in range(T-2, -1, -1):
            best_path[t] = backpointers[t+1, best_path[t+1]]
        
        return [self.state_space[idx] for idx in best_path]
    
    def save(self, output_file: str) -> None:
        """
        Save the trained HMM model to a file
        
        Parameters:
            output_file (str): Path to save the model
        """
        model_data = { # Requires more information for POS tagging
            "state_space": self.state_space,
            "vocabulary": self.vocabulary,
            "words": self.words,
            "pos_tags": self.pos_tags,
            "state_to_idx": self.state_to_idx,
            "obs_to_idx": self.obs_to_idx,
            "word_to_idx": self.word_to_idx,
            "pos_to_idx": self.pos_to_idx,
            "initial_probs": self.initial_probs,
            "transition_probs": self.transition_probs,
            "emission_probs": self.emission_probs,
            "word_emission_probs": self.word_emission_probs,
            "pos_emission_probs": self.pos_emission_probs,
            "smoothing": self.smoothing
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, input_file: str) -> 'HMMPOS':
        """
        Load a trained HMM model from a file
        
        Parameters:
            input_file (str): Path to the saved model
            
        Returns:
            HMMPOS: Loaded HMM model
        """
        with open(input_file, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            state_space=set(model_data["state_space"]),
            vocabulary=set(model_data["vocabulary"]),
            smoothing=model_data["smoothing"]
        )
        
        model.words = model_data["words"]
        model.pos_tags = model_data["pos_tags"]
        model.state_to_idx = model_data["state_to_idx"]
        model.obs_to_idx = model_data["obs_to_idx"]
        model.word_to_idx = model_data["word_to_idx"]
        model.pos_to_idx = model_data["pos_to_idx"]
        model.initial_probs = model_data["initial_probs"]
        model.transition_probs = model_data["transition_probs"]
        model.emission_probs = model_data["emission_probs"]
        model.word_emission_probs = model_data["word_emission_probs"]
        model.pos_emission_probs = model_data["pos_emission_probs"]
        
        return model


class HMMBIOPOS(HMMPOS):
    """
    Enhanced Hidden Markov Model implementation that uses BIO tagging
    for better boundary detection in sequence labeling with POS information
    """
    def __init__(self, state_space: Set[str], vocabulary: Set[Tuple[str, str]], smoothing: float = 0.01):
        """
        Initialize the HMM model with state space and vocabulary
        
        Parameters:
            state_space (Set[str]): Set of possible states (with BIO prefixes)
            vocabulary (Set[Tuple[str, str]]): Set of possible observations (word, POS)
            smoothing (float): Laplace smoothing parameter
        """
        super().__init__(state_space, vocabulary, smoothing) # Call the parent constructor POS
    
    def train(self, observations: List[List[Tuple[str, str]]], states: List[List[str]]) -> None:
        """
        Train the HMM by counting transitions and emissions
        
        Parameters:
            observations (List[List[Tuple[str, str]]]): List of observation sequences
            states (List[List[str]]): List of state sequences with BIO tags
        """
        super().train(observations, states) # Use the base POS training method
        
        # Additional BIO-specific transition constraints
        for i, state in enumerate(self.state_space):
            if state.startswith("B-"):
                entity_type = state[2:]  # Extract entity type (e.g., "NEG", "NSCO")
                
                # Increase probability of B-X -> I-X transitions (e.g., B-NEG -> I-NEG more likely than B-NEG -> O)
                i_state = f"I-{entity_type}"
                if i_state in self.state_to_idx:
                    i_idx = self.state_to_idx[i_state]
                    # Increase P(I-X | B-X) without normalization
                    # We don't normalize to maintain the overall distribution
                    self.transition_probs[i, i_idx] *= 5  # Add stronger bias because of BIO pairings
                    
                    # Normalize the row to ensure it sums to 1
                    self.transition_probs[i, :] /= np.sum(self.transition_probs[i, :])

class HMMSecondOrderBaseline(HMMBaseline):
    """
    Second-order (trigram) Hidden Markov Model implementation
    that captures longer dependencies in sequence labeling
    """
    def __init__(self, state_space: Set[str], vocabulary: Set[str], smoothing: float = 0.01):
        """
        Initialize the second-order HMM model with state space and vocabulary
        
        Parameters:
            state_space (Set[str]): Set of possible states
            vocabulary (Set[str]): Set of possible observations
            smoothing (float): Laplace smoothing parameter
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
    
    def train(self, observations: List[List[Any]], states: List[List[str]]) -> None:
        """
        Train the second-order HMM by counting transitions and emissions
        
        Parameters:
            observations (List[List[Any]]): List of observation sequences
            states (List[List[str]]): List of state sequences
        """
        n_states = len(self.state_space)
        n_obs = len(self.vocabulary)
        
        # Initialize counts with smoothing
        initial_counts = np.ones(n_states) * self.smoothing
        initial_bigram_counts = np.ones((n_states, n_states)) * self.smoothing
        
        # For second-order transitions P(s_t | s_{t-1}, s_{t-2}) - We need a 3D array
        transition_counts = np.ones((n_states, n_states, n_states)) * self.smoothing
        emission_counts = np.ones((n_states, n_obs)) * self.smoothing

        # Count occurrences of states and observations
        for obs_seq, state_seq in zip(observations, states):
            if len(state_seq) >= 1:
                try:
                    state_idx = self.state_to_idx[state_seq[0]]
                    initial_counts[state_idx] += 1
                    
                    # Emission for first state
                    if len(obs_seq) >= 1:
                        obs = obs_seq[0]
                        
                        if obs in self.obs_to_idx:
                            obs_idx = self.obs_to_idx[obs]
                            emission_counts[state_idx, obs_idx] += 1
                except KeyError:
                    pass
            
            if len(state_seq) >= 2:
                try: # First bigram counts
                    idx_t_0 = self.state_to_idx[state_seq[0]]
                    idx_t_1 = self.state_to_idx[state_seq[1]]
                    initial_bigram_counts[idx_t_0, idx_t_1] += 1
                    
                    # Emission for second state
                    if len(obs_seq) >= 2:
                        obs = obs_seq[1]
                        
                        if obs in self.obs_to_idx:
                            obs_idx = self.obs_to_idx[obs]
                            emission_counts[idx_t_1, obs_idx] += 1
                except KeyError:
                    pass
            
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
                    if i < len(obs_seq):
                        obs = obs_seq[i]
                        
                        if obs in self.obs_to_idx:
                            obs_idx = self.obs_to_idx[obs]
                            emission_counts[idx_t, obs_idx] += 1
                except KeyError:
                    continue
        
        self.initial_probs = initial_counts / np.sum(initial_counts)
        
        self.initial_bigram_probs = np.zeros_like(initial_bigram_counts)
        for i in range(n_states):
            if np.sum(initial_bigram_counts[i, :]) > 0:
                self.initial_bigram_probs[i, :] = initial_bigram_counts[i, :] / np.sum(initial_bigram_counts[i, :])
        
        self.transition_probs = np.zeros_like(transition_counts)
        for i in range(n_states):
            for j in range(n_states):
                if np.sum(transition_counts[i, j, :]) > 0:
                    self.transition_probs[i, j, :] = transition_counts[i, j, :] / np.sum(transition_counts[i, j, :])
        
        # Normalize emission probabilities
        self.emission_probs = emission_counts / np.sum(emission_counts, axis=1, keepdims=True)
    
    def viterbi(self, observations: List[Any]) -> List[str]:
        """
        Implement a modified Viterbi algorithm for second-order HMM decoding
        
        Parameters:
            observations (List[Any]): Sequence of observations
            
        Returns:
            List[str]: Most likely sequence of states
        """
        if not observations:
            return []
        
        n_states = len(self.state_space)
        T = len(observations)
        
        if T == 1:
            # For single observation, just return the most likely state
            scores = np.zeros(n_states)
            for s in range(n_states):
                if observations[0] in self.obs_to_idx:
                    obs_idx = self.obs_to_idx[observations[0]]
                    scores[s] = np.log(self.initial_probs[s]) + np.log(self.emission_probs[s, obs_idx])
                else:
                    scores[s] = np.log(self.initial_probs[s]) + np.log(self.smoothing)
            
            best_state_idx = np.argmax(scores)
            return [self.state_space[best_state_idx]]
        
        # Initialize DP table for Viterbi
        V = np.zeros((T, n_states, n_states))
        
        # Initialize backpointers
        bp = np.zeros((T, n_states, n_states), dtype=int)
        
        # Base case: t = 1
        for s0 in range(n_states):
            for s1 in range(n_states):
                prob_s0 = self.initial_probs[s0]
                prob_s1_given_s0 = self.initial_bigram_probs[s0, s1]
                
                # Emission for s0
                if observations[0] in self.obs_to_idx:
                    obs_idx = self.obs_to_idx[observations[0]]
                    emission_prob_s0 = self.emission_probs[s0, obs_idx]
                else:
                    emission_prob_s0 = self.smoothing
                
                # Emission for s1
                if observations[1] in self.obs_to_idx:
                    obs_idx = self.obs_to_idx[observations[1]]
                    emission_prob_s1 = self.emission_probs[s1, obs_idx]
                else:
                    emission_prob_s1 = self.smoothing
                
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
                    
                    # Emission for s2
                    if observations[t] in self.obs_to_idx:
                        obs_idx = self.obs_to_idx[observations[t]]
                        emission_prob = self.emission_probs[s2, obs_idx]
                    else:
                        emission_prob = self.smoothing
                    
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
    
    def save(self, output_file: str) -> None:
        """
        Save the trained second-order HMM model to a file
        
        Parameters:
            output_file (str): Path to save the model
        """
        model_data = {
            "state_space": self.state_space,
            "vocabulary": self.vocabulary,
            "state_to_idx": self.state_to_idx,
            "obs_to_idx": self.obs_to_idx,
            "initial_probs": self.initial_probs,
            "initial_bigram_probs": self.initial_bigram_probs,
            "transition_probs": self.transition_probs,
            "emission_probs": self.emission_probs,
            "smoothing": self.smoothing
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, input_file: str) -> 'HMMSecondOrder':
        """
        Load a trained second-order HMM model from a file
        
        Parameters:
            input_file (str): Path to the saved model
            
        Returns:
            HMMSecondOrder: Loaded second-order HMM model
        """
        with open(input_file, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            state_space=set(model_data["state_space"]),
            vocabulary=set(model_data["vocabulary"]),
            smoothing=model_data["smoothing"]
        )
        
        model.state_to_idx = model_data["state_to_idx"]
        model.obs_to_idx = model_data["obs_to_idx"]
        model.initial_probs = model_data["initial_probs"]
        model.initial_bigram_probs = model_data["initial_bigram_probs"]
        model.transition_probs = model_data["transition_probs"]
        model.emission_probs = model_data["emission_probs"]
        
        return model

class HMMSecondOrder(HMMBIOPOS):
    """
    Second-order (trigram) Hidden Markov Model implementation that uses BIO tagging
    with POS information for better entity detection
    """
    def __init__(self, state_space: Set[str], vocabulary: Set[Tuple[str, str]], smoothing: float = 0.01):
        """
        Initialize the second-order HMM model with state space and vocabulary
        
        Parameters:
            state_space (Set[str]): Set of possible states (with BIO prefixes)
            vocabulary (Set[Tuple[str, str]]): Set of possible observations (word, POS)
            smoothing (float): Laplace smoothing parameter
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
    
    def train(self, observations: List[List[Tuple[str, str]]], states: List[List[str]]) -> None:
        """
        Train the second-order HMM by counting transitions and emissions
        
        Parameters:
            observations (List[List[Tuple[str, str]]]): List of observation sequences
            states (List[List[str]]): List of state sequences with BIO tags
        """
        n_states = len(self.state_space)
        n_extended = len(self.extended_state_space)
        n_obs = len(self.vocabulary)
        n_words = len(self.words)
        n_pos = len(self.pos_tags)
        
        # Initialize counts with smoothing
        initial_counts = np.ones(n_states) * self.smoothing
        initial_bigram_counts = np.ones((n_states, n_states)) * self.smoothing
        
        # For second-order transitions P(s_t | s_{t-1}, s_{t-2}) - We need a 3D array: [s_{t-2}, s_{t-1}, s_t]
        transition_counts = np.ones((n_states, n_states, n_states)) * self.smoothing # P(s_t | s_{t-1}, s_{t-2}) is the transition of the second order
        emission_counts = np.ones((n_states, n_obs)) * self.smoothing # P(o_t | s_t) is the emission, this do not change

        # Backoff counts
        word_emission_counts = np.ones((n_states, n_words)) * self.smoothing # Emission counts for words
        pos_emission_counts = np.ones((n_states, n_pos)) * self.smoothing # Emission counts for POS tags

        # Count occurrences of states and observations
        for obs_seq, state_seq in zip(observations, states):
            if len(state_seq) >= 1:
                try:
                    state_idx = self.state_to_idx[state_seq[0]]
                    initial_counts[state_idx] += 1
                    
                    # Emission for first state
                    if len(obs_seq) >= 1:
                        obs = obs_seq[0]
                        word, pos = obs
                        
                        # Full tuple emission
                        if obs in self.obs_to_idx:
                            obs_idx = self.obs_to_idx[obs]
                            emission_counts[state_idx, obs_idx] += 1
                        
                        # Backoff to word only
                        if word in self.word_to_idx:
                            word_idx = self.word_to_idx[word]
                            word_emission_counts[state_idx, word_idx] += 1
                        
                        # Backoff to POS only
                        if pos in self.pos_to_idx:
                            pos_idx = self.pos_to_idx[pos]
                            pos_emission_counts[state_idx, pos_idx] += 1
                except KeyError:
                    pass
            
            if len(state_seq) >= 2:
                try: # First bigram counts
                    idx_t_0 = self.state_to_idx[state_seq[0]]
                    idx_t_1 = self.state_to_idx[state_seq[1]]
                    initial_bigram_counts[idx_t_0, idx_t_1] += 1
                    
                    # Emission for second state
                    if len(obs_seq) >= 2:
                        obs = obs_seq[1]
                        word, pos = obs
                        
                        # Full tuple emission
                        if obs in self.obs_to_idx:
                            obs_idx = self.obs_to_idx[obs]
                            emission_counts[idx_t_1, obs_idx] += 1
                        
                        # Backoff to word only
                        if word in self.word_to_idx:
                            word_idx = self.word_to_idx[word]
                            word_emission_counts[idx_t_1, word_idx] += 1
                        
                        # Backoff to POS only
                        if pos in self.pos_to_idx:
                            pos_idx = self.pos_to_idx[pos]
                            pos_emission_counts[idx_t_1, pos_idx] += 1
                except KeyError:
                    pass
            
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
                    if i < len(obs_seq):
                        obs = obs_seq[i]
                        word, pos = obs
                        
                        # Full tuple emission
                        if obs in self.obs_to_idx:
                            obs_idx = self.obs_to_idx[obs]
                            emission_counts[idx_t, obs_idx] += 1
                        
                        # Backoff to word only
                        if word in self.word_to_idx:
                            word_idx = self.word_to_idx[word]
                            word_emission_counts[idx_t, word_idx] += 1
                        
                        # Backoff to POS only
                        if pos in self.pos_to_idx:
                            pos_idx = self.pos_to_idx[pos]
                            pos_emission_counts[idx_t, pos_idx] += 1
                except KeyError:
                    continue
        
        # Additional BIO-specific transition constraints - Increase likelihood of B-X → I-X → I-X patterns as before
        for i, state1 in enumerate(self.state_space):
            if state1.startswith("B-"):
                entity_type = state1[2:]  # Extract entity type
                i_state = f"I-{entity_type}"
                
                if i_state in self.state_to_idx:
                    i_idx = self.state_to_idx[i_state]
                    for j, state2 in enumerate(self.state_space): # Boost B-X → I-X → I-X
                        if state2 == i_state:
                            transition_counts[i, i_idx, i_idx] += 5  # Add stronger bias
        
        self.initial_probs = initial_counts / np.sum(initial_counts) # Normalize to get probabilities for initial state 
        
        self.initial_bigram_probs = np.zeros_like(initial_bigram_counts) # Normalize initial bigram probabilities at start of sequence
        for i in range(n_states): # Initial bigram probabilities
            if np.sum(initial_bigram_counts[i, :]) > 0:
                self.initial_bigram_probs[i, :] = initial_bigram_counts[i, :] / np.sum(initial_bigram_counts[i, :])
        
        self.transition_probs = np.zeros_like(transition_counts) # Normalize transition probabilities (for each pair of previous states)
        for i in range(n_states): # Raw transition counts of trigram (second order) in training data
            for j in range(n_states):
                if np.sum(transition_counts[i, j, :]) > 0:
                    self.transition_probs[i, j, :] = transition_counts[i, j, :] / np.sum(transition_counts[i, j, :])
        
        # Normalize emission probabilities
        self.emission_probs = emission_counts / np.sum(emission_counts, axis=1, keepdims=True)
        
        # Normalize backoff emission probabilities
        self.word_emission_probs = word_emission_counts / np.sum(word_emission_counts, axis=1, keepdims=True)
        self.pos_emission_probs = pos_emission_counts / np.sum(pos_emission_counts, axis=1, keepdims=True)
    
    def viterbi(self, observations: List[Tuple[str, str]]) -> List[str]:
        """
        Implement a modified Viterbi algorithm for second-order HMM decoding
        
        Parameters:
            observations (List[Tuple[str, str]]): Sequence of observations
            
        Returns:
            List[str]: Most likely sequence of states
        """
        if not observations:
            return []
        
        n_states = len(self.state_space)
        T = len(observations)
        
        if T == 1:
            # For single observation, just return the most likely state
            scores = np.zeros(n_states)
            for s in range(n_states):
                emission_prob = self.get_emission_prob(s, observations[0])
                scores[s] = np.log(self.initial_probs[s]) + np.log(emission_prob)
            
            best_state_idx = np.argmax(scores)
            return [self.state_space[best_state_idx]]
        
        # Initialize DP table for Viterbi
        V = np.zeros((T, n_states, n_states)) # MAX probability of being in states s1, s2 at times t-1, t
        
        # Initialize backpointers to store the previous state
        bp = np.zeros((T, n_states, n_states), dtype=int) # Previous state that led to s1, s2 at times t-1, t
        
        # Base case: t = 1
        for s0 in range(n_states):
            for s1 in range(n_states):
                prob_s0 = self.initial_probs[s0] # Initial probability of s0
                prob_s1_given_s0 = self.initial_bigram_probs[s0, s1] # Transition from s0 to s1
                # Emission for s0 with backoff
                emission_prob_s0 = self.get_emission_prob(s0, observations[0])
                # Emission for s1 with backoff
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
                        prev_prob = V[t-1, s0, s1] # Up to t-1 for (s0, s1)
                        trans_prob = self.transition_probs[s0, s1, s2] # Transition from (s0, s1) to s2
                        
                        # Total probability
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
    
    def save(self, output_file: str) -> None:
        """
        Save the trained second-order HMM model to a file
        
        Parameters:
            output_file (str): Path to save the model
        """
        model_data = {
            "state_space": self.state_space,
            "vocabulary": self.vocabulary,
            "words": self.words,
            "pos_tags": self.pos_tags,
            "state_to_idx": self.state_to_idx,
            "obs_to_idx": self.obs_to_idx,
            "word_to_idx": self.word_to_idx,
            "pos_to_idx": self.pos_to_idx,
            "initial_probs": self.initial_probs,
            "initial_bigram_probs": self.initial_bigram_probs,
            "transition_probs": self.transition_probs,
            "emission_probs": self.emission_probs,
            "word_emission_probs": self.word_emission_probs,
            "pos_emission_probs": self.pos_emission_probs,
            "smoothing": self.smoothing
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, input_file: str) -> 'HMMSecondOrder':
        """
        Load a trained second-order HMM model from a file
        
        Parameters:
            input_file (str): Path to the saved model
            
        Returns:
            HMMSecondOrder: Loaded second-order HMM model
        """
        with open(input_file, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            state_space=set(model_data["state_space"]),
            vocabulary=set(model_data["vocabulary"]),
            smoothing=model_data["smoothing"]
        )
        
        model.words = model_data["words"]
        model.pos_tags = model_data["pos_tags"]
        model.state_to_idx = model_data["state_to_idx"]
        model.obs_to_idx = model_data["obs_to_idx"]
        model.word_to_idx = model_data["word_to_idx"]
        model.pos_to_idx = model_data["pos_to_idx"]
        model.initial_probs = model_data["initial_probs"]
        model.initial_bigram_probs = model_data["initial_bigram_probs"]
        model.transition_probs = model_data["transition_probs"]
        model.emission_probs = model_data["emission_probs"]
        model.word_emission_probs = model_data["word_emission_probs"]
        model.pos_emission_probs = model_data["pos_emission_probs"]
        
        return model