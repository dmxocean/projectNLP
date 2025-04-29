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