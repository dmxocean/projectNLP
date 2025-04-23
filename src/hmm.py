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