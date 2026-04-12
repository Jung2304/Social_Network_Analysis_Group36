from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import torch

class BaseRecommendationModel(ABC):
    """
    Abstract Base Class for all recommendation models.
    Ensures a consistent interface for training, evaluation, and prediction.
    """

    @abstractmethod
    def train_model(self, train_loader: Any, num_epochs: int) -> None:
        """
        Train the model using the provided data loader.
        
        Args:
            train_loader: The data loader for training interactions.
            num_epochs: Number of training epochs.
        """
        pass

    @abstractmethod
    def evaluate_model(self, test_data: Any, k: int = 10) -> Dict[str, float]:
        """
        Evaluate the model metrics (Recall, MRR, NDCG).
        
        Args:
            test_data: The data for evaluation.
            k: Top-K recommendation list size.
            
        Returns:
            A dictionary containing metric names and their values.
        """
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save model weights to a file."""
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load model weights from a file."""
        pass
