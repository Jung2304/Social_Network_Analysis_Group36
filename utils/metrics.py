import numpy as np
from typing import Dict, List, Set

class Evaluator:
    """Calculates common recommendation metrics."""

    @staticmethod
    def recall_at_k(top_items: np.ndarray, ground_truth: np.ndarray) -> float:
        hit = np.intersect1d(top_items, ground_truth)
        return len(hit) / len(ground_truth) if len(ground_truth) > 0 else 0

    @staticmethod
    def mrr_at_k(top_items: np.ndarray, ground_truth: np.ndarray) -> float:
        for i, item in enumerate(top_items):
            if item in ground_truth:
                return 1 / (i + 1)
        return 0

    @staticmethod
    def ndcg_at_k(top_items: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
        dcg = 0
        for i, item in enumerate(top_items):
            if item in ground_truth:
                dcg += 1 / np.log2(i + 2)
        
        idcg = sum([1 / np.log2(i + 2) for i in range(min(len(ground_truth), k))])
        return dcg / idcg if idcg > 0 else 0
