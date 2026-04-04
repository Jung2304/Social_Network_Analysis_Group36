# File: mf_ips.py
**Path:** `recbole_debias/model/debiased_recommender/mf_ips.py`
**Module Type:** Model

## 1. Purpose & Overview
This file implements the MF-IPS (Matrix Factorization with Inverse Propensity Scoring) model. It aims to eliminate selection bias in explicit or implicit feedback by re-weighting the traditional loss using pre-calculated propensity scores for user or item interactions.

## 2. Core Classes & Functions
- `MF_IPS`: A weighted Matrix Factorization model inheriting from `DebiasedRecommender`.
  - `__init__(self, config, dataset)`: Prepares standard embeddings, establishes an unreduced MSE loss, and loads dataset propensity scores.
  - `get_user_embedding(self, user)`: Returns user embeddings.
  - `get_item_embedding(self, item)`: Returns item embeddings.
  - `forward(self, user, item)`: Calculates the standard inner product of user and item embeddings.
  - `calculate_loss(self, interaction)`: Combines the dot product outputs with the label using MSE, strictly applying an Inverse Propensity Score (IPS) multiplier before reducing the loss.
  - `predict(self, interaction)`: Produces user-item interaction scores.
  - `full_sort_predict(self, interaction)`: Emits all item scores for a user to allow broad ranking operations.

## 3. Debiasing / Recommendation Logic (If Applicable)
To debias interactions, the model applies Inverse Propensity Scoring (IPS). For any given interaction, the prediction error term (`MSE(y_hat, y)`) is divided by `propensity_score`, mitigating the fact that highly popular items (high propensity score) represent overabundant training signals. The dataset handles generating these weights (typically User Propensity, Item Propensity, or a Naive Bayes uniform PS), and the algorithm inversely applies them during gradient calculation.

## 4. Dependencies & Connections
- **Imports from core RecBole**: `torch`, `torch.nn`, `InputType`, `xavier_normal_initialization`.
- **Imports from framework**: Inherits from `DebiasedRecommender`. It heavily relies on dataset implementations invoking `dataset.estimate_pscore()` to retrieve propensity weights.
