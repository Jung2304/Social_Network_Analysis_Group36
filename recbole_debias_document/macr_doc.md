# File: macr.py
**Path:** `recbole_debias/model/debiased_recommender/macr.py`
**Module Type:** Model

## 1. Purpose & Overview
This file implements the MACR (Model-Agnostic Counterfactual Reasoning) model. This approach reduces popularity bias by explicitly modeling item popularity and user engagement as independent causal components, and dynamically subtracting their effect during inference using counterfactual reasoning.

## 2. Core Classes & Functions
- `MACR`: The main class implementing the counterfactual inference algorithm, extending `DebiasedRecommender`.
  - `__init__(self, config, dataset)`: Sets up user and item embeddings, and initializes MLPs to model user activity and item popularity.
  - `get_user_embedding(self, user)`: Recalls user ID embeddings.
  - `get_item_embedding(self, item)`: Recalls item ID embeddings.
  - `forward(self, user, item)`: Computes the base embedding score, the user-specific multiplier, and item-specific multiplier components.
  - `calculate_loss(self, interaction)`: Combines the multi-task BCELoss representing joint rating probability, user engagement probability, and item popularity probability.
  - `predict(self, interaction)`: Conducts inference via counterfactual estimation by adjusting the interaction formula with coefficient `c`.
  - `full_sort_predict(self, interaction)`: Efficiently performs counterfactual predictions across all items for full ranking.

## 3. Debiasing / Recommendation Logic (If Applicable)
MACR assumes three causal paths for an interaction: pure match (`yk`), generic user activeness (`yu`), and generic item popularity (`yi`). During training, it leverages MLPs to strictly optimize `yu` and `yi` against popularity biases using Binary Cross Entropy. During inference, MACR applies a counterfactual strategy by evaluating the direct path between user and item embeddings (`yk`) and diminishing the spurious popularity biases. The predictive score is formalized as `(yk - c) * yu * yi`, thereby offsetting the generic popularity effect with a baseline constant `c`.

## 4. Dependencies & Connections
- **Imports from core RecBole**: `torch`, `torch.nn`, `MLPLayers`, `BPRLoss`, `InputType`, `xavier_normal_initialization`.
- **Imports from framework**: Inherits from `DebiasedRecommender`.
