# File: pda.py
**Path:** `recbole_debias/model/debiased_recommender/pda.py`
**Module Type:** Model

## 1. Purpose & Overview
This file contains the PDA (Popularity-Bias Deconfounding and Adjusting) model. It leverages causal intervention (specifically adjustments for do-calculus) to adjust the propensity scores dynamically based on the global item interaction frequencies and BPR pairwise loss.

## 2. Core Classes & Functions
- `PDA`: The central model class for PDA methodology, derived from `DebiasedRecommender` using pairwise mechanics.
  - `__init__(self, config, dataset)`: Instantiates embeddings, loads standard prediction methods (`PDA` or standard), and registers propensity scores from the dataset.
  - `get_user_embedding(self, user)`: Captures the specified user embeddings.
  - `get_item_embedding(self, item)`: Captures the specified item embeddings.
  - `forward(self, user, item)`: Fetches and aligns user and item embeddings for scoring tasks.
  - `calculate_loss(self, interaction)`: Computes independent `ELU` adjusted propensities for positive and negative interactions and leverages them within a BPR Loss function alongside embedding regularization.
  - `predict(self, interaction)`: Calculates test preferences, injecting point-wise causal adjustment (multiplying by propensity score) if the `predict_method` implies PDA mode.
  - `full_sort_predict(self, interaction)`: Processes item wide predictions for candidate lists, again leveraging the propensity adjustments appropriately.

## 3. Debiasing / Recommendation Logic (If Applicable)
PDA approaches debiasing globally from an unobserved confounder assumption where popularity influences both item visibility and user decision-making. During `calculate_loss`, the algorithm transforms dot products via an Exponential Linear Unit (`ELU(x) + 1`) to ensure absolute positive values, multiplying them by item propensity weights before taking difference metrics against negative samples via BPR. At inference (`predict`), PDA executes causal intervention by directly re-weighting predictive estimates with the pre-learned popularity representations.

## 4. Dependencies & Connections
- **Imports from core RecBole**: `torch`, `torch.nn`, `BPRLoss`, `RegLoss`, `EmbLoss`, `InputType`, `xavier_normal_initialization`.
- **Imports from framework**: Depends on `DebiasedRecommender`. It expects the dataset structure to supply valid propensity mappings by accessing `dataset.estimate_pscore()`.
