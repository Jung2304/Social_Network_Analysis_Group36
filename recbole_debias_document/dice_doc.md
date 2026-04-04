# File: dice.py
**Path:** `recbole_debias/model/debiased_recommender/dice.py`
**Module Type:** Model

## 1. Purpose & Overview
This file implements the DICE (Disentangling User Interest and Conformity for Recommendation with Causal Embedding) model. It separates user embeddings into interest and conformity components to counteract popularity bias, allowing the system to disentangle true user preferences from general item popularity.

## 2. Core Classes & Functions
- `DICE`: The core model class inheriting from `DebiasedRecommender` to perform disentangled embedding learning.
  - `__init__(self, config, dataset)`: Initializes separate embedding spaces (for interest and popularity) for both users and items, as well as the discrepancy loss criterion.
  - `get_user_emb_total(self, user)`: Concatenates and returns the user's interest and popularity embeddings.
  - `get_item_emb_total(self, item)`: Concatenates and returns the item's interest and popularity embeddings.
  - `dcor(self, x, y)`: Calculates Distance Correlation (dCor) to measure non-linear independence between variables.
  - `bpr_loss(self, p_score, n_score)`: Computes standard Bayesian Personalized Ranking (BPR) loss.
  - `mask_bpr_loss(self, p_score, n_score, mask)`: Computes a masked BPR loss based on specific interaction causes.
  - `forward(self, user, item, factor)`: Returns the inner product of user and item embeddings based on the requested factor (interest, popularity, or total).
  - `calculate_loss(self, interaction)`: Formulates the multi-task loss function integrating BPR losses over interest and popularity dimensions, and subtracts the discrepancy penalty.
  - `adapt(self, decay)`: Decays the multi-task weights for interest and popularity components over epochs.
  - `predict(self, interaction)`: Outputs the interaction prediction using the combined total embeddings.
  - `full_sort_predict(self, interaction)`: Outputs vector scores over all items for candidate ranking.

## 3. Debiasing / Recommendation Logic (If Applicable)
DICE combats popularity bias by maintaining two distinct sets of embeddings: one capturing true user interest (`int`) and another capturing conformity/popularity (`pop`). Interactions are split based on a conformity mask. The loss is computed through modified Pairwise BPR logic (`mask_bpr_loss`) that independently evaluates the interest and conformity scores. Furthermore, a discrepancy loss (typically L1, L2, or distance correlation `dcor`) penalizes similarities between the `int` and `pop` embeddings, forcing them to learn independent representations.

## 4. Dependencies & Connections
- **Imports from core RecBole**: `torch`, `torch.nn`, `xavier_normal_initialization`, `InputType`.
- **Imports from framework**: Inherits from `DebiasedRecommender`. Expects specialized data loading through `DICESampler` and specialized training handling via `DICETrainer`.
