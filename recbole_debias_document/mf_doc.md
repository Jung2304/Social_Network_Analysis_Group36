# File: mf.py
**Path:** `recbole_debias/model/debiased_recommender/mf.py`
**Module Type:** Model

## 1. Purpose & Overview
This file provides a standard implementation of Matrix Factorization (MF) adapted for the debiasing framework. It serves as a foundational baseline model, operating primarily through dot-product scoring of user and item embeddings.

## 2. Core Classes & Functions
- `MF`: The basic Matrix Factorization class extending `DebiasedRecommender`.
  - `__init__(self, config, dataset)`: Prepares fundamental user and item embeddings and MSELoss.
  - `get_user_embedding(self, user)`: Extracts the given user's embeddings.
  - `get_item_embedding(self, item)`: Extracts the given item's embeddings.
  - `forward(self, user, item)`: Calculates the simple dot product between target user and target item embeddings.
  - `calculate_loss(self, interaction)`: Computes the basic Mean Squared Error (MSE) loss between the predicted score and the actual label.
  - `predict(self, interaction)`: Yields the predicted association score for given user-item pairs.
  - `full_sort_predict(self, interaction)`: Produces full recommendation vectors by multiplying user embeddings with the entire item embedding matrix.

## 3. Debiasing / Recommendation Logic (If Applicable)
While this model exists under the `debiased_recommender` package, it implements standard biased Matrix Factorization. It does not actively remove biases but acts as a vanilla architecture often used alongside Inverse Propensity Scoring (like in `MF_IPS`) or simply as a reference baseline metric against debiasing variants. It utilizes `MSELoss` for Pointwise score approximations.

## 4. Dependencies & Connections
- **Imports from core RecBole**: `torch`, `torch.nn`, `InputType`, `xavier_normal_initialization`. (Note: `BPRLoss` is imported but not utilized here as it trains pointwise).
- **Imports from framework**: Inherits from `DebiasedRecommender`.
