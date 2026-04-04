# File: rel_mf.py
**Path:** `recbole_debias/model/debiased_recommender/rel_mf.py`
**Module Type:** Model

## 1. Purpose & Overview
This file establishes the REL-MF (Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback) architecture. It proposes debiased models through specifically designed unbiased BCELoss approaches handling missing-not-at-random (MNAR) factors common in implicit recommendation behaviors.

## 2. Core Classes & Functions
- `REL_MF`: The primary formulation inheriting `DebiasedRecommender`.
  - `__init__(self, config, dataset)`: Defines base embeddings along with auxiliary components (user, item, and global bias terms). Captures propensity scores and establishes the unbiased BCE parameters.
  - `get_user_embedding(self, user)`: Extracts target user vectors.
  - `get_item_embedding(self, item)`: Extracts target item vectors.
  - `forward(self, user, item)`: Employs a classical formulation adding user bias, item bias, and a global constant to the core embedding dot product, finalizing with a Sigmoid mapping.
  - `calculate_loss(self, interaction)`: Employs either standard BCELoss coupled with strict IPS multipliers, or employs the customized `Unbiased_BCELoss` class depending on the `loss_choice` configuration. Incorporates regularization.
  - `predict(self, interaction)`: Infers scores for specific user-item queries.
  - `full_sort_predict(self, interaction)`: Formats evaluation matrices ignoring the item and user explicit bias terms for optimized wide ranking estimation.
- `Unbiased_BCELoss`: A custom loss Module.
  - `__init__(self, reduction='mean')`: Sets the tensor reduction methodology.
  - `forward(self, prediction, label, weight)`: Carries out Eq.(9) from the reference paper, implementing an adjusted BCE formula specifically formulated for MNAR bias characteristics relying heavily on sample weights.

## 3. Debiasing / Recommendation Logic (If Applicable)
The debiasing heavily features Inverse Propensity framework applied to BCE instead of MSE. The customized `Unbiased_BCELoss` divides label contributions by the estimated propensity weight `weight`, while the negative sampling term is discounted equivalently: `(1 - label / weight) * log(1 - prediction)`. This corrects biased implicit feedback datasets where absence of an interaction doesn't purely denote negative preference but could be due to item obscurity.

## 4. Dependencies & Connections
- **Imports from core RecBole**: `torch`, `torch.nn`, `InputType`, `RegLoss`, `EmbLoss`, `xavier_normal_initialization`.
- **Imports from framework**: Inherits from `DebiasedRecommender`. Extracts propensity structures via `dataset.estimate_pscore()`.
