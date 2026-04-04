# File: dataset.py
**Path:** `recbole_debias/data/dataset.py`
**Module Type:** Data

## 1. Purpose & Overview
The `dataset.py` file defines the `DebiasDataset` class, which extends the base RecBole `Dataset`. Its primary role is to handle data loading and provide utility for estimating propensity scores, which are crucial for debiasing techniques like Inverse Propensity Scoring (IPS).

## 2. Core Classes & Functions
- `DebiasDataset(Dataset)`: 
  - `__init__(self, config)`: Initializes the dataset with configuration parameters, specifically extracting `ITEM_ID_FIELD`, `USER_ID_FIELD`, and propensity-related settings (`pscore_method`, `eta`).
  - `estimate_pscore(self)`: Computes the propensity scores based on the specified method.

## 3. Algorithmic Logic
### Propensity Score Estimation (`estimate_pscore`)
The method calculates the popularity of items, users, or ratings to determine the propensity score:
1. **Method Selection**:
   - `item`: Calculates item popularity (number of interactions per item).
   - `user`: Calculates user activity (number of interactions per user).
   - `nb`: Calculates frequency of ratings (useful for explicit feedback).
2. **Frequency Count**: Uses `torch.unique` with `return_counts=True` on the specified column of the training data.
3. **Normalization & Power Law Transformation**: 
   - The counts are normalized by dividing by the maximum count.
   - The result is raised to the power of `eta` (a hyperparameter controlling the strength of the debiasing).
   - Formula: $P(x) = (\frac{count(x)}{\max(count)})^{\eta}$

## 4. Dependencies & Connections
- **RecBole Core**: Inherits from `recbole.data.dataset.Dataset`.
- **Debiasing Link**: The estimated propensity scores are typically consumed by models like `MF_IPS` in `recbole_debias/model/debiased_recommender/mf_ips.py`.
