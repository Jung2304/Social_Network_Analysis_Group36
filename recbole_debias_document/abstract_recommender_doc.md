# File: abstract_recommender.py
**Path:** `recbole_debias/model/abstract_recommender.py`
**Module Type:** Model (Abstract Base Class)

## 1. Purpose & Overview
This file defines the abstract base class `DebiasedRecommender` for all debiased recommendation models within the framework. It sets up the foundational model structure, including dataset interpretation, parameters initialization, and device configuration for all debiased models.

## 2. Core Classes & Functions
- `DebiasedRecommender`: Abstract base class extending RecBole's `AbstractRecommender` configured specifically for debiasing tasks.
  - `__init__(self, config, dataset)`: Initializes dataset fields (users, items, negative items) and loads parameters like the computation device.

## 3. Debiasing / Recommendation Logic (If Applicable)
This file primarily serves as an abstract utility. It declares the model type as `ModelType.DEBIAS` and prepares the basic variables (like generic ID fields and counts) required by downstream debiasing algorithms. It does not implement any specific debiasing logic itself.

## 4. Dependencies & Connections
- **Imports from core RecBole**: `recbole.model.abstract_recommender.AbstractRecommender`.
- **Imports from framework utils**: `recbole_debias.utils.ModelType`.
- **Used by**: Extended by almost all specific debiased models (like CausE, DICE, MACR, etc.).
