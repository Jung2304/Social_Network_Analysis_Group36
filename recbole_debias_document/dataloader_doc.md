# File: dataloader.py
**Path:** `recbole_debias/data/dataloader.py`
**Module Type:** Data

## 1. Purpose & Overview
This file manages the logic for negative sampling and data batching tailored for debiasing models. It provides specialized dataloaders, most notably for the DICE (De-Individuation Centrality Estimation) model, to handle interaction masks and specific sampling strategies.

## 2. Core Classes & Functions
- `DebiasDataloader`: An alias for the standard `TrainDataLoader` used for general debiasing models.
- `DICEDataloader(TrainDataLoader)`:
  - `__init__(self, config, dataset, sampler, shuffle=True)`: Sets up the masked field required for DICE.
  - `_neg_sampling(self, inter_feat)`: Implements dynamic negative sampling for DICE if configured.
  - `_neg_sample_by_pair_wise_sampling(self, inter_feat, neg_item_ids, mask)`: Augments interaction features with negative items and their corresponding masks.
  - `_neg_sample_by_point_wise_sampling(self, inter_feat, neg_item_ids)`: Standard pointwise sampling with label generation.
  - `get_model(self, model)`: Sets the model reference for dynamic sampling.

## 3. Algorithmic Logic
### DICE Dynamic Negative Sampling
The `DICEDataloader` supports a sophisticated negative sampling strategy:
1. **Candidate Selection**: Samples multiple negative candidates per positive interaction.
2. **Model-Guided Ranking**: Uses the current model state to predict scores for these candidates.
3. **Hard Negative Mining**: If `dynamic` sampling is enabled, it selects the negative item with the highest score (hard negative) to challenge the model during training.
4. **Masking**: Maintains a `mask_field` (typically an intervention mask) to distinguish between original and intervened interactions, which is vital for the DICE loss calculation.

## 4. Dependencies & Connections
- **RecBole Core**: Extends `TrainDataLoader` and uses `Interaction` and `InputType`.
- **Model Interaction**: Specifically coupled with the DICE model's training process to provide hard negatives and masks.
