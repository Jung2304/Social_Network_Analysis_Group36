# File: trainer.py
**Path:** `recbole_debias/trainer/trainer.py`
**Module Type:** Trainer

## 1. Purpose & Overview
The `trainer.py` file defines the training loops and evaluation logic for debiasing models. It introduces the `DICETrainer`, which extends the base RecBole `Trainer` to support adaptive hyperparameter tuning and specialized evaluation flows required for the DICE model.

## 2. Core Classes & Functions
- `DebiasTrainer`: An alias for the standard RecBole `Trainer`.
- `DICETrainer(DebiasTrainer)`:
  - `__init__(self, config, model)`: Initializes the trainer with additional settings like `decay`.
  - `fit(self, train_data, valid_data=None, ...)`: Overrides the standard training loop to incorporate DICE-specific logic like `eval_collector` data collection and dynamic model updates to the dataloader.
  - `adapt_hyperparams(self)`: Calls the model's `adapt` method, allowing for dynamic adjustments of regularization or loss weights during training.

## 3. Algorithmic Logic
### DICE Training Flow
1. **Collector Initialization**: The `eval_collector` gathers statistics from the training data before the loop starts.
2. **Model Sync with Dataloader**: If dynamic negative sampling is used, the trainer passes the model reference to the `train_data` (dataloader).
3. **Adaptive Mechanism**: At the end of each epoch, if `adaptive` is enabled, `adapt_hyperparams` is called. This is typically used in DICE to adjust the "disentanglement" penalty or other weights based on the current epoch/state.
4. **Early Stopping**: Standard early stopping logic is maintained to prevent overfitting on the validation set.

## 4. Dependencies & Connections
- **RecBole Core**: Extends `recbole.trainer.Trainer` and uses various utilities for logging, GPU usage, and early stopping.
- **Model Interaction**: Specifically designed to work with models that implement an `adapt` method (like DICE).
- **Data Interaction**: Interacts with `DICEDataloader` to provide the current model for hard negative mining.
