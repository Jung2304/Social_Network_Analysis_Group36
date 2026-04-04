# File: utils.py
**Path:** `recbole_debias/data/utils.py`
**Module Type:** Data Utility

## 1. Purpose & Overview
This utility file provides helper functions for dataset creation, data preparation, and dataloader retrieval specifically for the debiasing framework. it acts as a bridge between the configuration and the specialized data components.

## 2. Core Functions
- `create_dataset(config)`: Dynamically imports and instantiates the appropriate dataset class (e.g., `DebiasDataset`) based on the model and model type. Handles dataset serialization (loading/saving `.pth` files).
- `data_preparation(config, dataset)`: Manages the splitting of the dataset into training, validation, and test sets. It also creates the corresponding samplers and dataloaders.
- `get_dataloader(config, phase)`: Returns the correct dataloader class for a given phase ('train' or 'evaluation'). It includes a registration table for model-specific dataloaders like `DICE`.
- `_get_DICE_dataloader(config, phase)`: A specialized helper for retrieving the `DICEDataloader` during the training phase.
- `create_samplers(config, dataset, built_datasets)`: Constructs samplers (e.g., `DICESampler`, `Sampler`, `RepeatableSampler`) based on interaction distributions and negative sampling arguments.

## 3. Algorithmic Logic
### Dynamic Component Loading
The utility uses `importlib` and `getattr` to load classes at runtime. This allows the framework to scale with new models without hardcoding every dependency.
### Dataset Serialization
It checks for existing pre-processed dataset files in the `checkpoint_dir` to save time on repeated runs. It validates that the configuration of the saved dataset matches the current configuration before loading.
### Phase-Specific Logic
Differentiates between training (often requiring negative sampling or debiasing-specific logic) and evaluation (usually full-sort or negative-sample based evaluation).

## 4. Dependencies & Connections
- **RecBole Core**: Heavily uses `recbole.data.utils` and `recbole.sampler`.
- **Framework Components**: Imports from `recbole_debias.data.dataloader`, `recbole_debias.utils`, and `recbole_debias.sampler`.
