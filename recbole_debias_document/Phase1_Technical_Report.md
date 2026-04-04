# Phase 1 Technical Report: Reproducing Recommendation Models on ml-100k

## 1. Executive Summary
This report summarizes the reproduction of Matrix Factorization (MF) and its debiased variants (MF-IPS, DICE) using the RecBole-Debias framework on the `ml-100k` dataset. The experiments were conducted on a CPU-only environment with optimized hyperparameters.

## 2. Methodology
- **Dataset:** `ml-100k` with `intervene_mask`.
- **Environment:** Python 3.12, RecBole 1.1.1, PyTorch 2.11.0+cpu.
- **Evaluation:** Full sorting on the intervened test set.
- **Hyperparameters:**
  - `learning_rate`: 0.005
  - `embedding_size`: 16
  - `batch_size`: 2048 (MF, DICE) / 256 (MF-IPS)

## 3. Results Comparison

| Model | Recall@10 | MRR@10 | NDCG@10 | Hit@10 | Precision@10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MF Baseline** | **0.0855** | 0.1915 | **0.0998** | 0.5271 | 0.0794 |
| **MF-IPS (User)** | 0.0750 | 0.1880 | 0.0923 | 0.5028 | 0.0750 |
| **DICE** | 0.0833 | **0.1985** | 0.0991 | **0.5393** | **0.0800** |

## 4. Observations

### Training Performance (CPU)
- **Baseline MF:** Extremely fast convergence (~0.1s per epoch). Stabilized around epoch 32.
- **MF-IPS (User):** Required smaller batch size (256) as specified. Convergence was fast but test results were slightly lower than the baseline, likely due to propensity over-correction on a small dataset.
- **DICE:** Showed the highest MRR and Hit@10, indicating superior ranking quality for the top-ranked items, despite a slightly lower total recall.

### Analysis of Debiasing
- **DICE Improvements:** The improvement in MRR (0.1985 vs 0.1915) suggests that DICE is more effective at pushing relevant items to the top of the list by specifically addressing popularity bias through its structural causal model and disentangled embeddings.
- **MF-IPS Performance:** The lower performance of IPS on this specific split might be attributed to high variance in propensity scores or the specific nature of the `intervene_mask` resampling, which might not perfectly align with IPS assumptions on this small scale.

## 5. Architectural Summary
The `RecBole-Debias` framework successfully extends RecBole by:
1.  **Custom Datasets:** Introducing `DebiasDataset` for propensity estimation.
2.  **Custom Samplers:** Implementing `DICEDataloader` for adaptive negative sampling.
3.  **Specialized Trainers:** Providing `DICETrainer` for managing complex loss functions (disentanglement + ranking).

**Status:** Phase 1 Reproducibility Completed.
