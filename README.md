## Assumption
This README assumes you already have the phase4_outputs and phase5_outputs folders.

## 1) Prerequisites
- Python environment with: numpy, scipy, pandas, lightfm, implicit, torch.

## 2) Run the Five Models
Each model is implemented in a notebook. Open the notebook and run all cells.

| Model | Notebook | Required files | Optional files |
|---|---|---|---|
| Popularity | popularity_baseline.ipynb | phase5_outputs/train_interactions.npz, phase5_outputs/test_interactions.npz | phase4_outputs/book2id.json, phase4_outputs/user2id.json |
| Pure LightFM (WARP) | train_pure_cf.ipynb | phase5_outputs/train_interactions.npz, phase5_outputs/test_interactions.npz | phase5_outputs/train_weights.npz |
| Hybrid LightFM | LightFM.ipynb | phase5_outputs/train_interactions.npz, phase5_outputs/test_interactions.npz, phase4_outputs/item_features.npz | phase5_outputs/train_weights.npz |
| BPR-MF (implicit) | bpr_mf_baseline.ipynb | phase5_outputs/train_interactions.npz, phase5_outputs/test_interactions.npz | phase5_outputs/train_weights.npz |
| LightGCN | lightgcn_baseline.ipynb | phase5_outputs/train_interactions.npz, phase5_outputs/test_interactions.npz | phase4_outputs/book2id.json, phase4_outputs/user2id.json |

## 3) Evaluation
All notebooks use the same evaluation protocol (Precision@K, Recall@K, NDCG@K) with train items masked at ranking time. Metrics are reported at K=10.

## 4) Notes
- LightGCN can be run locally or on GPU in Google Colab.
- If file paths differ, update the notebook variables to point to your data locations.
