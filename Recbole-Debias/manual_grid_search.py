import sys
import os
import torch
import numpy as np

# Patching NumPy for compatibility with older RecBole
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool

# Patching Torch for compatibility with PyTorch 2.4+ weights_only=True default
orig_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return orig_load(*args, **kwargs)
torch.load = patched_load

# Add project root to sys.path
project_root = r'd:\UIT\HK6\MXH\DoAn\Recbole-Debias'
sys.path.insert(0, project_root)

from recbole_debias.quick_start import run_recbole_debias

lrs = [0.01, 0.005, 0.001]
embs = [16, 64]
results = []

for lr in lrs:
    for emb in embs:
        print(f"Running Grid Search: lr={lr}, emb={emb}")
        config_dict = {
            'learning_rate': lr,
            'embedding_size': emb,
            'epochs': 20,  # Short run for mini grid search
            'checkpoint_dir': 'saved/hyper',
            'state': 'ERROR' # Reduce logging
        }
        res = run_recbole_debias(
            model='MF',
            dataset='ml',
            config_file_list=[os.path.join(project_root, 'mf_baseline.yaml')],
            config_dict=config_dict,
            saved=False
        )
        test_res = res['test_result']
        results.append({
            'lr': lr,
            'emb': emb,
            'recall@10': test_res.get('recall@10', 0),
            'ndcg@10': test_res.get('ndcg@10', 0),
            'mrr@10': test_res.get('mrr@10', 0)
        })

import pandas as pd
df_results = pd.DataFrame(results)
print("\nGrid Search Results (20 epochs):")
print(df_results.to_markdown())
df_results.to_csv('manual_grid_search_results.csv', index=False)
