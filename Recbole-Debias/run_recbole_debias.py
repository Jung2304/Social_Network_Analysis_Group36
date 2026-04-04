# @Time   : 2022/3/22
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import argparse
import numpy as np
import torch

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

from recbole_debias.quick_start import run_recbole_debias

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MF', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', '-c', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole_debias(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
