import os
import yaml
from models.lightfm_model import LightFMModel
from utils.data_loader import DataFactory
from utils.path_manager import PathManager
from utils.genre_utils import get_genre_mapping, GENRES
from utils.logger import Logger
import torch
import numpy as np
import os

def train_and_eval_lightfm(config_name: str):
    # Setup Logger
    log_dir = PathManager.get_log_path("LIGHTFM")
    logger = Logger.get_logger("LightFM_Exp", log_dir)
    
    logger.info(f"--- Starting Experiment with config: {config_name} ---")

    # Load config
    config_path = PathManager.get_config_path(config_name)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Paths
    train_path = PathManager.get_data_path(config['train_path'])
    test_path = PathManager.get_data_path(config['test_path'])

    # Data
    train_df = DataFactory.load_inter_file(train_path)
    test_df = DataFactory.load_inter_file(test_path)
    
    n_users = max(train_df['user_id:token'].max(), test_df['user_id:token'].max())
    n_items = max(train_df['item_id:token'].max(), test_df['item_id:token'].max())

    # Features (Genres)
    item_genres = get_genre_mapping()
    genre_to_idx = {g: i+1 for i, g in enumerate(GENRES)}
    genre_matrix = np.zeros((n_items + 1, 3), dtype=int)
    for i in range(1, n_items + 1):
        gs = item_genres.get(str(i), [])
        for j, g in enumerate(gs[:3]): 
            genre_matrix[i, j] = genre_to_idx.get(g, 0)
    genre_tensor = torch.LongTensor(genre_matrix).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Model
    model = LightFMModel(
        n_users=n_users, 
        n_items=n_items, 
        n_genres=len(GENRES),
        embedding_dim=config['embedding_dim'],
        use_hybrid=config['use_hybrid'],
        loss_type=config['loss_type']
    )

    # Train
    train_loader = DataFactory.create_loader(train_df, batch_size=config['batch_size'])
    logger.info(f"Training for {config['epochs']} epochs...")
    model.train_model(train_loader, num_epochs=config['epochs'], genre_tensor=genre_tensor, logger=logger)

    # Eval
    metrics = model.evaluate_model(test_df, genre_tensor=genre_tensor)
    logger.info(f"Final Metrics for {config_name}:")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")
    
    logger.info(f"--- Finished Experiment: {config_name} ---\n")

if __name__ == "__main__":
    # Example execution for LightFM Hybrid
    train_and_eval_lightfm("lightfm_hybrid.yaml")
