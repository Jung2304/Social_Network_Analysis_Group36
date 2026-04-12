import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Tuple

class InteractionDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.users = torch.LongTensor(df['user_id:token'].values)
        self.items = torch.LongTensor(df['item_id:token'].values)
        self.ratings = torch.FloatTensor(df['rating:float'].values)
        
    def __len__(self) -> int:
        return len(self.users)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[idx], self.items[idx], self.ratings[idx]

class DataFactory:
    """Helper to create DataLoaders and load interaction files."""
    
    @staticmethod
    def load_inter_file(file_path: str) -> pd.DataFrame:
        """Loads a RecBole .inter file as a DataFrame."""
        return pd.read_csv(file_path, sep='\t')

    @staticmethod
    def create_loader(df: pd.DataFrame, batch_size: int = 2048, shuffle: bool = True) -> DataLoader:
        dataset = InteractionDataset(df)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
