import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from models.base_model import BaseRecommendationModel
from utils.metrics import Evaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LightFMNetwork(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_genres: int, embedding_dim: int, use_hybrid: bool = True):
        super(LightFMNetwork, self).__init__()
        self.use_hybrid = use_hybrid
        self.user_embeddings = nn.Embedding(n_users + 1, embedding_dim)
        self.item_id_embeddings = nn.Embedding(n_items + 1, embedding_dim)
        if use_hybrid:
            self.genre_embeddings = nn.Embedding(n_genres + 1, embedding_dim)
        self.user_bias = nn.Embedding(n_users + 1, 1)
        self.item_bias = nn.Embedding(n_items + 1, 1)
        
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_id_embeddings.weight, std=0.01)
        if use_hybrid:
            nn.init.normal_(self.genre_embeddings.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor, item_genres_list: Optional[torch.Tensor] = None) -> torch.Tensor:
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_id_embeddings(item_ids)
        if self.use_hybrid and item_genres_list is not None:
            genre_embs = self.genre_embeddings(item_genres_list) 
            item_genre_sum = torch.sum(genre_embs, dim=1) if item_genres_list.dim() > 1 else genre_embs
            item_emb = item_emb + item_genre_sum
        u_bias = self.user_bias(user_ids).squeeze()
        i_bias = self.item_bias(item_ids).squeeze()
        dot = torch.sum(user_emb * item_emb, dim=-1)
        return dot + u_bias + i_bias

class LightFMModel(BaseRecommendationModel):
    def __init__(self, n_users: int, n_items: int, n_genres: int, embedding_dim: int = 16, use_hybrid: bool = True, loss_type: str = 'warp'):
        self.n_users = n_users
        self.n_items = n_items
        self.n_genres = n_genres
        self.use_hybrid = use_hybrid
        self.loss_type = loss_type
        self.network = LightFMNetwork(n_users, n_items, n_genres, embedding_dim, use_hybrid).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.01)

    def _bpr_loss(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

    def _warp_loss(self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor, genre_tensor: torch.Tensor, max_trials: int = 50) -> torch.Tensor:
        batch_size = user_ids.size(0)
        pos_scores = self.network(user_ids, pos_item_ids, genre_tensor[pos_item_ids])
        
        ranks = torch.zeros(batch_size, device=device)
        violator_scores = torch.zeros(batch_size, device=device)
        found_violator = torch.zeros(batch_size, device=device, dtype=torch.bool)

        for trial in range(1, max_trials + 1):
            if found_violator.all(): break
            neg_i = torch.randint(1, self.n_items + 1, (batch_size,), device=device)
            mask = ~found_violator
            if mask.any():
                neg_scores = self.network(user_ids[mask], neg_i[mask], genre_tensor[neg_i[mask]])
                violation = (neg_scores > pos_scores[mask] - 1)
                v_idx = torch.where(mask)[0][violation]
                found_violator[v_idx] = True
                ranks[v_idx] = trial
                violator_scores[v_idx] = neg_scores[violation]

        loss = torch.zeros(1, device=device)
        mask = found_violator
        if mask.any():
            L = torch.log(torch.floor(torch.tensor((self.n_items - 1), device=device) / ranks[mask]))
            loss = torch.mean(L * (1 + violator_scores[mask] - pos_scores[mask]))
        return loss

    def train_model(self, train_loader: Any, num_epochs: int, genre_tensor: Optional[torch.Tensor] = None, logger: Any = None) -> None:
        self.network.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            for u, i, r in train_loader:
                u, i = u.to(device), i.to(device)
                self.optimizer.zero_grad()
                if self.loss_type == 'bpr':
                    neg_i = torch.randint(1, self.n_items + 1, i.shape).to(device)
                    loss = self._bpr_loss(self.network(u, i, genre_tensor[i]), self.network(u, neg_i, genre_tensor[neg_i]))
                else:
                    loss = self._warp_loss(u, i, genre_tensor)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                msg = f"Epoch {epoch} | Loss: {epoch_loss/len(train_loader):.4f}"
                if logger:
                    logger.info(msg)
                else:
                    print(msg)

    def evaluate_model(self, test_df: pd.DataFrame, genre_tensor: torch.Tensor, k: int = 10) -> Dict[str, float]:
        self.network.eval()
        users = test_df['user_id:token'].unique()
        recalls, mrrs, ndcgs = [], [], []
        
        with torch.no_grad():
            for u in users:
                u_items = test_df[test_df['user_id:token'] == u]['item_id:token'].values
                u_tensor = torch.LongTensor([u] * self.n_items).to(device)
                item_tensor = torch.LongTensor(np.arange(1, self.n_items + 1)).to(device)
                scores = self.network(u_tensor, item_tensor, genre_tensor[item_tensor])
                _, top_indices = torch.topk(scores, k)
                top_items = (top_indices + 1).cpu().numpy()
                
                recalls.append(Evaluator.recall_at_k(top_items, u_items))
                mrrs.append(Evaluator.mrr_at_k(top_items, u_items))
                ndcgs.append(Evaluator.ndcg_at_k(top_items, u_items, k))

        return {
            "Recall@10": np.mean(recalls),
            "MRR@10": np.mean(mrrs),
            "NDCG@10": np.mean(ndcgs)
        }

    def save_model(self, path: str) -> None:
        torch.save(self.network.state_dict(), path)

    def load_model(self, path: str) -> None:
        self.network.load_state_dict(torch.load(path))
