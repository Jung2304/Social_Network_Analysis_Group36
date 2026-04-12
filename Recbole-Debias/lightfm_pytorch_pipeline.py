import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import time
from genre_utils import get_genre_mapping, GENRES

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InteractionDataset(Dataset):
    def __init__(self, df):
        self.users = torch.LongTensor(df['user_id:token'].values)
        self.items = torch.LongTensor(df['item_id:token'].values)
        self.ratings = torch.FloatTensor(df['rating:float'].values)
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

class LightFM_PyTorch(nn.Module):
    def __init__(self, n_users, n_items, n_genres, embedding_dim, use_hybrid=True):
        super(LightFM_PyTorch, self).__init__()
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

    def forward(self, user_ids, item_ids, item_genres_list=None):
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

def bpr_loss(pos_scores, neg_scores):
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

def warp_loss_optimized(user_ids, pos_item_ids, model, n_items, genre_tensor, max_trials=50):
    batch_size = user_ids.size(0)
    pos_scores = model(user_ids, pos_item_ids, genre_tensor[pos_item_ids])
    
    loss = torch.zeros(1, device=device)
    # We sample a block of negatives to vectorize
    num_samples = max_trials
    neg_items = torch.randint(1, n_items + 1, (batch_size, num_samples), device=device)
    
    for i in range(num_samples):
        neg_i = neg_items[:, i]
        neg_scores = model(user_ids, neg_i, genre_tensor[neg_i])
        
        # Violators: neg_score > pos_score - 1
        violators = (neg_scores > pos_scores - 1).float()
        
        # We only want to count the first violator for each user in the batch
        # This is a bit tricky to vectorize perfectly without a loop over trials
        # but even this "check all trials" is faster than the Python while loop.
        
        # For this audit, we will use a simpler approximation of WARP 
        # which is to weight by the rank estimated by how many trials it took.
    
    # Actually, a real WARP needs to "break" after the first violator.
    # Let's do a loop over trials but keep it efficient.
    
    ranks = torch.zeros(batch_size, device=device)
    violator_scores = torch.zeros(batch_size, device=device)
    found_violator = torch.zeros(batch_size, device=device, dtype=torch.bool)

    for trial in range(1, max_trials + 1):
        if found_violator.all(): break
        
        neg_i = torch.randint(1, n_items + 1, (batch_size,), device=device)
        mask = ~found_violator
        
        if mask.any():
            neg_scores = model(user_ids[mask], neg_i[mask], genre_tensor[neg_i[mask]])
            violation = (neg_scores > pos_scores[mask] - 1)
            
            # Update violators
            v_idx = torch.where(mask)[0][violation]
            found_violator[v_idx] = True
            ranks[v_idx] = trial
            violator_scores[v_idx] = neg_scores[violation]

    # Calculate loss for those who found a violator
    mask = found_violator
    if mask.any():
        L = torch.log(torch.floor(torch.tensor((n_items - 1), device=device) / ranks[mask]))
        loss = torch.mean(L * (1 + violator_scores[mask] - pos_scores[mask]))
        
    return loss

def evaluate(model, test_df, train_item_counts, n_items, genre_tensor, topk=10):
    model.eval()
    users = test_df['user_id:token'].unique()
    recalls, mrrs, ndcgs = [], [], []
    cold_start_items = train_item_counts[train_item_counts < 5].index.tolist()
    cold_start_recalls = []

    with torch.no_grad():
        for u in users:
            u_items = test_df[test_df['user_id:token'] == u]['item_id:token'].values
            if len(u_items) == 0: continue
            u_tensor = torch.LongTensor([u] * n_items).to(device)
            item_tensor = torch.LongTensor(np.arange(1, n_items + 1)).to(device)
            scores = model(u_tensor, item_tensor, genre_tensor[item_tensor])
            _, top_indices = torch.topk(scores, topk)
            top_items = (top_indices + 1).cpu().numpy()
            
            hit = np.intersect1d(top_items, u_items)
            recalls.append(len(hit) / len(u_items))
            # MRR/NDCG
            mrr, dcg = 0, 0
            for k, p in enumerate(top_items):
                if p in u_items:
                    if mrr == 0: mrr = 1 / (k + 1)
                    dcg += 1 / np.log2(k + 2)
            mrrs.append(mrr)
            idcg = sum([1 / np.log2(k + 2) for k in range(min(len(u_items), topk))])
            ndcgs.append(dcg / idcg if idcg > 0 else 0)
            
            cs_u_items = [it for it in u_items if it in cold_start_items]
            if len(cs_u_items) > 0:
                cs_hit = np.intersect1d(top_items, cs_u_items)
                cold_start_recalls.append(len(cs_hit) / len(cs_u_items))

    return np.mean(recalls), np.mean(mrrs), np.mean(ndcgs), np.mean(cold_start_recalls) if cold_start_recalls else 0

def run_experiment(use_hybrid=True, loss_type='warp', num_epochs=50):
    print(f"\n--- Starting Experiment: Hybrid={use_hybrid}, Loss={loss_type}, Epochs={num_epochs} ---")
    train_df = pd.read_csv('dataset/ml/ml.train.inter', sep='\t')
    test_df = pd.read_csv('dataset/ml/ml.test.inter', sep='\t')
    n_users = max(train_df['user_id:token'].max(), test_df['user_id:token'].max())
    n_items = max(train_df['item_id:token'].max(), test_df['item_id:token'].max())
    item_genres = get_genre_mapping()
    genre_to_idx = {g: i+1 for i, g in enumerate(GENRES)}
    genre_matrix = np.zeros((n_items + 1, 3), dtype=int)
    for i in range(1, n_items + 1):
        gs = item_genres.get(str(i), [])
        for j, g in enumerate(gs[:3]): genre_matrix[i, j] = genre_to_idx.get(g, 0)
    genre_tensor = torch.LongTensor(genre_matrix).to(device)
    
    model = LightFM_PyTorch(n_users, n_items, len(GENRES), 16, use_hybrid=use_hybrid).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train_loader = DataLoader(InteractionDataset(train_df), batch_size=2048, shuffle=True)
    train_item_counts = train_df['item_id:token'].value_counts()
    
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        for u, i, r in train_loader:
            u, i = u.to(device), i.to(device)
            optimizer.zero_grad()
            if loss_type == 'bpr':
                neg_i = torch.randint(1, n_items + 1, i.shape).to(device)
                loss = bpr_loss(model(u, i, genre_tensor[i]), model(u, neg_i, genre_tensor[neg_i]))
            else:
                loss = warp_loss_optimized(u, i, model, n_items, genre_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | Time: {time.time()-epoch_start:.2f}s")
            
    total_time = time.time() - start_time
    metrics = evaluate(model, test_df, train_item_counts, n_items, genre_tensor)
    print(f"Results: R@10: {metrics[0]:.4f}, MRR@10: {metrics[1]:.4f}, NDCG@10: {metrics[2]:.4f}, CS_R: {metrics[3]:.4f}")
    print(f"Total Time: {total_time:.2f}s, Avg: {total_time/num_epochs:.2f}s/epoch")
    return metrics, total_time/num_epochs

if __name__ == "__main__":
    results = {}
    
    print("\n[SCENARIO 1] ID-only + BPR")
    results['S1'] = run_experiment(use_hybrid=False, loss_type='bpr', num_epochs=50)
    
    print("\n[SCENARIO 2] Hybrid + BPR")
    results['S2'] = run_experiment(use_hybrid=True, loss_type='bpr', num_epochs=50)

    print("\n[SCENARIO 3] ID-only + WARP")
    results['S3'] = run_experiment(use_hybrid=False, loss_type='warp', num_epochs=50)

    print("\n[SCENARIO 4] Hybrid + WARP")
    results['S4'] = run_experiment(use_hybrid=True, loss_type='warp', num_epochs=50)
    
    print("\n--- FINAL CONSOLIDATED RESULTS ---")
    for k, v in results.items():
        m, t = v
        print(f"{k}: R@10={m[0]:.4f}, MRR@10={m[1]:.4f}, NDCG@10={m[2]:.4f}, CS_R={m[3]:.4f}, Time={t:.4f}")
