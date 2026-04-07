import torch
import torch.nn.functional as F
import numpy as np

def audit_embeddings(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']
    
    # Extract embeddings
    users_int = state_dict['users_int.weight']
    users_pop = state_dict['users_pop.weight']
    items_int = state_dict['items_int.weight']
    items_pop = state_dict['items_pop.weight']
    
    # Calculate cosine similarity
    user_sim = F.cosine_similarity(users_int, users_pop, dim=1).mean().item()
    item_sim = F.cosine_similarity(items_int, items_pop, dim=1).mean().item()
    
    print(f"User Similarity (Interest vs Pro): {user_sim:.4f}")
    print(f"Item Similarity (Interest vs Pro): {item_sim:.4f}")

if __name__ == "__main__":
    audit_embeddings(r'd:\UIT\HK6\MXH\DoAn\Recbole-Debias\saved\DICE-Apr-04-2026_20-54-16.pth')
