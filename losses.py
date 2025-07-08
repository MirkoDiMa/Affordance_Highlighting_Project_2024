# src/losses.py

import torch.nn.functional as F
import torch

def clip_loss(text_emb: torch.Tensor,  # [1, D]
              img_embs: torch.Tensor   # [B, D]
             ) -> torch.Tensor:
    """
    Aggregate image embeddings by averaging, then compute
    negative cosine similarity against the text embedding.
    """
    # normalize image embeddings
    img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
    avg_emb = img_embs.mean(dim=0, keepdim=True)
    avg_emb = avg_emb / avg_emb.norm(dim=-1, keepdim=True)
    cos_sim = F.cosine_similarity(avg_emb, text_emb, dim=-1)
    return -cos_sim.mean()
