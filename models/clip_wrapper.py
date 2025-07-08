# src/models/clip_wrapper.py

import clip
import torch
from utils import device

def get_clip_model(model_name: str):
    """
    Load CLIP model + preprocess from open-clip.
    """
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess

def encode_text(model, text: str) -> torch.Tensor:
    """
    Encode a single prompt into a normalized CLIP text embedding: [1, D].
    """
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        t_emb = model.encode_text(tokens)
        t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
    return t_emb

def encode_image(model, preprocess, images: torch.Tensor) -> torch.Tensor:
    """
    Encode a batch of images into CLIP embeddings.
    images: [B,3,H,W] in [0,1]. We normalize using CLIP's stats.
    Returns: [B, D] normalized embeddings.
    """
    # CLIP’s recommended normalization: estrai il Normalize dal preprocess
    # preprocess è un Compose; l'ultimo elemento è Normalize(mean, std)
    # Cerchiamo torchvision.transforms.Normalize nella pipeline
    from torchvision.transforms import Normalize
    normalize: Normalize = None
    for t in preprocess.transforms:
        if isinstance(t, Normalize):
            normalize = t
            break
    if normalize is None:
        raise RuntimeError("Cannot find Normalize transform in CLIP preprocess pipeline")
    # Se ci sono più di 3 canali (es. RGBA), teniamo solo i primi 3 (RGB)
    if images.shape[1] != len(normalize.mean):
        images = images[:, :len(normalize.mean), :, :]
    imgs = normalize(images)
    
    with torch.no_grad():
        i_emb = model.encode_image(imgs)
        i_emb = i_emb / i_emb.norm(dim=-1, keepdim=True)
    return i_emb
