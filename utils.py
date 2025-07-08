# utils.py

import torch
import numpy as np
from pathlib import Path
from Normalization import MeshNormalizer

# Device: GPU se disponibile, altrimenti CPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def get_texture_map_from_color(mesh, color, H=224, W=224):
    """
    Restituisce un texture map uniforme a `color` per ogni pixel.
    Utile se qualche pipeline volesse ancora usare texture_map (ma non serve per PyTorch3D).
    """
    # Un solo batch, HxW, RGB
    texture = torch.ones(1, 3, H, W, device=device) * color.view(1,3,1,1)
    return texture


def get_face_attributes_from_color(mesh, color):
    """
    Funzione legacy: non più usata da PyTorch3D,
    ma lasciata per compatibilità se qualche vecchio modulo la richiama.
    Crea un attributo colore uniforme per ogni faccia.
    """
    F = mesh.faces.shape[0]
    # [1, F, 3, 3]: per ogni face, 3 canali * 3 componenti
    attr = torch.ones(1, F, 3, 3, device=device) * color.view(1,1,3,1)
    return attr


def color_mesh(pred_class: torch.Tensor, sampled_mesh, colors: torch.Tensor):
    """
    Aggiorna sampled_mesh.face_attributes per PyTorch3D, 
    prendendo il colore di ogni vertice e mappandolo sulle facce.
    
    pred_class: [V, C] probabilità per vertice
    sampled_mesh.faces: [F, 3] indici dei vertici per faccia
    colors: [C, 3] palette RGB
    """
    # 1) calcola colore continuo per vertice: [V,3]
    pred_rgb = segment2rgb(pred_class, colors)  # [V,3]
    # 2) per ciascuna faccia, raccogli i 3 colori dei suoi vertici:
    #    faces: [F,3], quindi pred_rgb[faces] ha shape [F,3,3]
    face_attrs = pred_rgb[sampled_mesh.faces].unsqueeze(0)  # [1, F, 3, 3]
    sampled_mesh.face_attributes = face_attrs.to(device)
    # 3) (legacy) normalizza la mesh, se serve
    MeshNormalizer(sampled_mesh)()


def segment2rgb(pred_class: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
    """
    Converte una matrice di probabilità [N,C] in colori continui [N,3].
    Ogni vertice ottiene il mix di `colors` pesato dalle probabilità.
    """
    # pred_class: [N, C], colors: [C,3]
    # risultato [N,3]
    return (pred_class @ colors.to(device))


# Se avevi in utils funzioni come standardize_mesh, normalize_mesh, add_vertices, add_vertices_with_labels,
# esse rimangono al loro posto e possono essere importate da qui se vuoi:
# from .some_module import standardize_mesh, normalize_mesh, add_vertices, add_vertices_with_labels
