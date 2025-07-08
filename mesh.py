# mesh.py

import torch
import numpy as np
from pytorch3d.io import load_obj
from utils import device
import PIL

class Mesh:
    """
    Mesh object leggero, carica .obj con PyTorch3D e mantiene:
      - vertices:         torch.Tensor [V,3]
      - faces:            torch.LongTensor [F,3]
      - face_attributes:  torch.Tensor [1,F,3,3] per il renderer
    Supporta set_mesh_color e export OBJ/PLY.
    """

    def __init__(self, obj_path: str, color: torch.Tensor = torch.tensor([0.0, 0.0, 1.0])):
        # 1) Carica con PyTorch3D
        verts, faces, aux = load_obj(obj_path)
        self.vertices = verts.to(device)                      # [V,3]
        self.faces    = faces.verts_idx.to(device)            # [F,3]

        # 2) Ignoro normals/uvs per semplicità

        # 3) Imposta colore uniforme
        self.set_mesh_color(color)

    def set_mesh_color(self, color: torch.Tensor):
        """
        Crea face_attributes uniformi: shape [1, F, 3, 3].
        color: [3] in [0,1]
        """
        F = self.faces.shape[0]
        # crea tensor [1,F,3,3] tutto a color
        attrs = color.view(1,1,3).expand(1, F, 3)        # [1,F,3]
        self.face_attributes = attrs.unsqueeze(-1).expand(1, F, 3, 3).to(device)

    def export(self, file: str, extension: str = "ply", color: torch.Tensor = None):
        """
        Esporta in PLY ASCII con colore per vertice se fornito,
        altrimenti in OBJ senza colori aggiuntivi.
        color: [V,3] in [0,255] interi, se extension=="ply".
        """
        verts = self.vertices.cpu().numpy()
        faces = self.faces.cpu().numpy()

        if extension == "obj":
            with open(file, "w") as f:
                for v in verts:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for face in faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        elif extension == "ply":
            # default bianco se non passo color
            if color is None:
                color_arr = np.ones_like(verts, dtype=np.uint8) * 255
            else:
                color_arr = np.clip(color.cpu().numpy(), 0, 255).astype(np.uint8)

            with open(file, "w") as f:
                f.write("ply\nformat ascii 1.0\n")
                f.write(f"element vertex {len(verts)}\n")
                f.write("property float x\nproperty float y\nproperty float z\n")
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                f.write(f"element face {len(faces)}\n")
                f.write("property list uchar int vertex_indices\n")
                f.write("end_header\n")
                for (x,y,z), (r,g,b) in zip(verts, color_arr):
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
                for face in faces:
                    f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        else:
            raise ValueError(f"Unknown extension '{extension}'")
