# mesh.py

import torch
import numpy as np
from pytorch3d.io import load_obj
from utils import (
    device,
    get_texture_map_from_color,
    get_face_attributes_from_color,
    standardize_mesh,
    normalize_mesh,
    add_vertices,
    add_vertices_with_labels
)
import copy
import PIL

class Mesh:
    """
    Mesh object compatibile con l’API esistente:
    - vertices:       torch.Tensor [V,3]
    - faces:          torch.LongTensor [F,3]
    - vertex_normals: optional
    - face_normals:   optional
    - texture_map, face_attributes: per il renderer
    Supporta normalize, standardize, update_vertex, divide, divide_with_labels, export.
    """

    def __init__(self, obj_path: str, color: torch.Tensor = torch.tensor([0.0, 0.0, 1.0])):
        # Carica vertici e facce da .obj
        verts, faces, aux = load_obj(obj_path)
        self.vertices = verts.to(device)                       # [V,3]
        self.faces    = faces.verts_idx.to(device)             # [F,3]

        # Normali optional (se presenti nel file OBJ)
        self.vertex_normals = None
        self.face_normals   = None
        if aux.normals is not None:
            self.vertex_normals = aux.normals.to(device)
            self.vertex_normals = torch.nn.functional.normalize(self.vertex_normals, dim=1)
        if aux.vertex_uvs is not None:
            # PyTorch3D non fornisce face_normals direttamente, saltiamo
            pass

        # Imposta colore uniforme
        self.set_mesh_color(color)

    def standardize_mesh(self, inplace: bool = False):
        mesh = self if inplace else copy.deepcopy(self)
        return standardize_mesh(mesh)

    def normalize_mesh(self, inplace: bool = False):
        mesh = self if inplace else copy.deepcopy(self)
        return normalize_mesh(mesh)

    def update_vertex(self, verts: torch.Tensor, inplace: bool = False):
        mesh = self if inplace else copy.deepcopy(self)
        mesh.vertices = verts.to(device)
        return mesh

    def set_mesh_color(self, color: torch.Tensor):
        """
        Genera:
         - self.texture_map: [1,3,H,W] fornito da utils.get_texture_map_from_color
         - self.face_attributes: [1,F,3,3] da utils.get_face_attributes_from_color
        """
        self.texture_map = get_texture_map_from_color(self, color)
        self.face_attributes = get_face_attributes_from_color(self, color)

    def set_image_texture(self, texture_map, inplace: bool = True):
        mesh = self if inplace else copy.deepcopy(self)
        if isinstance(texture_map, str):
            im = PIL.Image.open(texture_map)
            arr = np.array(im, dtype=np.float32) / 255.0
            texture_map = torch.tensor(arr, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
        mesh.texture_map = texture_map.to(device)
        return mesh

    def divide(self, inplace: bool = True):
        """
        Suddivide ogni face in 4 face con nuovi vertici: usa utils.add_vertices.
        Restituisce mesh con più vertici/facce.
        """
        mesh = self if inplace else copy.deepcopy(self)
        new_vertices, new_faces, new_vertex_normals, new_face_uvs, new_face_normals = add_vertices(mesh)
        mesh.vertices     = new_vertices.to(device)
        mesh.faces        = new_faces.to(device)
        mesh.vertex_normals = new_vertex_normals.to(device)
        # face_uvs e face_normals non usati in rendering P3D
        return mesh

    def divide_with_labels(self, face_label_map: torch.Tensor, inplace: bool = True):
        """
        Come divide(), ma torna anche la nuova mappatura delle labels per vertice.
        Usa utils.add_vertices_with_labels.
        """
        mesh = self if inplace else copy.deepcopy(self)
        new_vertices, new_faces, new_vertex_normals, new_face_label_map = add_vertices_with_labels(mesh, face_label_map)
        mesh.vertices       = new_vertices.to(device)
        mesh.faces          = new_faces.to(device)
        mesh.vertex_normals = new_vertex_normals.to(device)
        return mesh, new_face_label_map.to(device)

    def export(self, file: str, extension: str = "obj", color: torch.Tensor = None):
        """
        Esporta mesh:
         - OBJ (se extension=="obj")
         - PLY ASCII (se extension=="ply"), con colore per vertice se fornito.
        color: Tensor[V,3] valori 0–255
        """
        verts = self.vertices.cpu().numpy()
        faces = self.faces.cpu().numpy()
        if extension == "obj":
            with open(file, "w") as f:
                for vi, v in enumerate(verts):
                    if color is None:
                        f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                    else:
                        c = color[vi].cpu().numpy()
                        f.write(f"v {v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n")
                # Normali e uvs ignorati
                for face in faces:
                    # OBJ usa indici 1-based
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        elif extension == "ply":
            # default bianco se manca color
            if color is None:
                color = torch.ones_like(self.vertices) * 255
            col_np = color.cpu().numpy().astype(np.uint8)

            with open(file, "w") as f:
                f.write("ply\nformat ascii 1.0\n")
                f.write(f"element vertex {len(verts)}\n")
                f.write("property float x\nproperty float y\nproperty float z\n")
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                f.write(f"element face {len(faces)}\n")
                f.write("property list uchar int vertex_indices\n")
                f.write("end_header\n")
                # vertici
                for v, c in zip(verts, col_np):
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
                # facce
                for face in faces:
                    f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        else:
            raise ValueError(f"Unknown extension {extension} in mesh.export()")
