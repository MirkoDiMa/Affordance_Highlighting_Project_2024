# renderers/mesh_renderer.py
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVOrthographicCameras, RasterizationSettings, MeshRenderer,
    MeshRasterizer, SoftPhongShader, PointLights, BlendParams, TexturesVertex
)
from utils import device

class MeshRendererP3D:
    def __init__(self, image_size=224):
        self.cameras = FoVOrthographicCameras(device=device)
        self.raster_settings = RasterizationSettings(
            image_size=image_size, blur_radius=0.0, faces_per_pixel=1
        )
        self.lights = PointLights(device=device)
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device=device, cameras=self.cameras,
                lights=self.lights, blend_params=BlendParams()
            )
        )

    def render(self, verts, faces, vert_colors, background=None):
        textures = TexturesVertex(verts_features=vert_colors.unsqueeze(0))
        mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
        images = self.renderer(mesh)      # [1,H,W,3]
        return images.permute(0, 3, 1, 2)  # -> [1,3,H,W]

