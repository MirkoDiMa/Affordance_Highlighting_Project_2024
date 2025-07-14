import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    BlendParams,
    TexturesVertex
)
from utils import device

from pytorch3d.renderer import FoVPerspectiveCameras

class MeshRendererP3D:
    def __init__(self,
                 image_size: int = 224,
                 blur_radius: float = 0.01,
                 faces_per_pixel: int = 100,
                 blend_sigma: float = 0.01,
                 blend_gamma: float = 0.01,
                 fov: float = 60.0    # campo visivo in gradi
    ):
        # 1) proiezione prospettica, non ortografica
        self.cameras = FoVPerspectiveCameras(
            fov=(fov,),    # tuple di un solo elemento
            device=device
        )
        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel,
        )
        self.lights = PointLights(device=device)
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=self.cameras,
                lights=self.lights,
                blend_params=BlendParams(sigma=blend_sigma, gamma=blend_gamma)
            )
        )

    def render(self, verts, faces, vert_colors, background=None):
        textures = TexturesVertex(verts_features=vert_colors.unsqueeze(0))
        mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
        images = self.renderer(mesh)       # [1,H,W,3]
        return images.permute(0, 3, 1, 2)  # → [1,3,H,W]

