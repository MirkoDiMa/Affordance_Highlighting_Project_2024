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

class MeshRendererP3D:
    def __init__(
        self,
        image_size: int = 224,
        blur_radius: float = 1e-4,
        faces_per_pixel: int = 50,
        blend_sigma: float = 1e-4,
        blend_gamma: float = 1e-4
    ):
        # camere ortografiche
        self.cameras = FoVOrthographicCameras(device=device)
        # soft rasterization: piccolo blur + tanti frammenti
        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel,
        )
        # luci
        self.lights = PointLights(device=device)
        # renderer con SoftPhongShader e blend params morbidi
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
        images = self.renderer(mesh)       # [1,H,W,3] (soft RGBA già tagliato)
        return images.permute(0, 3, 1, 2)  # → [1,3,H,W]
