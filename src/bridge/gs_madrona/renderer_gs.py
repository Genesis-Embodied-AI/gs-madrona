import os
import ctypes
from pathlib import Path
from typing import Tuple

import torch

from gs_madrona._gs_madrona_batch_renderer import MadronaBatchRenderer


os.environ['MADRONA_ROOT_PATH'] = str(Path(__file__).parent.absolute())
os.environ['MADRONA_ROOT_CACHE_DIR'] = str(Path.home() / ".cache" / "madrona")


class GeomRetriever:
    def retrieve_rigid_meshes_static(self) -> dict:
        raise NotImplementedError()
    
    def retrieve_rigid_property_torch(self, num_worlds) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def retrieve_rigid_state_torch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class MadronaBatchRendererAdapter:
    """Wraps Genesis Model around MadronaBatchRenderer."""

    def __init__(
        self,
        geom_retriever: GeomRetriever,
        gpu_id: int,
        num_worlds: int,
        num_lights: int,
        cam_fovs_tensor: torch.Tensor,
        cam_znears_tensor: torch.Tensor,
        cam_zfars_tensor: torch.Tensor,
        batch_render_view_width: int = 128,
        batch_render_view_height: int = 128,
        add_cam_debug_geo: bool = False,
        use_rasterizer: bool = False,
    ):
        assert geom_retriever is not None, "GeomRetriever is required for MadronaBatchRendererAdapter"
        assert gpu_id >= 0, "GPU ID must be greater than or equal to 0"
        assert num_worlds > 0, "Number of worlds must be greater than 0"
        assert cam_fovs_tensor.shape[0] > 0, "Must have at least one camera for Madrona to work!"
        assert batch_render_view_width > 0, "Batch render view width must be greater than 0"
        assert batch_render_view_height > 0, "Batch render view height must be greater than 0"

        self.num_worlds = num_worlds
        self.geom_retriever = geom_retriever
        geom_args_static = self.geom_retriever.retrieve_rigid_meshes_static()
        for arg_name, arg_value in geom_args_static.items():
            assert arg_value.data.c_contiguous, f"{arg_name} data is not continuous."
        
        # TODO: Support mutable camera fov
        cam_fovy = cam_fovs_tensor.cpu().numpy()
        cam_znear = cam_znears_tensor.cpu().numpy()
        cam_zfar = cam_zfars_tensor.cpu().numpy()

        # Preload Nvidia compiler runtime if available (i.e. torch is not built from source)
        try:
            import nvidia.cuda_nvrtc
            nvrtc_dir = Path(nvidia.cuda_nvrtc.__file__).parent.absolute()
            libnvrtc_path, *_ = filter(Path.is_file, (nvrtc_dir / "lib").glob("libnvrtc.so.1*"))
            ctypes.CDLL(libnvrtc_path, ctypes.RTLD_LOCAL)
        except ImportError:
            pass

        self.madrona = MadronaBatchRenderer(
            gpu_id=gpu_id,
            **geom_args_static,
            num_lights=num_lights,
            num_worlds=num_worlds,
            batch_render_view_width=batch_render_view_width,
            batch_render_view_height=batch_render_view_height,
            cam_fovy=cam_fovy,
            cam_znear=cam_znear,
            cam_zfar=cam_zfar,
            add_cam_debug_geo=add_cam_debug_geo,
            use_rt=not use_rasterizer,
        )

    def init(
        self,
        cam_pos_tensor: torch.Tensor,
        cam_rot_tensor: torch.Tensor,
        lights_pos_tensor: torch.Tensor,
        lights_dir_tensor: torch.Tensor,
        lights_rgb_tensor: torch.Tensor,
        lights_directional_tensor: torch.Tensor,
        lights_castshadow_tensor: torch.Tensor,
        lights_cutoff_tensor: torch.Tensor,
        lights_attenuation_tensor: torch.Tensor,
        lights_intensity_tensor: torch.Tensor,
    ):
        geom_pos, geom_rot = self.geom_retriever.retrieve_rigid_state_torch()
        geom_mat_ids, geom_rgb, geom_sizes = self.geom_retriever.retrieve_rigid_property_torch(self.num_worlds)
        cam_pos, cam_rot = self.get_camera_pos_rot_torch(cam_pos_tensor, cam_rot_tensor)
        (
            light_pos,
            light_dir,
            light_rgb,
            light_directional,
            light_castshadow,
            light_cutoff,
            light_attenuation,
            light_intensity,
        ) = self.get_lights_properties_torch(
            lights_pos_tensor,
            lights_dir_tensor,
            lights_rgb_tensor,
            lights_directional_tensor,
            lights_castshadow_tensor,
            lights_cutoff_tensor,
            lights_attenuation_tensor,
            lights_intensity_tensor,
        )

        # Make a copy to actually shuffle the memory layout before passing to C++
        self.madrona.init(
            geom_pos,
            geom_rot,
            cam_pos,
            cam_rot,
            geom_mat_ids,
            geom_rgb,
            geom_sizes,
            light_pos,
            light_dir,
            light_rgb,
            light_directional,
            light_castshadow,
            light_cutoff,
            light_attenuation,
            light_intensity,
        )

    def render(
        self,
        cam_pos_tensor: torch.Tensor,
        cam_rot_tensor: torch.Tensor,
        render_options: torch.Tensor,
    ):
        # Assume execution on GPU
        # TODO: Need to check if the device is GPU or CPU, or assert if not GPU
        geom_pos, geom_rot = self.geom_retriever.retrieve_rigid_state_torch()
        cam_pos, cam_rot = self.get_camera_pos_rot_torch(cam_pos_tensor, cam_rot_tensor)

        self.madrona.render(
            geom_pos,
            geom_rot,
            cam_pos,
            cam_rot,
            render_options,
        )
        rgb_torch = self.madrona.rgb_tensor().to_torch()
        depth_torch = self.madrona.depth_tensor().to_torch()
        segmentation_torch = self.madrona.segmentation_tensor().to_torch()
        normal_torch = self.madrona.normal_tensor().to_torch()
        return rgb_torch, depth_torch, segmentation_torch, normal_torch 

    ########################## Utils ##########################
    def get_camera_pos_rot_torch(self, cam_pos_tensor, cam_rot_tensor):
        cam_pos = cam_pos_tensor
        cam_rot = cam_rot_tensor
        return cam_pos, cam_rot

    def get_lights_properties_torch(
        self,
        lights_pos_tensor: torch.Tensor,
        lights_dir_tensor: torch.Tensor,
        lights_rgb_tensor: torch.Tensor,
        lights_directional_tensor: torch.Tensor,
        lights_castshadow_tensor: torch.Tensor,
        lights_cutoff_tensor: torch.Tensor,
        lights_attenuation_tensor: torch.Tensor,
        lights_intensity_tensor: torch.Tensor,
    ):
        light_pos = lights_pos_tensor.reshape(-1, 3).unsqueeze(0).repeat(self.num_worlds, 1, 1)
        light_dir = lights_dir_tensor.reshape(-1, 3).unsqueeze(0).repeat(self.num_worlds, 1, 1)
        light_rgb_int = (lights_rgb_tensor * 255).to(torch.int32)  # Cast to int32
        light_rgb_uint = (light_rgb_int[:, 0] << 16) | (light_rgb_int[:, 1] << 8) | light_rgb_int[:, 2]
        light_rgb = light_rgb_uint.unsqueeze(0).repeat(self.num_worlds, 1)
        
        light_directional = lights_directional_tensor.reshape(-1).unsqueeze(0).repeat(self.num_worlds, 1)
        light_castshadow = lights_castshadow_tensor.reshape(-1).unsqueeze(0).repeat(self.num_worlds, 1)
        light_cutoff = lights_cutoff_tensor.reshape(-1).unsqueeze(0).repeat(self.num_worlds, 1)
        light_attenuation = lights_attenuation_tensor.reshape(-1).unsqueeze(0).repeat(self.num_worlds, 1)
        light_intensity = lights_intensity_tensor.reshape(-1).unsqueeze(0).repeat(self.num_worlds, 1)
        return (
            light_pos,
            light_dir,
            light_rgb,
            light_directional,
            light_castshadow,
            light_cutoff,
            light_attenuation,
            light_intensity,
        )
