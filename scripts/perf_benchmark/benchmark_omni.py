######################## Parse arguments #######################
from batch_benchmark import BenchmarkArgs, write_benchmark_result_file
benchmark_args = BenchmarkArgs.parse_benchmark_args()
######################## Launch app #######################
from isaaclab.app import AppLauncher
app = AppLauncher(
    headless=not benchmark_args.gui,
    enable_cameras=True,
    device="cuda:0",
).app

import os
import math
from scipy.spatial.transform import Rotation as R
import torch
from pxr import PhysxSchema
from PIL import Image

import carb
import omni.replicator.core as rep
import isaaclab.sim as sim_utils
import isaacsim.core.utils.stage as stage_utils
import isaaclab.assets as asset_utils
import isaaclab_assets.robots as asset_robots
from isaaclab.scene.interactive_scene import InteractiveScene
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.asset.importer.mjcf")
import isaacsim.asset.importer.mjcf

from benchmark_profiler import BenchmarkProfiler, get_utilization_percentages, print_system_utilization


# Get asset directory from environment variable
asset_dir = os.path.abspath(os.getenv("ASSET_DIR"))
benchmark_dir = os.path.abspath(os.path.dirname(__file__))


def load_mjcf(mjcf_path: str) -> str:
    return MjcfConverter(
        MjcfConverterCfg(
            asset_path=mjcf_path,
            fix_base=True,
            force_usd_conversion=True
        )
    ).usd_path


def get_robot_config() -> asset_utils.AssetBaseCfg:
    robot_name = f"{os.path.splitext(benchmark_args.mjcf)[0]}_new.xml"
    robot_path = load_mjcf(os.path.abspath(os.path.join(asset_dir, robot_name)))
    print("Robot asset:", robot_path)

    if benchmark_args.mjcf.endswith("g1.xml"):
        robot_cfg = asset_utils.AssetBaseCfg(
            spawn=asset_robots.unitree.G1_CFG.spawn.copy()
        )
    elif benchmark_args.mjcf.endswith("go2.xml"):
        robot_cfg = asset_utils.AssetBaseCfg(
            spawn=asset_robots.unitree.UNITREE_GO2_CFG.spawn.copy()
        )
    elif benchmark_args.mjcf.endswith("panda.xml"):
        robot_cfg = asset_utils.AssetBaseCfg(
            spawn=asset_robots.franka.FRANKA_PANDA_CFG.spawn.copy()
        )
    else:
        raise Exception(f"Invalid robot: {benchmark_args.mjcf}")
    robot_cfg.spawn.usd_path = robot_path
    return robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")


def get_dir_light_config() -> asset_utils.AssetBaseCfg:
    dir_light_pos = torch.Tensor([[0.0, 0.0, 1.5]])
    dir_light_quat = quat_from_matrix(
        create_rotation_matrix_from_view(
            dir_light_pos,
            torch.Tensor([[1.0, 1.0, -2.0]]),
            stage_utils.get_stage_up_axis()))
    dir_light_pos = tuple(dir_light_pos.detach().cpu().squeeze().numpy())
    dir_light_quat = tuple(dir_light_quat.detach().cpu().squeeze().numpy())
    dir_light_cfg = asset_utils.AssetBaseCfg(
        prim_path="/World/direct_light",
        spawn=sim_utils.DistantLightCfg(intensity=500.0, angle=45.0),
        init_state=asset_utils.AssetBaseCfg.InitialStateCfg(
            pos=dir_light_pos, rot=dir_light_quat
        )
    )
    return dir_light_cfg


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""
    ground = asset_utils.AssetBaseCfg(
        # prim_path="{ENV_REGEX_NS}/ground",  # Each environment should have a ground 
        prim_path="/World/ground",        # All environment shares a ground
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.abspath(os.path.join(benchmark_dir, "benchmark_assets/plane_usd/plane.usd"))
        ),
    )
    robot: asset_utils.ArticulationCfg = get_robot_config()
    dir_light = get_dir_light_config()


def apply_benchmark_physics_settings():
    stage = stage_utils.get_current_stage()
    physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
    physxSceneAPI.CreateGpuTempBufferCapacityAttr(16 * 1024 * 1024 * 2)
    physxSceneAPI.CreateGpuHeapCapacityAttr(64 * 1024 * 1024 * 2)
    physxSceneAPI.CreateGpuMaxRigidPatchCountAttr(8388608)
    physxSceneAPI.CreateGpuMaxRigidContactCountAttr(16777216)


def print_render_settings(settings):
    print("Render mode:", settings.get("/rtx/rendermode"))
    print("Sample per pixel:", settings.get("/rtx/pathtracing/spp"))
    print("Total spp:", settings.get("/rtx/pathtracing/totalSpp"))
    print("Clamp spp:", settings.get("/rtx/pathtracing/clampSpp"))
    print("Max bounce:", settings.get("/rtx/pathtracing/maxBounces"))
    print("Optix Denoiser", settings.get("/rtx/pathtracing/optixDenoiser/enabled"))
    print("Shadows", settings.get("/rtx/shadows/enabled"))
    print("dlss/enabled:", settings.get("/rtx/post/dlss/enabled"))
    print("dlss/auto:", settings.get("/rtx/post/dlss/auto"))
    print("upscaling/enabled:", settings.get("/rtx/post/upscaling/enabled"))
    print("aa/denoiser/enabled:", settings.get("/rtx/post/aa/denoiser/enabled"))
    print("aa/taa/enabled:", settings.get("/rtx/post/aa/taa/enabled"))
    print("motionBlur/enabled:", settings.get("/rtx/post/motionBlur/enabled"))
    print("dof/enabled:", settings.get("/rtx/post/dof/enabled"))
    print("bloom/enabled:", settings.get("/rtx/post/bloom/enabled"))
    print("tonemap/enabled:", settings.get("/rtx/post/tonemap/enabled"))
    print("exposure/enabled:", settings.get("/rtx/post/exposure/enabled"))
    print("vsync:", settings.get("/app/window/vsync"))


def apply_benchmark_carb_settings(print_changes: bool = False) -> None:
    # rep.settings.set_render_rtx_realtime()  # Keep default pipeline; explicitly set below
    settings = carb.settings.get_settings()
    
    # Print settings before applying the settings
    if print_changes:
        print("Before settings:")
        print_render_settings(settings)

    # Options: https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_pt.html
    if benchmark_args.rasterizer:
        settings.set("/rtx/rendermode", "RayTracedLighting")
    else:
        settings.set("/rtx/rendermode", "PathTracing")
    settings.set("/rtx/shadows/enabled", False)

    # Path tracing settings
    settings.set("/rtx/pathtracing/spp", benchmark_args.spp)
    settings.set("/rtx/pathtracing/totalSpp", benchmark_args.spp)
    settings.set("/rtx/pathtracing/clampSpp", benchmark_args.spp)
    settings.set("/rtx/pathtracing/maxBounces", benchmark_args.max_bounce)
    settings.set("/rtx/pathtracing/optixDenoiser/enabled", False)
    settings.set("/rtx/pathtracing/adaptiveSampling/enabled", False)

    # Disable DLSS & upscaling
    settings.set("/rtx-transient/dlssg/enabled", False)
    settings.set("/rtx/post/dlss/enabled", False)
    settings.set("/rtx/post/dlss/auto", False)
    settings.set("/rtx/post/upscaling/enabled", False)

    # Disable post-processing
    settings.set("/rtx/post/aa/denoiser/enabled", False)
    settings.set("/rtx/post/aa/taa/enabled", False)
    settings.set("/rtx/post/motionBlur/enabled", False)
    settings.set("/rtx/post/dof/enabled", False)
    settings.set("/rtx/post/bloom/enabled", False)
    settings.set("/rtx/post/tonemap/enabled", False)
    settings.set("/rtx/post/exposure/enabled", False)

    # Disable VSync
    settings.set("/app/window/vsync", False)

    # Print settings after applying the settings
    if print_changes:
        print("After settings:")
        print_render_settings(settings)


def create_scene():
    """Create simulation and scene with camera and physics/render settings applied."""
    sim_cfg = sim_utils.SimulationCfg(
        device="cuda:0", dt=0.01, use_fabric=False,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = RobotSceneCfg(num_envs=benchmark_args.n_envs, env_spacing=10.0)

    apply_benchmark_physics_settings()
    apply_benchmark_carb_settings(True)

    camera_fov = math.radians(benchmark_args.camera_fov)
    camera_aperture = 20.955
    camera_fol = camera_aperture / (2 * math.tan(camera_fov / 2))

    camera_pos = torch.tensor((
        benchmark_args.camera_posX,
        benchmark_args.camera_posY,
        benchmark_args.camera_posZ
    )).reshape(-1, 3)
    camera_lookat = torch.tensor((
        benchmark_args.camera_lookatX,
        benchmark_args.camera_lookatY,
        benchmark_args.camera_lookatZ
    )).reshape(-1, 3)
    camera_quat = quat_from_matrix(
        create_rotation_matrix_from_view(
            camera_lookat, camera_pos, stage_utils.get_stage_up_axis()
        ) @ R.from_euler('z', 180, degrees=True).as_matrix()   
    )
    camera_pos = tuple(camera_pos.detach().cpu().squeeze().numpy())
    camera_quat = tuple(camera_quat.detach().cpu().squeeze().numpy())
    camera_cfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/tiled_camera",
        update_period=0,
        height=benchmark_args.resY,
        width=benchmark_args.resX,
        offset=TiledCameraCfg.OffsetCfg(
            pos=camera_pos,
            rot=camera_quat,
            convention="ros"
        ),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=camera_fol,
            horizontal_aperture=camera_aperture,
        ),
    )
    setattr(scene_cfg, "tiled_camera", camera_cfg)
    scene = InteractiveScene(scene_cfg)
    return sim, scene


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
) -> None:
    """Run the simulator with all cameras, and return timing analytics. Visualize if desired."""
    n_envs = benchmark_args.n_envs
    n_steps = benchmark_args.n_steps
    camera = scene["tiled_camera"]
    camera_data_types = ["rgb", "depth"]

    # Initialize timing variables
    system_utilization_analytics = get_utilization_percentages()
    print_system_utilization(system_utilization_analytics)

    sim.reset()
    dt = sim.get_physics_dt()
    n_warm_steps = 3
    for i in range(n_warm_steps):
        print(f"Warm up step {i}.")
        sim.step()
        camera.update(dt)
        _ = camera.data

    print("Warm up finished.")
    output_dir = os.path.dirname(benchmark_args.benchmark_result_file)
    os.makedirs(output_dir, exist_ok=True)
    image_dirname = f'{benchmark_args.renderer}-{benchmark_args.rasterizer}-{benchmark_args.n_envs}-{benchmark_args.resX}'
    image_dir = os.path.join(output_dir, image_dirname)

    profiler = BenchmarkProfiler(n_steps, n_envs)
    for i in range(n_steps):
        print(f"Step {i}:")
        get_utilization_percentages()

        # Measure the total simulation step time
        profiler.on_simulation_start()
        sim.step(render=False)
        profiler.on_rendering_start()
        sim.render()

        # Update cameras and process vision data within the simulation step
        # Loop through all camera lists and their data_types
        camera.update(dt=dt)
        rgb_tiles = camera.data.output.get("rgb")
        depth_tiles = camera.data.output.get("depth")
        profiler.on_rendering_end()

        if n_steps < 10:
            os.makedirs(image_dir, exist_ok=True)
            rgb_tiles = rgb_tiles.detach().cpu().numpy()
            for j in range(n_envs):
                rgb_image = rgb_tiles[j]
                rgb_image = Image.fromarray(rgb_image)

                image_name = f"rgb_step{i}_env{j}.png"
                image_path = os.path.join(image_dir, image_name)
                rgb_image.save(image_path)
                print("Image saved:", image_path)
        # End timing for the step

    profiler.end()
    profiler.print_summary()

    system_utilization_analytics = get_utilization_percentages()
    print_system_utilization(system_utilization_analytics)

    performance_results = {
        "time_taken_gpu": profiler.get_total_rendering_gpu_time(),
        "time_taken_cpu": profiler.get_total_rendering_cpu_time(),
        "time_taken_per_env_gpu": profiler.get_total_rendering_gpu_time_per_env(),
        "time_taken_per_env_cpu": profiler.get_total_rendering_cpu_time_per_env(),
        "fps": profiler.get_rendering_fps(),
        "fps_per_env": profiler.get_rendering_fps_per_env(),
    }
    write_benchmark_result_file(benchmark_args, performance_results)
        
    print("App closing..")
    # app.close()
    print("App closed!")


def main() -> None:
    """Entry point for running the benchmark scene and simulator."""
    sim, scene = create_scene()
    run_simulator(sim=sim, scene=scene)


if __name__ == "__main__":
    # run the main function
    main()
    # simulation_app.close()
