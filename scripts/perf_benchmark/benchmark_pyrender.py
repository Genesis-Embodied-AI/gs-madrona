import os

import genesis as gs

from batch_benchmark import BenchmarkArgs
from benchmark_profiler import BenchmarkProfiler


def init_gs(benchmark_args):
    ########################## init ##########################
    try:
        gs.init(backend=gs.gpu)
    except Exception as e:
        print(f"Failed to initialize GPU backend: {e}")
        print("Falling back to CPU backend")
        gs.init(backend=gs.cpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(benchmark_args.camera_posX, benchmark_args.camera_posY, benchmark_args.camera_posZ),
            camera_lookat=(benchmark_args.camera_lookatX, benchmark_args.camera_lookatY, benchmark_args.camera_lookatZ),
            camera_fov=benchmark_args.camera_fov,
        ),
        vis_options=gs.options.VisOptions(
            lights=[
                {"type": "directional", "dir": (1.0, 1.0, -2.0), "color": (1.0, 1.0, 1.0), "intensity": 0.5},
                {"type": "point", "pos": (4, -4, 4), "color": (1.0, 1.0, 1.0), "intensity": 1},
            ],
        ),
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(
            # constraint_solver=gs.constraint_solver.Newton,
        ),
        renderer=benchmark_args.rasterizer and gs.options.renderers.Rasterizer() or gs.options.renderers.RayTracer(),
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file=benchmark_args.mjcf),
        visualize_contact=False,
    )

    ########################## cameras ##########################
    cam_0 = scene.add_camera(
        res=(benchmark_args.resX, benchmark_args.resY),
        pos=(benchmark_args.camera_posX, benchmark_args.camera_posY, benchmark_args.camera_posZ),
        lookat=(benchmark_args.camera_lookatX, benchmark_args.camera_lookatY, benchmark_args.camera_lookatZ),
        fov=benchmark_args.camera_fov,
    )
    ########################## build ##########################
    scene.build()
    return scene


def run_benchmark(scene, benchmark_args):
    try:
        n_envs = benchmark_args.n_envs
        n_steps = benchmark_args.n_steps

        # warmup
        scene.step()
        rgb, depth, _, _ = scene.visualizer.cameras[0].render(rgb=True, depth=True)

        # Profiler
        profiler = BenchmarkProfiler(n_steps, n_envs)
        for i in range(n_steps):
            profiler.on_simulation_start()
            scene.step()
            profiler.on_rendering_start()
            rgb, depth, _, _ = scene.visualizer.cameras[0].render(rgb=True, depth=True)
            profiler.on_rendering_end()

        profiler.end()
        profiler.print_summary()

        time_taken_gpu = profiler.get_total_rendering_gpu_time()
        time_taken_cpu = profiler.get_total_rendering_cpu_time()
        time_taken_per_env_gpu = profiler.get_total_rendering_gpu_time_per_env()
        time_taken_per_env_cpu = profiler.get_total_rendering_cpu_time_per_env()
        fps = profiler.get_rendering_fps()
        fps_per_env = profiler.get_rendering_fps_per_env()

        # Append a line with all args and results in csv format
        os.makedirs(os.path.dirname(benchmark_args.benchmark_result_file), exist_ok=True)
        with open(benchmark_args.benchmark_result_file, "a") as f:
            f.write(
                f"succeeded,{benchmark_args.mjcf},{benchmark_args.renderer},{benchmark_args.rasterizer},{benchmark_args.n_envs},{benchmark_args.n_steps},{benchmark_args.resX},{benchmark_args.resY},{benchmark_args.camera_posX},{benchmark_args.camera_posY},{benchmark_args.camera_posZ},{benchmark_args.camera_lookatX},{benchmark_args.camera_lookatY},{benchmark_args.camera_lookatZ},{benchmark_args.camera_fov},{time_taken_gpu},{time_taken_per_env_gpu},{time_taken_cpu},{time_taken_per_env_cpu},{fps},{fps_per_env}\n"
            )
    except Exception as e:
        print(f"Error during benchmark: {e}")
        raise


def main():
    ######################## Parse arguments #######################
    benchmark_args = BenchmarkArgs.parse_benchmark_args()

    ######################## Initialize scene #######################
    scene = init_gs(benchmark_args)

    ######################## Run benchmark #######################
    run_benchmark(scene, benchmark_args)


if __name__ == "__main__":
    main()
