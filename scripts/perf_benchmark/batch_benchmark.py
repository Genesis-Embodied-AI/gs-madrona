import argparse
import subprocess
import os
from datetime import datetime

import pandas as pd

from benchmark_report_generator import generate_report
from benchmark_configs import BenchmarkConfigs

# Example command:
# python batch_benchmark.py -f benchmark_config_smoke_test.yml


# Create a struct to store the arguments
class BenchmarkArgs:
    def __init__(
        self,
        renderer,
        rasterizer,
        n_envs,
        n_steps,
        resX,
        resY,
        camera_posX,
        camera_posY,
        camera_posZ,
        camera_lookatX,
        camera_lookatY,
        camera_lookatZ,
        camera_fov,
        mjcf,
        benchmark_result_file,
        benchmark_config_file,
        max_bounce,
        spp,
        gui=False,
        benchmark_script=None,
        renderer_timeout=None,
    ):
        self.renderer = renderer
        self.rasterizer = rasterizer
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.resX = resX
        self.resY = resY
        self.camera_posX = camera_posX
        self.camera_posY = camera_posY
        self.camera_posZ = camera_posZ
        self.camera_lookatX = camera_lookatX
        self.camera_lookatY = camera_lookatY
        self.camera_lookatZ = camera_lookatZ
        self.camera_fov = camera_fov
        self.mjcf = mjcf
        self.benchmark_result_file = benchmark_result_file
        self.benchmark_config_file = benchmark_config_file
        self.max_bounce = max_bounce
        self.spp = spp
        self.gui = gui
        self.benchmark_script = benchmark_script
        self.renderer_timeout = renderer_timeout

    @staticmethod
    def parse_benchmark_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("-d", "--renderer", required=True, type=str)
        parser.add_argument("-r", "--rasterizer", action="store_true", default=False)
        parser.add_argument("-n", "--n_envs", required=True, type=int)
        parser.add_argument("-x", "--resX", required=True, type=int)
        parser.add_argument("-y", "--resY", required=True, type=int)
        parser.add_argument("-f", "--mjcf", required=True, type=str)
        parser.add_argument("-g", "--benchmark_result_file", required=True, type=str)
        parser.add_argument("-c", "--benchmark_config_file", required=True, type=str)
        args = parser.parse_args()
        benchmark_config = BenchmarkConfigs(args.benchmark_config_file)
        benchmark_args = BenchmarkArgs(
            renderer=args.renderer,
            rasterizer=args.rasterizer,
            n_envs=args.n_envs,
            n_steps=benchmark_config.n_steps,
            resX=args.resX,
            resY=args.resY,
            camera_posX=benchmark_config.camera_pos[0],
            camera_posY=benchmark_config.camera_pos[1],
            camera_posZ=benchmark_config.camera_pos[2],
            camera_lookatX=benchmark_config.camera_lookat[0],
            camera_lookatY=benchmark_config.camera_lookat[1],
            camera_lookatZ=benchmark_config.camera_lookat[2],
            camera_fov=benchmark_config.camera_fov,
            mjcf=args.mjcf,
            benchmark_result_file=args.benchmark_result_file,
            benchmark_config_file=args.benchmark_config_file,
            max_bounce=benchmark_config.max_bounce,
            spp=benchmark_config.spp,
            gui=benchmark_config.gui,
        )
        print(f"Benchmark with args:")
        print(f"  renderer: {benchmark_args.renderer}")
        print(f"  rasterizer: {benchmark_args.rasterizer}")
        print(f"  n_envs: {benchmark_args.n_envs}")
        print(f"  n_steps: {benchmark_args.n_steps}")
        print(f"  resolution: {benchmark_args.resX}x{benchmark_args.resY}")
        print(
            f"  camera_pos: ({benchmark_args.camera_posX}, {benchmark_args.camera_posY}, {benchmark_args.camera_posZ})"
        )
        print(
            f"  camera_lookat: ({benchmark_args.camera_lookatX}, {benchmark_args.camera_lookatY}, {benchmark_args.camera_lookatZ})"
        )
        print(f"  camera_fov: {benchmark_args.camera_fov}")
        print(f"  mjcf: {benchmark_args.mjcf}")
        print(f"  benchmark_result_file: {benchmark_args.benchmark_result_file}")
        print(f"  benchmark_config_file: {benchmark_args.benchmark_config_file}")
        print(f"  max_bounce: {benchmark_args.max_bounce}")
        print(f"  spp: {benchmark_args.spp}")
        print(f"  gui: {benchmark_args.gui}")
        return benchmark_args


class BatchBenchmarkArgs:
    def __init__(self, config_file, continue_from):
        self.config_file = config_file
        self.continue_from = continue_from

    def parse_batch_benchmark_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("-f", "--config_file", type=str, default="benchmark_config_smoke_test.yml")
        parser.add_argument("-c", "--continue_from", type=str, default=None)
        args = parser.parse_args()
        return BatchBenchmarkArgs(config_file=args.config_file, continue_from=args.continue_from)


def create_batch_args(benchmark_result_file, config_file):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(benchmark_result_file), exist_ok=True)

    # Load configuration
    config = BenchmarkConfigs(config_file)
    mjcf_list = config.mjcf_list
    renderer_list = config.renderer_list
    rasterizer_list = config.rasterizer_list
    batch_size_list = config.batch_size_list
    resolution_list = config.resolution_list
    n_steps = config.n_steps
    camera_pos = config.camera_pos
    camera_lookat = config.camera_lookat
    camera_fov = config.camera_fov
    max_bounce = config.max_bounce
    spp = config.spp
    gui = config.gui

    # Batch data for resolution and batch size needs to be sorted in ascending order of resX x resY
    # so that if one resolution fails, all the resolutions, which are larger, will be skipped.
    resolution_list.sort(key=lambda x: x[0] * x[1])

    # Create a hierarchical dictionary to store all combinations
    batch_args_dict = {}

    # Build hierarchical structure
    for renderer_info in renderer_list:
        renderer = renderer_info["renderer"]
        benchmark_script = renderer_info["benchmark_script"]
        renderer_timeout = renderer_info["timeout"]
        batch_args_dict[renderer] = {}
        for rasterizer in rasterizer_list:
            batch_args_dict[renderer][rasterizer] = {}
            for mjcf in mjcf_list:
                batch_args_dict[renderer][rasterizer][mjcf] = {}
                for batch_size in batch_size_list:
                    batch_args_dict[renderer][rasterizer][mjcf][batch_size] = {}
                    for resolution in resolution_list:
                        resX, resY = resolution
                        # Create benchmark args for this combination
                        args = BenchmarkArgs(
                            renderer=renderer,
                            rasterizer=rasterizer,
                            n_envs=batch_size,
                            n_steps=n_steps,
                            resX=resX,
                            resY=resY,
                            camera_posX=camera_pos[0],
                            camera_posY=camera_pos[1],
                            camera_posZ=camera_pos[2],
                            camera_lookatX=camera_lookat[0],
                            camera_lookatY=camera_lookat[1],
                            camera_lookatZ=camera_lookat[2],
                            camera_fov=camera_fov,
                            mjcf=mjcf,
                            benchmark_result_file=benchmark_result_file,
                            benchmark_config_file=config_file,
                            max_bounce=max_bounce,
                            spp=spp,
                            gui=gui,
                            benchmark_script=benchmark_script,
                            renderer_timeout=renderer_timeout,
                        )
                        batch_args_dict[renderer][rasterizer][mjcf][batch_size][(resX, resY)] = args

    return batch_args_dict


def create_benchmark_result_file(continue_from_file):
    if continue_from_file is not None:
        if not os.path.exists(continue_from_file):
            raise FileNotFoundError(f"Continue from file not found: {continue_from_file}")
        print(f"Continuing from file: {continue_from_file}")
        return continue_from_file
    else:
        # Create benchmark result data file with header
        benchmark_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Get the benchmark project root directory
        project_root = os.path.dirname(os.path.abspath(__file__))
        benchmark_data_directory = os.path.join(
            project_root, "benchmark_reports", f"perf_benchmark_{benchmark_timestamp}"
        )

        if not os.path.exists(benchmark_data_directory):
            os.makedirs(benchmark_data_directory)

        benchmark_result_file = f"{benchmark_data_directory}/perf_data.csv"
        with open(benchmark_result_file, "w") as f:
            f.write(
                "result,mjcf,renderer,rasterizer,n_envs,n_steps,resX,resY,camera_posX,camera_posY,camera_posZ,camera_lookatX,camera_lookatY,camera_lookatZ,camera_fov,time_taken_gpu,time_taken_per_env_gpu,time_taken_cpu,time_taken_per_env_cpu,fps,fps_per_env\n"
            )
        print(f"Created new benchmark result file: {benchmark_result_file}")
        return benchmark_result_file


def get_previous_runs(continue_from_file):
    if continue_from_file is None:
        return []

    # Read the existing benchmark data file
    df = pd.read_csv(continue_from_file)

    # Create a list of tuples containing run info and status
    previous_runs = []

    for _, row in df.iterrows():
        run_info = (
            row["mjcf"],
            row["renderer"],
            row["rasterizer"],
            row["n_envs"],
            (row["resX"], row["resY"]),
            row["result"],  # 'succeeded' or 'failed'
        )
        previous_runs.append(run_info)

    return previous_runs


def run_batch_benchmark(batch_args_dict, previous_runs=None):
    if previous_runs is None:
        previous_runs = []

    for renderer in batch_args_dict:
        print(f"Running benchmark for {renderer}")
        for rasterizer in batch_args_dict[renderer]:
            for mjcf in batch_args_dict[renderer][rasterizer]:
                for batch_size in batch_args_dict[renderer][rasterizer][mjcf]:
                    last_resolution_failed = False
                    for resolution in batch_args_dict[renderer][rasterizer][mjcf][batch_size]:
                        if last_resolution_failed:
                            break

                        # Check if this run was in a previous execution
                        run_info = (mjcf, renderer, rasterizer, batch_size, resolution)
                        skip_this_run = False

                        for prev_run in previous_runs:
                            if run_info == prev_run[:5]:  # Compare only the run parameters, not the status
                                skip_this_run = True
                                if prev_run[4] == "failed":
                                    # Skip this and subsequent resolutions if it failed before
                                    last_resolution_failed = True
                                break

                        if skip_this_run:
                            continue

                        # Run the benchmark
                        batch_args = batch_args_dict[renderer][rasterizer][mjcf][batch_size][resolution]

                        # launch a process to run the benchmark
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        benchmark_script_path = os.path.join(current_dir, batch_args.benchmark_script)
                        if not os.path.exists(benchmark_script_path):
                            raise FileNotFoundError(f"Benchmark script not found: {benchmark_script_path}")
                        cmd = ["python3", benchmark_script_path]
                        if batch_args.rasterizer:
                            cmd.append("--rasterizer")
                        cmd.extend(
                            [
                                "--renderer",
                                batch_args.renderer,
                                "--n_envs",
                                str(batch_args.n_envs),
                                "--resX",
                                str(batch_args.resX),
                                "--resY",
                                str(batch_args.resY),
                                "--mjcf",
                                batch_args.mjcf,
                                "--benchmark_result_file",
                                batch_args.benchmark_result_file,
                                "--benchmark_config_file",
                                batch_args.benchmark_config_file,
                            ]
                        )
                        try:
                            # Read timeout from config
                            process = subprocess.Popen(cmd)
                            try:
                                # Hack to avoid omniverse runs to take forever.
                                timeout = batch_args.renderer_timeout
                                return_code = process.wait(timeout=timeout)
                                if return_code != 0:
                                    raise subprocess.CalledProcessError(return_code, cmd)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()  # Wait for the process to be killed
                                raise TimeoutError(f"Process did not complete within {timeout} seconds")
                        except Exception as e:
                            print(f"Error running benchmark: {str(e)}")
                            if isinstance(e, subprocess.CalledProcessError):
                                last_resolution_failed = True
                            # Write failed result without timing data
                            with open(batch_args.benchmark_result_file, "a") as f:
                                f.write(
                                    f"failed,{batch_args.mjcf},{batch_args.renderer},{batch_args.rasterizer},{batch_args.n_envs},{batch_args.n_steps},{batch_args.resX},{batch_args.resY},{batch_args.camera_posX},{batch_args.camera_posY},{batch_args.camera_posZ},{batch_args.camera_lookatX},{batch_args.camera_lookatY},{batch_args.camera_lookatZ},{batch_args.camera_fov},,,,,,\n"
                                )


def sort_and_dedupe_benchmark_result_file(benchmark_result_file):
    # Sort by mjcf asc, renderer asc, rasterizer desc, n_envs asc, resX asc, resY asc, n_envs asc
    df = pd.read_csv(benchmark_result_file)
    df = df.sort_values(
        by=["mjcf", "renderer", "rasterizer", "resX", "resY", "n_envs", "result"],
        ascending=[True, True, False, True, True, True, False],
    )

    # Deduplicate by keeping the first occurrence of each unique combination of mjcf, renderer, rasterizer, resX, resY, n_envs
    # Keep succeeded runs if there are multiple runs for the same combination.
    df = df.drop_duplicates(subset=["mjcf", "renderer", "rasterizer", "resX", "resY", "n_envs"], keep="first")
    df.to_csv(benchmark_result_file, index=False)


def main():
    batch_benchmark_args = BatchBenchmarkArgs.parse_batch_benchmark_args()
    benchmark_result_file = create_benchmark_result_file(batch_benchmark_args.continue_from)

    # Get list of previous runs if continuing from a previous run
    previous_runs = get_previous_runs(batch_benchmark_args.continue_from)

    # Run benchmark in batch
    batch_args_dict = create_batch_args(benchmark_result_file, config_file=batch_benchmark_args.config_file)
    run_batch_benchmark(batch_args_dict, previous_runs)

    # Sort benchmark result file
    sort_and_dedupe_benchmark_result_file(benchmark_result_file)

    # Generate plots
    generate_report(benchmark_result_file, config_file=batch_benchmark_args.config_file)


if __name__ == "__main__":
    main()
