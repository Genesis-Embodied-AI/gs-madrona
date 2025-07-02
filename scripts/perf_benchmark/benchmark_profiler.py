import torch
import numpy as np
import time


class BenchmarkProfiler:
    def __init__(self, n_steps, n_envs):
        self.reset(n_steps)
        self.n_envs = n_envs

    def reset(self, n_steps):
        self.n_steps = n_steps
        # Create arrays of CUDA events for each step
        # Each step has 3 events: simulation_start, render_start, render_end
        self.events = []
        # CPU timing arrays
        self.cpu_times = []
        for _ in range(n_steps):
            step_events = {
                "simulation_start": torch.cuda.Event(enable_timing=True),
                "render_start": torch.cuda.Event(enable_timing=True),
                "render_end": torch.cuda.Event(enable_timing=True),
            }
            self.events.append(step_events)
            # Initialize CPU timing structure for each step
            self.cpu_times.append({"simulation_start": 0.0, "render_start": 0.0, "render_end": 0.0})
        self.current_step = 0

        # Synchronize all previous GPU events
        torch.cuda.synchronize()
        self.is_synchronized = False

    ######################## Profiling Events #######################
    def on_simulation_start(self):
        """Record the start of simulation for current step"""
        if self.current_step >= self.n_steps:
            raise Exception("All steps have been profiled")
        self.events[self.current_step]["simulation_start"].record()
        self.cpu_times[self.current_step]["simulation_start"] = time.time()

    def on_rendering_start(self):
        """Record the start of rendering for current step"""
        if self.current_step >= self.n_steps:
            raise Exception("All steps have been profiled")
        self.events[self.current_step]["render_start"].record()
        self.cpu_times[self.current_step]["render_start"] = time.time()

    def on_rendering_end(self):
        """Record the end of rendering for current step"""
        if self.current_step >= self.n_steps:
            raise Exception("All steps have been profiled")
        self.events[self.current_step]["render_end"].record()
        self.cpu_times[self.current_step]["render_end"] = time.time()
        self.current_step += 1

    def end(self):
        """End the profiler"""
        self._synchronize()

    def _synchronize(self):
        """Synchronize GPU to ensure all events are recorded"""
        torch.cuda.synchronize()
        self.is_synchronized = True

    ######################## Simulation Performance #######################
    def get_total_simulation_gpu_time(self):
        """Calculate total simulation GPU time across all steps in seconds"""
        if not self.is_synchronized:
            raise Exception("GPU profiler is not synchronized")
        total_time = 0.0
        for step in range(self.current_step):
            events = self.events[step]
            total_time += events["simulation_start"].elapsed_time(events["render_start"])
        return total_time / 1000.0

    def get_total_simulation_cpu_time(self):
        """Calculate total simulation CPU time across all steps in seconds"""
        if not self.is_synchronized:
            raise Exception("GPU profiler is not synchronized")
        total_time = 0.0
        for step in range(self.current_step):
            cpu_times = self.cpu_times[step]
            total_time += (cpu_times["render_start"] - cpu_times["simulation_start"]) * 1000  # Convert to ms
        return total_time / 1000.0

    def get_simulation_fps(self):
        """Get the FPS for the current step"""
        if not self.is_synchronized:
            raise Exception("GPU profiler is not synchronized")
        return self.n_envs * self.n_steps / self.get_total_simulation_gpu_time()

    def get_simulation_fps_per_env(self):
        """Get the FPS per env for the current step"""
        if not self.is_synchronized:
            raise Exception("GPU profiler is not synchronized")
        return self.n_steps / self.get_total_simulation_gpu_time()

    ######################## Rendering Performance #######################
    def get_total_rendering_gpu_time(self):
        """Calculate total rendering GPU time across all steps in seconds"""
        if not self.is_synchronized:
            raise Exception("GPU profiler is not synchronized")
        total_time = 0.0
        for step in range(self.current_step):
            events = self.events[step]
            total_time += events["render_start"].elapsed_time(events["render_end"])
        return total_time / 1000.0

    def get_total_rendering_cpu_time(self):
        """Calculate total rendering CPU time across all steps in seconds"""
        if not self.is_synchronized:
            raise Exception("GPU profiler is not synchronized")
        total_time = 0.0
        for step in range(self.current_step):
            cpu_times = self.cpu_times[step]
            total_time += (cpu_times["render_end"] - cpu_times["render_start"]) * 1000  # Convert to ms
        return total_time / 1000.0

    def get_rendering_fps(self):
        """Get the FPS for the current step"""
        if not self.is_synchronized:
            raise Exception("GPU profiler is not synchronized")
        return self.n_envs * self.n_steps / self.get_total_rendering_gpu_time()

    def get_rendering_fps_per_env(self):
        """Get the FPS per env for the current step"""
        if not self.is_synchronized:
            raise Exception("GPU profiler is not synchronized")
        return self.n_steps / self.get_total_rendering_gpu_time()

    def get_total_rendering_gpu_time_per_env(self):
        """Get the total rendering GPU time per env for the current step in seconds"""
        if not self.is_synchronized:
            raise Exception("GPU profiler is not synchronized")
        return self.get_total_rendering_gpu_time() / self.n_envs

    def get_total_rendering_cpu_time_per_env(self):
        """Get the total rendering CPU time per env for the current step in seconds"""
        if not self.is_synchronized:
            raise Exception("GPU profiler is not synchronized")
        return self.get_total_rendering_cpu_time() / self.n_envs

    ######################## Total Performance #######################
    def get_total_gpu_time(self):
        """Calculate total GPU time across all steps in seconds"""
        if not self.is_synchronized:
            raise Exception("GPU profiler is not synchronized")
        total_time = 0.0
        for step in range(self.current_step):
            events = self.events[step]
            total_time += events["simulation_start"].elapsed_time(events["render_end"])
        return total_time / 1000.0

    def get_total_cpu_time(self):
        """Calculate total CPU time across all steps in seconds"""
        if not self.is_synchronized:
            raise Exception("GPU profiler is not synchronized")
        total_time = 0.0
        for step in range(self.current_step):
            cpu_times = self.cpu_times[step]
            total_time += (cpu_times["render_end"] - cpu_times["simulation_start"]) * 1000  # Convert to ms
        return total_time / 1000.0

    def get_total_gpu_time_per_env(self):
        """Get the total GPU time per env for the current step in seconds"""
        if not self.is_synchronized:
            raise Exception("GPU profiler is not synchronized")
        return self.get_total_gpu_time() / self.n_envs

    def get_total_cpu_time_per_env(self):
        """Get the total CPU time per env for the current step in seconds"""
        if not self.is_synchronized:
            raise Exception("GPU profiler is not synchronized")
        return self.get_total_cpu_time() / self.n_envs

    def get_step_times(self, step_idx):
        """Get detailed timing for a specific step in seconds"""
        if not self.is_synchronized:
            raise Exception("GPU profiler is not synchronized")
        if step_idx >= self.current_step:
            raise Exception(f"Step {step_idx} has not been profiled yet")

        events = self.events[step_idx]
        cpu_times = self.cpu_times[step_idx]

        return {
            "simulation": {
                "gpu_ms": events["simulation_start"].elapsed_time(events["render_start"]),
                "cpu_ms": (cpu_times["render_start"] - cpu_times["simulation_start"]) * 1000,
            },
            "rendering": {
                "gpu_ms": events["render_start"].elapsed_time(events["render_end"]),
                "cpu_ms": (cpu_times["render_end"] - cpu_times["render_start"]) * 1000,
            },
            "total": {
                "gpu_ms": events["simulation_start"].elapsed_time(events["render_end"]),
                "cpu_ms": (cpu_times["render_end"] - cpu_times["simulation_start"]) * 1000,
            },
        }

    ######################## Print Summary #######################
    def print_rendering_summary(self):
        """Print a summary of the profiler"""
        print(f"Total rendering GPU time: {self.get_total_rendering_gpu_time()} seconds")
        print(f"Total rendering CPU time: {self.get_total_rendering_cpu_time()} seconds")
        print(f"Total rendering GPU time per env: {self.get_total_rendering_gpu_time_per_env()} seconds")
        print(f"Total rendering CPU time per env: {self.get_total_rendering_cpu_time_per_env()} seconds")
        print(f"Rendering FPS: {self.get_rendering_fps()}")
        print(f"Rendering FPS per env: {self.get_rendering_fps_per_env()}")

    def print_simulation_summary(self):
        """Print a summary of the profiler"""
        print(f"Total simulation GPU time: {self.get_total_simulation_gpu_time()} seconds")
        print(f"Total simulation CPU time: {self.get_total_simulation_cpu_time()} seconds")
        print(f"Simulation FPS: {self.get_simulation_fps()}")
        print(f"Simulation FPS per env: {self.get_simulation_fps_per_env()}")

    def print_summary(self):
        """Print a summary of the profiler"""
        self.print_rendering_summary()
        self.print_simulation_summary()
