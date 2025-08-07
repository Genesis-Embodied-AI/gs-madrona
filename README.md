# gs-madrona — A Fork of Madrona and Madrona-MJX for Genesis

It started as a fork of [Madrona](https://github.com/shacklettbp/madrona) for the core C++ implementation, created in 2025. Since the fork, substantial modifications have been made to adapt the codebase for [Genesis simulator](https://github.com/Genesis-Embodied-AI/Genesis), including:
- Integrating Python interface based on [Madrona-MJX](https://github.com/shacklettbp/madrona_mjx) and designed for Genesis.
- Adding new features to support the continuous development. For more details, please refer to [Features Added Since Fork](#features-added-since-fork)
- Removing features not used by Genesis. For more details, please refer to [Removed Features](#removed-features)

Due to the extent of these changes, gs-madrona has diverged significantly from its origins. It is now a fully independent project with no intention of maintaining backward compatibility. Our focus is solely on evolving gs-madrona into a robust and efficient batch renderer for Genesis.

## Scope and Objectives
**gs-madrona** aims to provide a general-purpose **high-throughput batch renderer** that supports both rasterization and single-bounce ray-tracing pipelines.

It is a renderer used in Genesis to provide native support of batched processing. With gs-madrona, multiple environments and cameras can be rendered offscreen in parallel, significantly improving performance compared to renderers doing sequential processing.

At present, batch rendering supports only basic materials, lighting, and shadows. However, we aim to expand its capabilities to include more advanced rendering features.

While gs-madrona currently depends on Genesis, we plan to decouple it in the near future to support broader use cases through a more generic interface.

## Features Added Since Fork
- Support for non-square resolutions
- Shadow rendering in the rasterizer pipeline
- Anti-aliasing support for rasterizer output
- Lighting support (rasterizer and ray tracer) using Genesis-defined light sources
- Unified lighting model between rasterizer and ray tracer
- Support for spot lights and directional lights with intensity and shadow-casting flags
- Automatic mipmap generation for all textures
- CUDA kernel caching with dirty-check rebuild
- Fixed vertex normal computation in the ray tracer
- Benchmark scripts comparing Madrona with other batch renderers, including IsaacLab and ManiSkill

## Removed Features
- Legacy depth-only rendering via color buffer
- Batch rendering pipeline based on JAX

## Known Limitations
- Only color and depth outputs are currently supported
- Shadows are only cast from the first light with `castshadow=true`
- When rendering multiple cameras with different resolutions, the first camera’s resolution is used for the entire batch

## Roadmap / Future Plans
**gs-madrona** will continue evolving to support higher-quality rendering and broader functionality. Upcoming features include:
- Batch rendering support for cameras with varying resolutions
- Normal buffer and semantic/instance segmentation output
- Per-camera dynamic FOV control
- Camera-specific near/far plane configuration
- Light color specification
- Dynamic light parameters (position, direction, intensity, color, enable/disable)
- Light attenuation based on distance and angle
- Ambient lighting control (color and intensity)
- PBR material and texture support
- Output rendering results to video files

## Supported Platforms and Environments
**gs-madrona** should be compatible with any Linux distribution and Python>=3.10. However, it has been tested only with Ubuntu 22.04 and Ubuntu 24.04 for Python 3.10 and 3.12. The rendering pipeline also depends on CUDA, so an NVIDIA graphics card with CUDA 12.4+ support is required for running it. There is no plan to support Windows OS and Mac OS at the moment.

## Performance
FPS comparison of rendering [Franka](https://github.com/Genesis-Embodied-AI/Genesis/blob/main/genesis/assets/xml/franka_emika_panda/panda.xml) with gs-madrona rasterizer and raytracer

Resolution: 128x128

<p align="center">
  <img src="./scripts/perf_benchmark/example_report/panda_madrona rasterizer_ madrona raytracer_128x128_comparison_table.png" width="600" alt="FPS of gs-madrona rasterizer vs raytracer" align="center"/>
</p>

<p align="center">
  <img src="./scripts/perf_benchmark/example_report/panda_madrona rasterizer_ madrona raytracer_128x128_comparison_plot.png" width="600" alt="FPS of gs-madrona rasterizer vs raytracer" align="center"/>
</p>

## Install (Linux Only)

### Pre-requisite
Please first install the latest version of Genesis to date following the [official README instructions](https://github.com/Genesis-Embodied-AI/Genesis#quick-installation).

### Easy install (x86 only)
Pre-compiled binary wheels for Python>=3.10 are available on PyPI. They can be installed using any Python package manager (e.g. `uv` or `pip`):
```sh
pip install gs-madrona
```

### Build from source
Please first install CUDA toolkit (>= 12.4) following the [official instructions](https://developer.nvidia.com/cuda-toolkit-archive).
```sh
git clone --recurse-submodules https://github.com/Genesis-Embodied-AI/gs-madrona.git
cd gs-madrona
pip install --no-build-isolation -Cbuild-dir=build -v -e .
```

Make sure that the version of `torch` installed on your system that has been pre-compiled against a version of CUDA Toolkit that is equal or newer than the one available on your system. Otherwise, gs-madrona will crash at import (segmentation fault). Please follow the [official instructions](https://pytorch.org/get-started/locally/).

### Testing (Optional)
1. Clone Genesis Simulator repository if not already done
```sh
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
```

2. Run the following example script provided with Genesis
```sh
python Genesis/examples/rigid/single_franka_batch_render.py
```

All the generated images will be stored in the current directory under `./image_output`.

2. To use ray tracer, change the `use_rasterizer=False` in `single_franka_batch_render.py`
```
renderer = gs.options.renderers.BatchRenderer(
    use_rasterizer=True,
)
```

### Performance Benchmark
For comprehensive performance benchmarking across multiple renderers (Madrona, Omniverse, PyRender, ManiSkill), please refer to the detailed documentation in `scripts/perf_benchmark/README.md`.

The benchmark suite includes:
- Multi-renderer performance testing
- Batch size and resolution scaling tests
- Rasterizer vs raytracer comparisons
- Automated report generation
- Asset preprocessing utilities

Quick start:
```bash
cd scripts/perf_benchmark
python batch_benchmark.py -f benchmark_config_smoke_test.yml
```

## Acknowledgments

The development of gs-madrona is actively supported by [Genesis AI](https://genesis-ai.company/).
