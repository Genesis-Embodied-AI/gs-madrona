# Performance Benchmark Suite

This directory contains a comprehensive performance benchmarking suite for evaluating rendering and simulation performance across different renderers, configurations, and hardware setups.

## Overview

The benchmark suite is designed to systematically test performance across multiple dimensions:
- **Renderers**: Madrona, Omniverse, PyRender, ManiSkill
- **Rendering Modes**: Rasterizer vs Raytracer
- **Batch Sizes**: Multiple environment counts
- **Resolutions**: Various image resolutions
- **Assets**: Different MJCF/URDF models

## Directory Structure

```
perf_benchmark/
├── README.md                           # This file
├── batch_benchmark.py                  # Main batch execution script
├── benchmark_configs.py                # Configuration parser
├── benchmark_report_generator.py       # Report and visualization generator
├── benchmark_profiler.py               # Performance profiling utilities
├── benchmark_madrona.py                # Madrona renderer benchmark
├── benchmark_omni.py                   # Omniverse renderer benchmark
├── benchmark_pyrender.py               # PyRender benchmark
├── benchmark_maniskill.py              # ManiSkill renderer benchmark
├── process_xml.py                      # XML asset preprocessing utility
├── configs/                            # Configuration files
│   ├── benchmark_config_smoke_test.yml # Quick test configuration
│   ├── benchmark_config_madrona.yml    # Madrona-specific config
│   ├── benchmark_config_omni.yml       # Omniverse-specific config
│   └── benchmark_config_full.yml       # Comprehensive test config
```

## Quick Start


### 1. Optional steps
IsaacLab and Maniskill needs to be installed if they need to be benchmarked.

Install IsaacLab
-  Download and install IsaacLab from https://developer.nvidia.com/isaac-sim
-  Add IsaacLab to your PATH:

Install Maniskill
-  Install ManiSkill2 following the [official instructions](https://github.com/haosulab/ManiSkill2)

### 2. Run a Quick Smoke Test

```bash
python batch_benchmark.py -f benchmark_config_smoke_test.yml
```

### 3. Run a Full Benchmark Suite

```bash
python batch_benchmark.py -f benchmark_config_full.yml
```

### 4. Continue from a Previous Run

```bash
python batch_benchmark.py -f benchmark_config_full.yml -c /path/to/previous/perf_data.csv
```

### 5. Preprocess MUJUCO XML Assets to make it compatible with Omniverse (if needed)

```bash
python process_xml.py --file ./genesis/assets/xml/franka_emika_panda/panda.xml
```

## Configuration Files

Configuration files are YAML-based and define the test parameters. Here's an example structure:

```yaml
# List of MJCF/URDF files to test
mjcf_list:
  - xml/franka_emika_panda/panda.xml

# Renderer configurations
renderer_list:
  - renderer: madrona
    benchmark_script: benchmark_madrona.py
    timeout: 120
  - renderer: omniverse
    benchmark_script: benchmark_omni.py
    timeout: 300
  - renderer: maniskill
    benchmark_script: benchmark_maniskill.py
    timeout: 180

# Test rasterizer and raytracer modes
rasterizer_list:
  - true   # Rasterizer
  - false  # Raytracer

# Batch sizes to test
batch_size_list:
  - 256
  - 512
  - 1024

# Resolutions to test (width x height)
resolution_list:
  - [128, 128]
  - [256, 256]
  - [512, 512]

# Raytracer settings
raytracer:
  max_bounce: 2
  spp: 1

# Simulation settings
simulation:
  n_steps: 1000

# Camera settings
camera:
  position: [1.5, 0.5, 1.5]
  lookat: [0.0, 0.0, 0.5]
  fov: 45.0

# Display settings
display:
  gui: false

# Performance comparisons to generate
comparison_list:
  - - renderer: madrona
      rasterizer: true
    - renderer: madrona
      rasterizer: false
```

## Core Components

### batch_benchmark.py

The main orchestration script that:
- Parses configuration files
- Creates test combinations
- Executes benchmarks in parallel
- Handles failures and timeouts
- Generates reports

**Key Features:**
- Hierarchical test execution (renderer → rasterizer → mjcf → batch_size → resolution)
- Automatic failure handling (skips larger resolutions if smaller ones fail)
- Resume capability from previous runs
- Timeout management per renderer

### benchmark_configs.py

Configuration parser that loads and validates YAML configuration files.

**Supported Configuration Sections:**
- `mjcf_list`: List of asset files to test
- `renderer_list`: Renderer configurations with timeouts
- `rasterizer_list`: Boolean flags for rasterizer/raytracer modes
- `batch_size_list`: Environment counts to test
- `resolution_list`: Image resolutions to test
- `raytracer`: Raytracing parameters (max_bounce, spp)
- `simulation`: Simulation parameters (n_steps)
- `camera`: Camera positioning and settings
- `display`: Display options (gui)
- `comparison_list`: Performance comparison definitions

### benchmark_profiler.py

High-precision performance profiling using CUDA events and CPU timing.

**Profiling Capabilities:**
- GPU timing using CUDA events
- CPU timing using high-resolution timers
- Per-step detailed timing
- Per-environment timing calculations
- FPS calculations (total and per-environment)

**Key Methods:**
- `on_simulation_start()`: Start simulation timing
- `on_rendering_start()`: Start rendering timing
- `on_rendering_end()`: End rendering timing
- `get_total_rendering_gpu_time()`: Total GPU rendering time
- `get_rendering_fps()`: Overall FPS
- `get_rendering_fps_per_env()`: FPS per environment

### benchmark_report_generator.py

Generates comprehensive performance reports and visualizations.

**Output Types:**
- Individual performance plots per MJCF/rasterizer combination
- Comparison plots between renderers
- HTML report with embedded plots and tables
- Performance summary tables

**Visualization Features:**
- FPS vs batch size plots
- Resolution-based color coding
- Performance comparison charts
- Interactive HTML reports

### process_xml.py

Utility script for preprocessing MJCF/XML assets to ensure compatibility with different renderers.

**Features:**
- Wraps visual geometry elements in body tags
- Generates unique identifiers for mesh elements
- Handles collision geometry appropriately
- Creates processed XML files with `_new.xml` suffix

**Usage:**
```bash
python process_xml.py --file ./genesis/assets/xml/franka_emika_panda/panda.xml
```

## Renderer-Specific Scripts

### benchmark_madrona.py

Benchmarks the Madrona renderer using the Genesis framework.

**Features:**
- GPU/CPU fallback handling
- Batch rendering support
- Configurable camera and lighting
- Performance profiling integration

### benchmark_omni.py

Benchmarks the Omniverse renderer using Isaac Sim.

**Features:**
- Omniverse-specific settings optimization
- Path tracing and rasterizer modes
- GPU memory management
- Performance-optimized rendering settings

### benchmark_pyrender.py

Benchmarks the PyRender renderer.

**Features:**
- OpenGL-based rendering
- CPU-based simulation
- Cross-platform compatibility

### benchmark_maniskill.py

Benchmarks the ManiSkill renderer using the ManiSkill framework.

**Features:**
- Custom ManiSkill environment implementation
- Support for multiple robot types (Panda, Unitree Go2, Unitree G1)
- Configurable camera modes (minimal for rasterizer, rt-fast for raytracer)
- GPU memory optimization for large batch sizes
- Image saving capabilities for debugging

**Supported Robots:**
- `panda.xml` → Panda robot
- `go2.xml` → Unitree Go2 robot  
- `g1.xml` → Unitree G1 robot

**Camera Modes:**
- `minimal`: Fast rasterizer mode
- `rt-fast`: Fast raytracer mode

## Output Structure

Benchmark runs create the following structure:

```
benchmark_reports/
└── perf_benchmark_YYYYMMDD_HHMMSS/
    ├── perf_data.csv              # Raw benchmark data
    ├── plots/                     # Generated plots
    │   ├── panda_plot.png
    │   ├── panda_comparison_plot.png
    │   └── ...
    ├── images/                    # Rendered images (ManiSkill)
    │   ├── step00_env00_minimal_panda.png
    │   └── ...
    └── report.html                # Interactive HTML report
```

### CSV Data Format

The `perf_data.csv` file contains the following columns:
- `result`: "succeeded" or "failed"
- `mjcf`: Asset file path
- `renderer`: Renderer name
- `rasterizer`: Boolean rasterizer flag
- `n_envs`: Number of environments
- `n_steps`: Number of simulation steps
- `resX`, `resY`: Image resolution
- `camera_posX/Y/Z`: Camera position
- `camera_lookatX/Y/Z`: Camera lookat point
- `camera_fov`: Camera field of view
- `time_taken_gpu`: Total GPU time (seconds)
- `time_taken_per_env_gpu`: GPU time per environment (seconds)
- `time_taken_cpu`: Total CPU time (seconds)
- `time_taken_per_env_cpu`: CPU time per environment (seconds)
- `fps`: Overall FPS
- `fps_per_env`: FPS per environment

## Advanced Usage

### Custom Configuration

Create a custom configuration file:

```yaml
# custom_config.yml
mjcf_list:
  - path/to/your/model.xml

renderer_list:
  - renderer: madrona
    benchmark_script: benchmark_madrona.py
    timeout: 180
  - renderer: maniskill
    benchmark_script: benchmark_maniskill.py
    timeout: 240

rasterizer_list: [true, false]
batch_size_list: [64, 128, 256]
resolution_list: [[256, 256], [512, 512]]

raytracer:
  max_bounce: 4
  spp: 2

simulation:
  n_steps: 500

camera:
  position: [2.0, 1.0, 2.0]
  lookat: [0.0, 0.0, 0.0]
  fov: 60.0
```

Run with custom config:
```bash
python batch_benchmark.py -f custom_config.yml
```

### Asset Preprocessing

Some renderers may require specific XML formatting. Use the preprocessing utility:

```bash
# Process a single file
python process_xml.py --file ./genesis/assets/xml/franka_emika_panda/panda.xml

# Process multiple files
for file in ./genesis/assets/xml/*/*.xml; do
    python process_xml.py --file "$file"
done
```

### Performance Analysis

The benchmark suite provides detailed performance analysis:

1. **Raw Data**: CSV files with detailed timing information
2. **Visualizations**: Plots showing FPS vs batch size relationships
3. **Comparisons**: Side-by-side renderer performance comparisons
4. **HTML Reports**: Interactive reports with embedded plots and tables
5. **Rendered Images**: Sample images for visual verification (ManiSkill)

### Failure Handling

The benchmark suite includes robust failure handling:

- **Timeout Management**: Each renderer has configurable timeouts
- **Progressive Skipping**: If a resolution fails, larger resolutions are skipped
- **Resume Capability**: Can continue from previous runs
- **Error Logging**: Failed runs are recorded with "failed" status

## Dependencies

### Required Python Packages
- `pandas`: Data analysis and CSV handling
- `matplotlib`: Plot generation
- `numpy`: Numerical computations
- `pyyaml`: Configuration file parsing
- `torch`: CUDA event handling (for profiling)

### Renderer-Specific Dependencies
- **Madrona**: Genesis framework
- **Omniverse**: Isaac Sim, Omniverse Kit
- **PyRender**: PyRender, OpenGL
- **ManiSkill**: ManiSkill, SAPIEN, Gymnasium

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch sizes or resolutions
2. **Timeout Errors**: Increase timeout values in configuration
3. **Missing Assets**: Ensure MJCF/URDF files are in the correct paths
4. **Renderer Failures**: Check renderer-specific dependencies
5. **XML Compatibility**: Use `process_xml.py` to preprocess assets if needed

### Debug Mode

Enable GUI mode for debugging:
```yaml
display:
  gui: true
```

### Verbose Logging

Check the console output for detailed timing information and error messages.

### ManiSkill-Specific Issues

- **Robot Compatibility**: Ensure robot XML files match supported robot types
- **Memory Configuration**: Adjust GPU memory settings for large batch sizes
- **Image Saving**: Check image output directory permissions

## Contributing

To add a new renderer:

1. Create a new benchmark script (e.g., `benchmark_newrenderer.py`)
2. Implement the required interface (see `benchmark_madrona.py` for reference)
3. Add the renderer to your configuration file
4. Test with the smoke test configuration first

## License

This benchmark suite is part of the Genesis project. See the main LICENSE file for details. 