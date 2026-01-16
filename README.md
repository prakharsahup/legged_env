# Legged Environment with Diffusion Policy - Complete Guide

This project implements a bipedal robot locomotion environment using Isaac Gym with a diffusion-based policy architecture for terrain-adaptive control based on the BiRoDiff paper (arXiv:2407.05424).

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Training Policies](#training-policies)
- [Running Policies (Play Mode)](#running-policies-play-mode)
- [Available Experiments](#available-experiments)
- [Performance Optimization](#performance-optimization)
- [Terrain Configuration](#terrain-configuration)
- [Troubleshooting](#troubleshooting)

## Requirements

### System Requirements
- **OS:** Ubuntu 22.04 (tested)
- **GPU:** NVIDIA GPU with CUDA 12.1 or higher
- **Python:** 3.8
- **IDE:** Visual Studio Code (recommended)

### Environment Management
- **Micromamba** (recommended) or Conda

## Installation

### Step 1: Install Micromamba

If you haven't installed Micromamba yet:

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

Recommended installation location: `~/repo/micromamba`

### Step 2: Create Python Environment

```bash
# Set up alias for convenience
alias conda="micromamba"

# Navigate to the project
cd /path/to/legged_env

# Create environment from YAML file
conda env create --file envs/setup/conda_env.yaml -y

# Activate the environment
conda activate py38

# Export library path (IMPORTANT!)
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib
```

**Pro Tip:** Add this to your `~/.bashrc` to auto-activate the environment:
```bash
alias conda="micromamba"
alias activate_py38="conda activate py38 && export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib"
```

### Step 3: Install Isaac Gym

```bash
# Create a repo directory
mkdir -p ~/repo && cd ~/repo

# Clone Isaac Gym (custom fork with fixes)
git clone https://github.com/boxiXia/isaacgym.git
cd ~/repo/isaacgym/python

# Install Isaac Gym
conda activate py38 && export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib
python -m pip install -e .
```

**Verify Installation:**
```bash
cd ~/repo/isaacgym/python/examples
python interop_torch.py
```

**If you get "crypt.h: No such file or directory" error:**
```bash
cp /usr/include/crypt.h ${CONDA_PREFIX}/include/python3.8/
```

### Step 4: Install rl_games

```bash
cd ~/repo
git clone https://github.com/Denys88/rl_games.git
cd ~/repo/rl_games/

# Optional: Remove version control from setup.py for easier installation
python -m pip install -e .
```

### Step 5: Install IsaacGymEnvs

```bash
cd ~/repo
git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git
cd ~/repo/IsaacGymEnvs

# Optional: Modify setup.py to remove version control
python -m pip install -e .
```

### Step 6: Verify Installation

Test that everything is set up correctly:

```bash
# Activate environment
conda activate py38 && export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib

# Navigate to the project
cd /path/to/legged_env/envs

# Run a dry run (prints command without executing)
bash run.sh dukehumanoid_baseline -d
```

## Getting Started

### Always Activate Environment First

Before running any commands, ensure the environment is activated:

```bash
conda activate py38 && export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib
cd /path/to/legged_env/envs
```

## Training Policies

### Basic Training Command

```bash
cd envs
bash run.sh <experiment_name>
```

### Training Examples

#### 1. Train Diffusion Policy on Rough Terrain
```bash
bash run.sh dukehumanoid_diffusion_rough
```
- **Duration:** ~5000 epochs
- **Output:** `outputs/dukehumanoid_diffusion_rough/runs/*/nn/*.pth`
- **Checkpoints saved every:** 50 epochs

#### 2. Train Diffusion Policy on Stairs
```bash
bash run.sh dukehumanoid_diffusion_stairs
```

#### 3. Train Baseline (Non-Diffusion) Policy
```bash
bash run.sh dukehumanoid_baseline
```

### Training Tips

- **Monitor training:** Checkpoints are saved in `outputs/<experiment_name>/runs/*/nn/`
- **Resume training:** Automatically resumes from the latest checkpoint
- **Early stopping:** Press `Ctrl+C` to stop training gracefully
- **Tensorboard:** Logs are saved in the same directory as checkpoints

## Running Policies (Play Mode)

### Basic Play Command

```bash
cd envs
bash run.sh <experiment_name> -p
```

### Play Mode Examples

#### Run Trained Diffusion Policy on Rough Terrain
```bash
bash run.sh dukehumanoid_diffusion_rough -p
```

#### Run with Keyboard Control (WASD)
```bash
bash run.sh dukehumanoid_baseline -pk
```
- `W` - Forward
- `A` - Turn left
- `S` - Backward
- `D` - Turn right

### Play Mode Options

- `-p` - Play mode (inference)
- `-pk` - Play mode with keyboard control
- `-d` - Dry run (print command without executing)
- `-pd` - Play mode with debugging enabled

## Debugging

### Remote Debugging with VSCode

1. Start debugging session:
```bash
bash run.sh dukehumanoid_baseline -pd  # Play mode + debug
```

2. In VSCode:
   - Press `Ctrl+Shift+D`
   - Select "Python Debugger: Remote Attach"
   - Click "Start Debugging"

## Quick Start

### Training
```bash
cd envs
bash run.sh dukehumanoid_diffusion_rough
```

### Play Mode (Inference)
```bash
cd envs
bash run.sh dukehumanoid_diffusion_rough -p
```

## Available Experiments

### Diffusion Policy Experiments

#### 1. **dukehumanoid_diffusion** (Base)
Training on mixed terrain with diffusion policy architecture.
```bash
bash run.sh dukehumanoid_diffusion
```

#### 2. **dukehumanoid_diffusion_plain**
Training on flat terrain for baseline performance.
```bash
bash run.sh dukehumanoid_diffusion_plain
bash run.sh dukehumanoid_diffusion_plain -p  # Play mode
```

#### 3. **dukehumanoid_diffusion_rough**
Training on rough terrain (slopes, discrete obstacles, stepping stones, gaps).
- **Terrain:** Trimesh with rough slopes emphasized
- **Difficulty:** 0.5 (challenging)
- **No stairs**
```bash
bash run.sh dukehumanoid_diffusion_rough
bash run.sh dukehumanoid_diffusion_rough -p
```

#### 4. **dukehumanoid_diffusion_stairs**
Training on staircase terrain (both ascending and descending).
- **Terrain:** Trimesh stairs
- **Proportions:** Equal stairs up/down only
```bash
bash run.sh dukehumanoid_diffusion_stairs
bash run.sh dukehumanoid_diffusion_stairs -p
```

### Baseline Experiments (Non-Diffusion)

#### 5. **dukehumanoid_baseline**
Standard PPO with asymmetric actor-critic.
```bash
bash run.sh dukehumanoid_baseline
bash run.sh dukehumanoid_baseline -p
```

#### 6. **dukehumanoid_rough**
Non-diffusion policy trained on rough heightfield terrain.
```bash
bash run.sh dukehumanoid_rough
bash run.sh dukehumanoid_rough -p
```

#### 7. **dukehumanoid_stairs**
Non-diffusion policy for stairs.
```bash
bash run.sh dukehumanoid_stairs
bash run.sh dukehumanoid_stairs -p
```

## Performance Optimization Tips

### If you experience low FPS or crashes:

1. **Disable dataPublisher** (already configured in play mode)
2. **Reduce terrain complexity:**
   - `numLevels`: Reduce from 8 to 2-3
   - `numTerrains`: Reduce from 10 to 2-3
3. **Reduce number of environments:**
   - Set `num_envs=1` in play mode
4. **Lower render FPS:**
   - Current: `renderFPS=50`
   - Can reduce to 30 if needed

### Current Optimized Settings (Play Mode)
- `num_envs=1` - Single environment for stability
- `dataPublisher.enable=false` - Prevents crashes
- `renderFPS=50` - Smooth visualization
- Small terrain grids for better performance

## Terrain Types

### Trimesh Terrain
Used for: Stairs, complex obstacles
- Generates mesh-based terrain with precise geometry
- Better for stairs and discrete obstacles
- More computationally expensive

### Heightfield Terrain  
Used for: Smooth slopes, rough terrain
- Generates height-based terrain
- Faster rendering
- Better for continuous surfaces

## Terrain Proportions
Format: `[smooth_slope, rough_slope, stairs_up, stairs_down, discrete, stepping_stones, gaps, pit, flat]`

Examples:
- **Rough terrain (no stairs):** `[0,2,0,0,1,1,1,0,0]`
- **Stairs only:** `[0,0,1,1,0,0,0,0,0]`
- **Mixed:** `[1,1,2,1,1,1,1,2,0]`

## Checkpoints

Checkpoints are automatically loaded from:
```
outputs/{experiment_name}/runs/*/nn/*.pth
```

The `get_latest_checkpoint` function finds the most recent checkpoint for each experiment.

## Troubleshooting

### Issue: "checkpoint=null" or no checkpoint found
**Solution:** Train the experiment first before running play mode.

### Issue: Crashes with "terminate called without an active exception"
**Solution:** 
- Disable dataPublisher: `task.env.dataPublisher.enable=false`
- Reduce num_envs to 1

### Issue: Low FPS / visual lag
**Solution:**
- Check `renderFPS` setting (should be 50-60)
- Reduce `numTerrains` and `numLevels`
- Set `num_envs=1`

### Issue: Terrain looks wrong (rough instead of stairs, etc.)
**Solution:**
- Check `PLAY_ARGS` vs `BASE_ARGS` in `exp.sh`
- Ensure terrain proportions match desired terrain type
- Verify `terrainType` is set correctly (trimesh vs heightfield)

## Architecture Overview

The diffusion policy uses:
- **Unified Encoder:** Processes observations into state (64D) and terrain (4D) latents
- **Terrain Noise Predictor:** Estimates terrain difficulty
- **Adaptive Diffusion Net:** Generates actions via iterative denoising
  - Adapts diffusion steps based on terrain difficulty (20-40 steps)
  - Uses time embeddings and terrain-conditioned denoising

## Key Configuration Files

- `envs/exp.sh` - Experiment definitions and configurations
- `envs/run.sh` - Main entry point script
- `envs/cfg/config.yaml` - Base configuration
- `envs/cfg/train/BipedPPODiffusion.yaml` - Diffusion policy training config

## Training Parameters

### Diffusion Experiments
- **Max epochs:** 5000
- **Horizon length:** 32
- **Mini epochs:** 8
- **Learning rate:** 0.0003 (adaptive)
- **Architecture:** 
  - Actor: Diffusion policy (64D state + 4D terrain latents)
  - Critic: MLP [512, 256, 128]

## References

- **Paper:** BiRoDiff - Bidirectional Robotic Diffusion Policy (arXiv:2407.05424)
- **Framework:** Isaac Gym, rl_games (PPO)
