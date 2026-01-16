# Terrain-Adaptive Diffusion Policy Guide

## Overview

This project implements a **terrain-adaptive diffusion policy** alongside vanilla RL (PPO, SAC, A2C) for legged robot locomotion in Isaac Gym. The diffusion model uses DDPM (Denoising Diffusion Probabilistic Models) with terrain-adaptive noise levels to generate robust robot actions across diverse terrains.

---

## Architecture

### Diffusion Policy Components

```
┌─────────────────────────────────────────────────────────────────┐
│ Observations (158D: state + gait + contacts + terrain)         │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│           Unified Encoder: 3-layer MLP [256, 18]                │
│                     ↓                    ↓                      │
│            State Latent (64D)    Terrain Latent (4D)            │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│      Terrain Noise Predictor: Predicts σ (noise level)          │
│      Enables adaptive diffusion steps (20-40)                   │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│      Adaptive Diffusion Net: 5-layer MLP [128,128,64,32]        │
│      DDPM reverse process with cosine schedule                  │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  Outputs: Joint Angles (10D) + α-Scaling (10D)                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features

1. **Terrain Adaptation**: Noise level adapts to terrain difficulty (20-40 diffusion steps)
2. **Unified Encoder**: Processes all observations into state and terrain latents
3. **DDPM Sampling**: Full denoising process during inference
4. **Single-Step Training**: Uses t=0 prediction for policy gradient
5. **PPO Compatible**: Integrates seamlessly with PPO training

---

## Quick Start

### 1. Training Diffusion Policy

```bash
cd /home/jaggu/Difussion/legged_env/envs

# Quick start - uses all default settings
bash run.sh train_diffusion

# Or use specific terrain configurations:

# Train on flat/plane terrain (easier, 2000 epochs)
bash run.sh dukehumanoid_diffusion_plain

# Train on rough terrain (harder, 1500 epochs)
bash run.sh dukehumanoid_diffusion_rough

# Train on stairs (2000 epochs)
bash run.sh dukehumanoid_diffusion_stairs

# Full mixed terrain training (5000 epochs)
bash run.sh dukehumanoid_diffusion
```

### 2. Playing/Testing Trained Policy

```bash
# Play with the latest checkpoint (auto-detected)
bash run.sh train_diffusion -p

# Or specific configurations
bash run.sh dukehumanoid_diffusion_plain -p
bash run.sh dukehumanoid_diffusion_rough -p
bash run.sh dukehumanoid_diffusion_stairs -p
```

### 3. Keyboard Control During Play

```bash
# Enable keyboard control for manual commands
bash run.sh train_diffusion -pk
```

---

## Training Configurations

### Available Diffusion Training Modes

| Command | Terrain Type | Curriculum | Max Epochs | Difficulty | Use Case |
|---------|-------------|-----------|-----------|-----------|----------|
| `train_diffusion` | Mixed (trimesh) | Yes | 5000 | Medium | General purpose |
| `dukehumanoid_diffusion_plain` | Plane | No | 2000 | Easy | Flat ground testing |
| `dukehumanoid_diffusion_rough` | Trimesh | Yes | 1500 | Hard | Rough obstacles |
| `dukehumanoid_diffusion_stairs` | Trimesh | Yes | 2000 | Medium | Stair climbing |

### Key Hyperparameters

**Diffusion Settings** (from `BipedPPODiffusion.yaml`):
```yaml
diffusion:
  state_latent_dim: 64          # State representation size
  terrain_latent_dim: 4         # Terrain representation size
  max_diffusion_steps: 40       # Max denoising steps (hard terrain)
  min_diffusion_steps: 20       # Min denoising steps (easy terrain)
  encoder_hidden_dims: [256, 18]
  diffusion_hidden_dims: [128, 128, 64, 32]
```

**PPO Settings**:
```yaml
horizon_length: 32
mini_epochs: 8
max_epochs: 5000
learning_rate: 3e-4
```

---

## Configuration Files

### Main Files

1. **Diffusion Policy**: `legged_env/envs/common/diffusion_policy.py`
   - Core diffusion model implementation
   - DiffusionPolicy, UnifiedEncoder, AdaptiveDiffusionNet classes

2. **Network Builder**: `legged_env/envs/common/diffusion_network_builder.py`
   - Integrates diffusion policy with rl_games
   - DiffusionActorCriticBuilder class

3. **Training Config**: `legged_env/envs/cfg/train/BipedPPODiffusion.yaml`
   - Diffusion hyperparameters
   - Network architecture settings

4. **Task Config**: `legged_env/envs/cfg/task/Biped.yaml`
   - Environment settings
   - Observation/action spaces
   - Reward function parameters

5. **Experiment Script**: `legged_env/envs/exp.sh`
   - Pre-configured training commands
   - Terrain-specific settings

### Registration

The diffusion network is registered in `train.py`:
```python
# At line 128
from envs.common import diffusion_network_builder

# At line 239
model_builder.register_network('diffusion_actor_critic',
    lambda **kwargs: diffusion_network_builder.DiffusionActorCriticBuilder(**kwargs))
```

---

## Checkpoints and Outputs

### Directory Structure

```
outputs/
└── Biped/
    └── train/
        ├── dukehumanoid_diffusion/
        │   └── runs/
        │       └── BipedDiffusion_DD-HH-MM-SS/
        │           ├── nn/                    # Model checkpoints
        │           │   └── BipedDiffusion.pth
        │           ├── config.yaml            # Full config
        │           └── summaries/             # TensorBoard logs
        ├── dukehumanoid_diffusion_plain/
        ├── dukehumanoid_diffusion_rough/
        └── dukehumanoid_diffusion_stairs/
```

### Checkpoint Auto-Detection

The `exp.sh` script automatically finds the latest checkpoint:
```bash
# Automatically uses latest .pth file
checkpoint=$(get_latest_checkpoint dukehumanoid_diffusion)
```

---

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard to monitor training
cd /home/jaggu/Difussion/legged_env/envs
tensorboard --logdir=../outputs/Biped/train/dukehumanoid_diffusion/runs

# Then open: http://localhost:6006
```

### Key Metrics to Watch

See the **Verification Guide** section below for detailed interpretation.

---

## Comparing with Vanilla RL

### Training Vanilla PPO (for comparison)

```bash
# Standard PPO baseline
bash run.sh dukehumanoid_baseline

# PPO with asymmetric observations (critic has more info)
bash run.sh dukehumanoid_baseline -p
```

### Key Differences

| Aspect | Vanilla PPO | Diffusion PPO |
|--------|------------|---------------|
| **Policy Type** | Direct MLP | Diffusion denoising |
| **Action Generation** | Single forward pass | 20-40 iterative steps |
| **Terrain Adaptation** | Implicit (learned) | Explicit (noise predictor) |
| **Network Size** | [512, 256, 128] MLP | Encoder + Diffusion Net |
| **Training Config** | `BipedPPOAsymm.yaml` | `BipedPPODiffusion.yaml` |
| **Inference Speed** | ~1ms | ~5-10ms (multi-step) |
| **Sample Efficiency** | Standard | Potentially better |
| **Robustness** | Good | Potentially superior |

---

## Advanced Usage

### Custom Diffusion Training

Create your own experiment in `exp.sh`:

```bash
my_custom_diffusion(){
    dukehumanoid_diffusion  # Inherit from base

    # Override specific settings
    BASE_ARGS+=(
        # Modify diffusion parameters
        train.params.network.diffusion.max_diffusion_steps=50
        train.params.network.diffusion.min_diffusion_steps=15

        # Change terrain
        task.env.terrain.terrainType=heightfield
        task.env.terrain.difficultySale=0.5

        # Adjust training
        train.params.config.max_epochs=3000
        train.params.config.learning_rate=1e-4

        # Custom output directory
        hydra.run.dir=../outputs/\${task_name}/\${test}/my_custom_diffusion
    )
}
```

Then run:
```bash
bash run.sh my_custom_diffusion
```

### Modifying Diffusion Architecture

Edit `legged_env/envs/cfg/train/BipedPPODiffusion.yaml`:

```yaml
params:
  network:
    diffusion:
      state_latent_dim: 128        # Increase capacity
      terrain_latent_dim: 8        # More terrain info
      max_diffusion_steps: 60      # More refinement steps
      diffusion_hidden_dims: [256, 256, 128, 64]  # Deeper network
```

### Terrain Configuration

Modify terrain proportions in `exp.sh`:
```bash
# Terrain types: [smooth_slope, rough_slope, stairs_up, stairs_down,
#                 discrete, stepping_stones, gaps, pit, flat]
task.env.terrain.terrainProportions=[1,1,2,1,1,1,1,2,0]
```

---

## Troubleshooting

### Common Issues

1. **Import Error: `diffusion_network_builder` not found**
   - Ensure you're in the correct directory: `legged_env/envs`
   - Check that `diffusion_network_builder.py` exists in `envs/common/`

2. **Config not found: `BipedPPODiffusion.yaml`**
   - Verify file exists at `envs/cfg/train/BipedPPODiffusion.yaml`
   - Check for typos in experiment name

3. **Checkpoint not loading**
   - Check path in `outputs/Biped/train/<experiment>/runs/`
   - Ensure `.pth` file exists and is not corrupted

4. **Training crashes with CUDA OOM**
   - Reduce `num_envs` in `exp.sh` (default: 4096)
   - Reduce network size in YAML config
   - Use smaller diffusion steps

5. **Low rewards / Not learning**
   - Check reward scaling in task config
   - Verify terrain difficulty isn't too high initially
   - Enable curriculum learning: `task.env.terrain.curriculum=true`

---

## Next Steps

1. **Verify Training**: See next section for detailed verification guide
2. **Compare Performance**: Train both vanilla PPO and diffusion
3. **Visualize**: Use TensorBoard to compare reward curves
4. **Deploy**: Export trained policy for real robot (if applicable)

---

# Verification Guide: How to Verify Diffusion Training

## 1. Pre-Training Verification

### Check Configuration

```bash
# Run in dry-run mode to see the full command
cd /home/jaggu/Difussion/legged_env/envs
bash run.sh train_diffusion -r

# Expected output should include:
# - train=BipedPPODiffusion
# - task=Biped
# - All diffusion parameters
```

### Verify Network Registration

```bash
# Quick test - should not error
cd /home/jaggu/Difussion/legged_env/envs
python -c "from envs.common.diffusion_network_builder import DiffusionActorCriticBuilder; print('✓ Network builder imported successfully')"

python -c "from envs.common.diffusion_policy import DiffusionPolicy; print('✓ Diffusion policy imported successfully')"
```

---

## 2. During Training Verification

### Monitor TensorBoard

```bash
# Start TensorBoard (in a separate terminal)
cd /home/jaggu/Difussion/legged_env/envs
tensorboard --logdir=../outputs/Biped/train/dukehumanoid_diffusion/runs
```

Open browser to `http://localhost:6006`

### Key Metrics to Watch

#### A. Reward Metrics

**Graph: `rewards/mean`**
- **What to expect**: Steady increase from ~0-50 (initial) to 200-400+ (good)
- **Timeline**:
  - Epochs 0-500: Random exploration, rewards 0-100
  - Epochs 500-2000: Learning locomotion, rewards 100-200
  - Epochs 2000+: Refinement, rewards 200-400+
- **Red flags**:
  - Reward stuck at 0-50 after 1000 epochs → Check reward scaling
  - Sudden drops → Curriculum too aggressive or instability
  - Oscillating wildly → Reduce learning rate

**Example of good progression:**
```
Epoch    Mean Reward    Max Reward    Min Reward
------   -----------    ----------    ----------
100      23.5           45.2          -12.1
500      87.3           142.5         21.4
1000     156.8          234.7         89.2
2000     287.4          389.1         178.5
3000     342.9          456.3         245.7
5000     398.2          512.8         298.4
```

#### B. Loss Metrics

**Graph: `losses/a_loss` (Actor Loss)**
- **What to expect**: High initially (>1.0), decreases to 0.01-0.1
- **Healthy pattern**: Gradual decrease with small oscillations
- **Red flags**:
  - Stays above 1.0 for >2000 epochs
  - Goes to 0 (collapsed policy)
  - Explodes (>10.0) → Gradient clipping issue

**Graph: `losses/c_loss` (Critic Loss)**
- **What to expect**: Similar to actor loss, 0.1-1.0 range after convergence
- **Pattern**: Should decrease alongside rewards

**Graph: `losses/entropy`**
- **What to expect**: Starts high (~2.0-3.0), decreases to 0.5-1.5
- **Purpose**: Encourages exploration early, exploitation later
- **Red flags**:
  - Drops to <0.1 too early → Policy becoming deterministic prematurely
  - Stays >2.5 → Not learning, still random

#### C. Performance Metrics

**Graph: `performance/fps`**
- **What to expect**:
  - With diffusion: 2000-8000 FPS (depends on GPU)
  - With vanilla PPO: 5000-15000 FPS
- **Note**: Diffusion is slower due to iterative denoising

**Graph: `info/lr` (Learning Rate)**
- **What to expect**:
  - Starts at 3e-4 (0.0003)
  - May decrease if adaptive scheduling is enabled
  - Should not be 0

#### D. Diffusion-Specific Metrics

**Custom metric: `terrain_noise_level`** (if logged)
- **What to expect**: Values between 0-1
- **Interpretation**:
  - Low values (0-0.3): Easy terrain, fewer diffusion steps
  - High values (0.7-1.0): Hard terrain, more diffusion steps

---

## 3. Interpreting Reward/Iteration Graph

### Good Training Curve Example

```
Reward
  |
400|                                    ╱──────────
   |                                ╱───
350|                            ╱───
   |                        ╱───
300|                    ╱───
   |                ╱───
250|            ╱───
   |        ╱───
200|    ╱───
   |╱───
100|───────────────────────────────────────────> Iterations
   0    500   1000  1500  2000  2500  3000  3500

Phase 1: Exploration (0-500)
Phase 2: Learning (500-2000)
Phase 3: Refinement (2000+)
```

### Bad Training Curves

**❌ Not Learning (Flat Line)**
```
Reward
100|──────────────────────────────────────
 50|
  0|──────────────────────────────────────> Iterations
```
**Fix**:
- Check reward scaling is appropriate
- Verify environment is reset correctly
- Check if observations are normalized

**❌ Unstable (High Variance)**
```
Reward
   |  ╱╲    ╱╲    ╱╲    ╱╲
200|╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲
   |    ╲╱    ╲╱    ╲╱    ╲╱
  0|──────────────────────────────────────> Iterations
```
**Fix**:
- Reduce learning rate (try 1e-4)
- Increase minibatch size
- Check for reward explosion bugs

**❌ Collapse (Sudden Drop)**
```
Reward
   |        ╱──╲
200|    ╱───    ╲
   |╱───         ╲___________________
  0|──────────────────────────────────────> Iterations
```
**Fix**:
- Reduce learning rate
- Check gradient clipping
- Verify curriculum isn't too aggressive

---

## 4. Validation Tests

### Qualitative Tests (Visual Inspection)

```bash
# Play with trained model and observe
bash run.sh train_diffusion -p

# Things to check:
# ✓ Robot walks forward smoothly
# ✓ Maintains balance on terrain
# ✓ Recovers from pushes
# ✓ Adapts gait to terrain changes
# ✓ No knee dragging or falling

# ✗ Red flags:
# - Falls immediately
# - Stuck in place
# - Erratic movements
# - Only works on flat terrain
```

### Quantitative Tests

**A. Checkpoint Performance**
```bash
# Test on plane terrain
bash run.sh dukehumanoid_diffusion_plain -p
# Expected: Robot walks smoothly, reward >300

# Test on rough terrain
bash run.sh dukehumanoid_diffusion_rough -p
# Expected: Robot navigates obstacles, reward >200

# Test on stairs
bash run.sh dukehumanoid_diffusion_stairs -p
# Expected: Robot climbs stairs, reward >250
```

**B. Compare with Vanilla PPO**
```bash
# Train vanilla PPO for comparison
bash run.sh dukehumanoid_baseline

# After training, compare TensorBoard curves:
# - Final reward: Diffusion vs PPO
# - Sample efficiency: Epochs to reach reward=200
# - Robustness: Performance on new terrains
```

---

## 5. Success Criteria

### Minimum Viable Performance

After **2000 epochs** on mixed terrain:
- ✓ Mean reward > 150
- ✓ Max reward > 300
- ✓ Robot walks forward at commanded velocity
- ✓ Doesn't fall on flat terrain
- ✓ Actor loss < 0.5

### Good Performance

After **5000 epochs**:
- ✓ Mean reward > 300
- ✓ Max reward > 450
- ✓ Walks on rough terrain without falling
- ✓ Climbs moderate stairs
- ✓ Recovers from external pushes
- ✓ Actor loss < 0.1

### Excellent Performance

- ✓ Mean reward > 400
- ✓ Handles all terrain types
- ✓ Robust to disturbances
- ✓ Smooth, natural-looking gait
- ✓ Generalizes to unseen terrains

---

## 6. Debugging Poor Performance

### Step-by-Step Debugging

1. **Check logs for errors**
   ```bash
   # Check if training is actually using diffusion
   tail -100 <training_output.log> | grep -i "diffusion"
   # Should see: "Diffusion Actor-Critic Network"
   ```

2. **Verify observations are not NaN**
   - Check TensorBoard for NaN values
   - Monitor `info/obs_mean` and `info/obs_std`

3. **Test on easy terrain first**
   ```bash
   # Start with plane, no curriculum
   bash run.sh dukehumanoid_diffusion_plain
   ```

4. **Reduce complexity**
   - Decrease diffusion steps (min=10, max=20)
   - Simplify terrain (plane only)
   - Reduce number of environments (num_envs=1024)

5. **Compare with working baseline**
   - Train vanilla PPO to ensure environment is working
   - If vanilla PPO also fails → Issue is in environment/rewards
   - If only diffusion fails → Issue is in diffusion implementation

---

## 7. Expected Training Timeline

### Full Training Run (5000 epochs)

**Hardware**: 1x RTX 3090 / RTX 4090
**Environments**: 4096 parallel envs

| Metric | Value |
|--------|-------|
| Total time | ~6-10 hours |
| Time per epoch | 4-7 seconds |
| FPS | 3000-7000 |
| Peak GPU memory | 8-12 GB |
| Checkpoint size | ~50-100 MB |

**Training phases:**
- **Epochs 0-500** (30-60 min): Random exploration, reward 0-100
- **Epochs 500-2000** (2-3 hours): Learning locomotion, reward 100-250
- **Epochs 2000-5000** (4-6 hours): Refinement, reward 250-400+

---

## 8. Comparing Results

### Create Comparison Plot

After training both models, compare in TensorBoard:

```bash
# Load both runs
tensorboard --logdir_spec \
  vanilla:outputs/Biped/train/dukehumanoid_baseline/runs,\
  diffusion:outputs/Biped/train/dukehumanoid_diffusion/runs
```

### Metrics to Compare

| Metric | Vanilla PPO | Diffusion PPO | Winner |
|--------|------------|---------------|--------|
| Final Mean Reward | ??? | ??? | Higher is better |
| Sample Efficiency | Epochs to 200 | Epochs to 200 | Fewer is better |
| Robustness | Test on new terrain | Test on new terrain | Subjective |
| Training Speed | FPS | FPS | Higher is better |
| Inference Speed | ~1ms | ~5-10ms | Faster is better |

---

## 9. Exporting Results

### Save Best Checkpoint

```bash
# Checkpoints are automatically saved every N epochs
# Find best checkpoint by monitoring TensorBoard

# Best checkpoint location:
# outputs/Biped/train/dukehumanoid_diffusion/runs/BipedDiffusion_<date>/nn/BipedDiffusion.pth
```

### Create Video

```bash
# Enable video capture (requires opencv)
bash run.sh train_diffusion -p \
  task.env.dataPublisher.enable=true \
  capture_video=true \
  capture_video_freq=100 \
  capture_video_len=500
```

---

## 10. Final Checklist

Before concluding training is successful:

- [ ] Mean reward > 300 after 5000 epochs
- [ ] TensorBoard shows smooth learning curve
- [ ] Actor/Critic losses have converged (<0.2)
- [ ] Robot walks smoothly on plane terrain
- [ ] Robot navigates rough terrain without falling
- [ ] Checkpoint file exists and loads correctly
- [ ] Compared with vanilla PPO (optional but recommended)
- [ ] Documented hyperparameters and results
- [ ] Saved best checkpoint with clear naming

---

## Additional Resources

- **Diffusion Policy Paper**: [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- **DDPM Paper**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- **Isaac Gym**: [NVIDIA Isaac Gym Docs](https://developer.nvidia.com/isaac-gym)
- **rl_games**: [rl_games GitHub](https://github.com/Denys88/rl_games)

---

**Good luck with your diffusion policy training!** 🚀

If you encounter issues not covered in this guide, check:
1. Console logs for error messages
2. TensorBoard for metric anomalies
3. Checkpoint files for corruption
4. Environment configuration for mismatches
