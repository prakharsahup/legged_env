# Fix Current Error - Step by Step

## The Error

If you're seeing:
```
ModuleNotFoundError: No module named 'isaacgym'
```

This means **the conda environment is not activated**.

---

## The Fix (3 Steps)

### Step 1: Activate Conda Environment

```bash
conda activate py38
```

**Verify it worked**:
```bash
# You should see "(py38)" at the beginning of your prompt
# Also check:
conda info | grep "active environment"
# Should show: active environment : py38
```

### Step 2: Navigate to Training Directory

```bash
cd /home/jaggu/Difussion/legged_env/envs
```

### Step 3: Run Training

```bash
# For training:
bash run.sh train_diffusion

# For testing/playing:
bash run.sh train_diffusion -p
```

---

## Expected Output

When training starts correctly, you should see:

```
Loading experiment: train_diffusion
Starting diffusion policy training with terrain-adaptive model...

[Isaac Gym initialization messages...]

Diffusion Actor-Critic Network:
  Obs dim: 225
  Action dim: 10
  Diffusion steps: 20-40
  State latent: 64D
  Terrain latent: 4D
  Critic MLP: [512, 256, 128]

[Training begins with epoch updates...]
```

---

## If Still Not Working

### Test 1: Verify Environment

```bash
conda activate py38
which python
# Should show: /home/jaggu/miniconda3/envs/py38/bin/python

python -c "import isaacgym; print('✓ Isaac Gym loaded')"
# Should print: ✓ Isaac Gym loaded
```

### Test 2: Run Verification Script

```bash
cd /home/jaggu/Difussion
conda activate py38
python test_diffusion_setup.py
```

Should show:
```
======================================================================
DIFFUSION POLICY SETUP TEST
======================================================================
...
RESULTS: X/X tests passed
✅ ALL TESTS PASSED! Diffusion policy is ready to use.
```

### Test 3: Try Vanilla PPO First

If diffusion still doesn't work, test that the base system works:

```bash
conda activate py38
cd /home/jaggu/Difussion/legged_env/envs
bash run.sh dukehumanoid_baseline
```

If vanilla PPO works but diffusion doesn't, there's an issue with the diffusion integration.
If vanilla PPO also fails, there's an issue with the Isaac Gym/environment setup.

---

## Complete Setup from Scratch

If nothing works, here's the complete setup:

### 1. Create/Update Conda Environment

```bash
cd /home/jaggu/Difussion/legged_env/envs/setup

# If environment doesn't exist:
conda env create -f conda_env.yaml

# If environment exists but incomplete:
conda activate py38
conda env update -f conda_env.yaml
```

### 2. Install Isaac Gym

```bash
cd /home/jaggu/Difussion/legged_env/isaacgym/python
conda activate py38
pip install -e .
```

### 3. Install Isaac Gym Envs

```bash
cd /home/jaggu/Difussion/legged_env/IsaacGymEnvs
conda activate py38
pip install -e .
```

### 4. Install rl_games

```bash
cd /home/jaggu/Difussion/legged_env/rl_games
conda activate py38
pip install -e .
```

### 5. Test Everything

```bash
cd /home/jaggu/Difussion
conda activate py38
python test_diffusion_setup.py
```

### 6. Try Training

```bash
cd /home/jaggu/Difussion/legged_env/envs
conda activate py38
bash run.sh train_diffusion
```

---

## Quick Debug Commands

```bash
# Check if conda environment exists
conda env list | grep py38

# Check if in correct directory
pwd
# Should show: /home/jaggu/Difussion/legged_env/envs

# Check if files exist
ls -lh common/diffusion*.py
ls -lh cfg/train/BipedPPODiffusion.yaml

# Check if training script has diffusion registered
grep -n "diffusion_network_builder" train.py
# Should show line 128 (import) and line 239 (registration)

# Test python imports directly
conda activate py38
python -c "from envs.common.diffusion_policy import DiffusionPolicy; print('OK')"
```

---

## Most Common Mistakes

1. ❌ **Not activating conda environment**
   - ✅ Always run `conda activate py38` first

2. ❌ **Wrong directory**
   - ✅ Must be in `/home/jaggu/Difussion/legged_env/envs`

3. ❌ **Using wrong Python**
   - ✅ Check with `which python` - should be in miniconda3/envs/py38

4. ❌ **Isaac Gym not installed**
   - ✅ Follow complete setup from scratch above

5. ❌ **Typos in command**
   - ✅ Use exact command: `bash run.sh train_diffusion`

---

## Success Indicators

You know it's working when:

✓ No `ModuleNotFoundError`
✓ See "Diffusion Actor-Critic Network:" in output
✓ Training epochs start counting up
✓ TensorBoard shows reward increasing
✓ No Python exceptions or crashes

---

## Still Stuck?

Share the following information:

```bash
# System info
conda activate py38
python --version
which python
conda list | grep -E "torch|isaac"

# Error output
bash run.sh train_diffusion 2>&1 | tee error.log
# Then share error.log

# File verification
ls -lh common/diffusion_policy.py
ls -lh common/diffusion_network_builder.py
ls -lh cfg/train/BipedPPODiffusion.yaml
```

---

## Summary

**The #1 fix for most errors**:
```bash
conda activate py38
```

**Then run**:
```bash
cd /home/jaggu/Difussion/legged_env/envs
bash run.sh train_diffusion
```

That's it! 🚀
