#!/bin/bash
echo "=== Diffusion Policy Setup Check ==="
echo ""

echo "1. Checking conda environment..."
if command -v conda &> /dev/null; then
    echo "✓ Conda is installed"
    conda info | grep "active environment" || echo "⚠ No environment active"
else
    echo "✗ Conda not found"
fi
echo ""

echo "2. Checking Python..."
which python
python --version
echo ""

echo "3. Checking Isaac Gym..."
python -c "import isaacgym; print('✓ Isaac Gym found')" 2>&1 || echo "✗ Isaac Gym not found"
echo ""

echo "4. Checking diffusion files..."
ls -lh legged_env/envs/common/diffusion*.py 2>&1 || echo "✗ Diffusion files not found"
echo ""

echo "5. Checking config..."
ls -lh legged_env/envs/cfg/train/BipedPPODiffusion.yaml 2>&1 || echo "✗ Config not found"
echo ""

echo "6. Testing diffusion import..."
cd legged_env/envs
python -c "from envs.common.diffusion_policy import DiffusionPolicy; print('✓ Diffusion policy imports correctly')" 2>&1
echo ""

echo "=== End of Check ==="
