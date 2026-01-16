#!/usr/bin/env python
"""
Test script to verify diffusion policy setup is correct.
Run this to check if all components are properly installed.
"""

import sys
import os

# Setup paths like train.py does
sys.path.append(os.path.abspath(__file__ + "/../.."))

print("=" * 70)
print("DIFFUSION POLICY SETUP TEST")
print("=" * 70)

tests_passed = 0
tests_total = 0

def test_import(module_name, import_statement):
    global tests_passed, tests_total
    tests_total += 1
    try:
        exec(import_statement)
        print(f"✓ {module_name} imported successfully")
        tests_passed += 1
        return True
    except Exception as e:
        print(f"✗ {module_name} import failed: {e}")
        return False

print("\n1. Testing Core Dependencies:")
print("-" * 70)
test_import("PyTorch", "import torch")
test_import("NumPy", "import numpy")

print("\n2. Testing Isaac Gym:")
print("-" * 70)
isaac_ok = test_import("Isaac Gym", "import isaacgym")
test_import("Isaac Gym Envs", "import isaacgymenvs")

print("\n3. Testing RL Games:")
print("-" * 70)
test_import("rl_games", "import sys; sys.path.insert(0, 'legged_env/rl_games'); from rl_games.torch_runner import Runner")

print("\n4. Testing Diffusion Components:")
print("-" * 70)
test_import("Diffusion Policy", "from legged_env.envs.common.diffusion_policy import DiffusionPolicy")
test_import("Diffusion Network Builder", "from legged_env.envs.common.diffusion_network_builder import DiffusionActorCriticBuilder")

print("\n5. Testing Configuration Files:")
print("-" * 70)
tests_total += 3

config_file = "legged_env/envs/cfg/train/BipedPPODiffusion.yaml"
if os.path.exists(config_file):
    print(f"✓ BipedPPODiffusion.yaml exists")
    tests_passed += 1
else:
    print(f"✗ BipedPPODiffusion.yaml not found at {config_file}")

policy_file = "legged_env/envs/common/diffusion_policy.py"
if os.path.exists(policy_file):
    print(f"✓ diffusion_policy.py exists")
    tests_passed += 1
else:
    print(f"✗ diffusion_policy.py not found")

builder_file = "legged_env/envs/common/diffusion_network_builder.py"
if os.path.exists(builder_file):
    print(f"✓ diffusion_network_builder.py exists")
    tests_passed += 1
else:
    print(f"✗ diffusion_network_builder.py not found")

print("\n" + "=" * 70)
print(f"RESULTS: {tests_passed}/{tests_total} tests passed")
print("=" * 70)

if tests_passed == tests_total:
    print("\n✅ ALL TESTS PASSED! Diffusion policy is ready to use.")
    print("\nTo start training:")
    print("  cd legged_env/envs")
    print("  bash run.sh train_diffusion")
elif not isaac_ok:
    print("\n⚠️  Isaac Gym not found!")
    print("Make sure you activate the conda environment:")
    print("  conda activate <your_env_name>")
    print("\nOr check the conda_env.yaml file for the correct environment.")
else:
    print(f"\n⚠️  {tests_total - tests_passed} tests failed. Please check the errors above.")

sys.exit(0 if tests_passed == tests_total else 1)
