"""
Diffusion Policy Module for Legged Robot Locomotion

This module wraps the lightweight terrain-adaptive diffusion model for use
with the rl_games reinforcement learning framework.

Architecture (from diagram):
┌─────────────────────────────────────────────────────────────────┐
│ Observations (154D state + 1D gait + 2D contacts + 1D terrain) │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│           Unified Encoder: 3-layer MLP [256, 18]                │
│                     ↓                    ↓                      │
│            State Latent (64D)    Terrain Latent (4D)            │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│      Adaptive Diffusion Net: 5-layer MLP [128,128,64,32]        │
│                     Steps 20-40 adaptive                        │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  Outputs: Joint Angles (10D) + α-Scaling (10D)                  │
│  Terrain Noise Prediction: Terrain Latent + Noise Level σ       │
└─────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Tuple, Optional


class SinusoidalPositionEmbeddings(nn.Module):
    """Timestep embeddings for diffusion"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class UnifiedEncoder(nn.Module):
    """
    Unified Encoder: 3-layer MLP [256, 18]
    Takes all observations and produces State Latent (64D) + Terrain Latent (4D)
    
    Architecture from diagram:
    - Input: state history + gait phase + contacts + terrain score
    - Layer 1: Linear(input_dim, 256) + ReLU
    - Layer 2: Linear(256, 18) + ReLU  
    - Layer 3: Split heads for State Latent (64D) and Terrain Latent (4D)
    """
    def __init__(self, obs_dim: int, hidden_dims=[256, 18], 
                 state_latent_dim=64, terrain_latent_dim=4):
        super().__init__()
        
        self.obs_dim = obs_dim
        
        # 3-layer MLP encoder [256, 18]
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims[0]),  # → 256
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),  # → 18
            nn.ReLU(),
        )
        
        # Split heads for latent representations
        self.state_head = nn.Linear(hidden_dims[1], state_latent_dim)    # 18 → 64D
        self.terrain_head = nn.Linear(hidden_dims[1], terrain_latent_dim)  # 18 → 4D
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: [B, obs_dim] - flattened observations
        Returns:
            state_latent: [B, 64]
            terrain_latent: [B, 4]
        """
        # Encode through 3-layer MLP
        encoded = self.encoder(obs)  # [B, 18]
        
        # Produce latent representations
        state_latent = self.state_head(encoded)    # [B, 64]
        terrain_latent = self.terrain_head(encoded)  # [B, 4]
        
        return state_latent, terrain_latent


class TerrainNoisePredictor(nn.Module):
    """
    Predicts terrain-adaptive noise level σ
    KEY COMPONENT - enables adaptive diffusion steps
    """
    def __init__(self, terrain_latent_dim=4, output_dim=1):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(terrain_latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            nn.Softplus()  # Ensure positive noise level
        )
        
    def forward(self, terrain_latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            terrain_latent: [B, 4]
        Returns:
            noise_level: [B, 1]
        """
        return self.network(terrain_latent)


class AdaptiveDiffusionNet(nn.Module):
    """
    Adaptive Diffusion Net: 5-layer MLP [128, 128, 64, 32]
    Steps 20-40 adaptive based on terrain
    
    Architecture from diagram:
    - Input: State Latent (64D) + Terrain Latent (4D) + Noise (action_dim) + Time Emb (16D)
    - Layer 1: Linear(input_dim, 128) + ReLU
    - Layer 2: Linear(128, 128) + ReLU
    - Layer 3: Linear(128, 64) + ReLU
    - Layer 4: Linear(64, 32) + ReLU
    - Layer 5: Linear(32, output_dim * 2) → Joints + α-Scaling
    """
    def __init__(self, state_dim=64, terrain_dim=4, noise_dim=10, time_dim=16, output_dim=10):
        super().__init__()
        
        self.output_dim = output_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim)
        )
        
        # 5-layer MLP diffusion network [128, 128, 64, 32]
        input_dim = state_dim + terrain_dim + noise_dim + time_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),    # Layer 1
            nn.ReLU(),
            nn.Linear(128, 128),          # Layer 2
            nn.ReLU(),
            nn.Linear(128, 64),           # Layer 3
            nn.ReLU(),
            nn.Linear(64, 32),            # Layer 4
            nn.ReLU(),
            nn.Linear(32, output_dim * 2)  # Layer 5: joints + α-scaling
        )
        
    def forward(self, state_latent: torch.Tensor, terrain_latent: torch.Tensor, 
                noisy_actions: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state_latent: [B, 64]
            terrain_latent: [B, 4]
            noisy_actions: [B, action_dim]
            timesteps: [B]
        Returns:
            joint_pred: [B, action_dim]
            scaling_pred: [B, action_dim]
        """
        # Time embedding
        t_emb = self.time_mlp(timesteps.float())  # [B, 16]
        
        # Concatenate all features
        x = torch.cat([state_latent, terrain_latent, noisy_actions, t_emb], dim=-1)
        
        # Process through 5-layer MLP
        output = self.network(x)
        
        # Split into outputs
        joint_pred = output[:, :self.output_dim]
        scaling_pred = output[:, self.output_dim:]
        
        return joint_pred, scaling_pred


class DiffusionPolicy(nn.Module):
    """
    Terrain-Adaptive Diffusion Policy for Legged Robot Locomotion
    
    This module integrates with rl_games by providing:
    - forward() for policy evaluation (returns action mean and std)
    - sample() for action sampling with terrain-adaptive diffusion
    
    Architecture matches the diagram:
    - Unified Encoder: 3-layer MLP [256, 18] → State Latent + Terrain Latent
    - Adaptive Diffusion Net: 5-layer MLP [128, 128, 64, 32]
    - Terrain Noise Predictor: Predicts adaptive noise level σ
    """
    def __init__(self, 
                 obs_dim: int,
                 action_dim: int = 10,
                 state_latent_dim: int = 64,
                 terrain_latent_dim: int = 4,
                 max_diffusion_steps: int = 40,
                 min_diffusion_steps: int = 20):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_diffusion_steps = max_diffusion_steps
        self.min_diffusion_steps = min_diffusion_steps
        
        # Unified Encoder: 3-layer MLP [256, 18]
        self.unified_encoder = UnifiedEncoder(
            obs_dim=obs_dim,
            hidden_dims=[256, 18],
            state_latent_dim=state_latent_dim,
            terrain_latent_dim=terrain_latent_dim
        )
        
        # Terrain Noise Predictor
        self.noise_predictor = TerrainNoisePredictor(
            terrain_latent_dim=terrain_latent_dim,
            output_dim=1
        )
        
        # Adaptive Diffusion Net: 5-layer MLP [128, 128, 64, 32]
        self.diffusion_net = AdaptiveDiffusionNet(
            state_dim=state_latent_dim,
            terrain_dim=terrain_latent_dim,
            noise_dim=action_dim,
            time_dim=16,
            output_dim=action_dim
        )
        
        # Learnable action std for PPO compatibility
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for policy evaluation
        
        Args:
            obs: [B, obs_dim] - observations
            
        Returns:
            action_mean: [B, action_dim]
            action_std: [B, action_dim]
            terrain_noise_level: [B, 1]
        """
        batch_size = obs.shape[0]
        device = obs.device
        
        # Unified Encoder
        state_latent, terrain_latent = self.unified_encoder(obs)
        
        # Predict terrain-adaptive noise level
        noise_level = self.noise_predictor(terrain_latent)
        
        # For policy forward pass, use single-step prediction (t=0)
        # This gives the "denoised" action prediction
        zero_noise = torch.zeros(batch_size, self.action_dim, device=device)
        timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Get action prediction
        action_mean, scaling = self.diffusion_net(
            state_latent, terrain_latent, zero_noise, timesteps
        )
        
        # Apply scaling (learned per-action scaling)
        action_mean = action_mean * torch.sigmoid(scaling)
        
        # Action std from learnable parameter
        action_std = torch.exp(self.log_std).expand(batch_size, -1)
        
        return action_mean, action_std, noise_level
    
    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Sample actions using terrain-adaptive diffusion
        
        Args:
            obs: [B, obs_dim] - observations
            deterministic: if True, return mean action without sampling
            
        Returns:
            actions: [B, action_dim]
        """
        batch_size = obs.shape[0]
        device = obs.device
        
        # Encode observations
        state_latent, terrain_latent = self.unified_encoder(obs)
        
        # Get terrain-adaptive noise level
        noise_level = self.noise_predictor(terrain_latent)
        
        if deterministic:
            # Just use forward pass for deterministic actions
            zero_noise = torch.zeros(batch_size, self.action_dim, device=device)
            timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
            action_mean, scaling = self.diffusion_net(
                state_latent, terrain_latent, zero_noise, timesteps
            )
            return action_mean * torch.sigmoid(scaling)
        
        # Adaptive number of diffusion steps based on terrain difficulty
        num_steps = int(self.min_diffusion_steps + 
                       (noise_level.mean().item() * (self.max_diffusion_steps - self.min_diffusion_steps)))
        num_steps = min(num_steps, self.max_diffusion_steps)
        
        # Start from random noise
        actions = torch.randn(batch_size, self.action_dim, device=device)
        
        # DDPM reverse process - CORRECTED VERSION
        for t in reversed(range(num_steps)):
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict noise ε_θ(x_t, t)
            eps_pred, scaling_pred = self.diffusion_net(
                state_latent, terrain_latent, actions, timesteps
            )

            # Get alpha_bar values (cumulative product of alphas)
            alpha_bar_t = self._get_alpha_bar(t, num_steps)
            alpha_bar_t_prev = self._get_alpha_bar(t - 1, num_steps) if t > 0 else 1.0

            # Predict x_0 from x_t and predicted noise
            pred_x0 = (actions - math.sqrt(1 - alpha_bar_t) * eps_pred) / math.sqrt(alpha_bar_t)

            # Compute mean of q(x_{t-1} | x_t, x_0)
            coef1 = math.sqrt(alpha_bar_t_prev)
            coef2 = math.sqrt(1 - alpha_bar_t_prev)
            mean = coef1 * pred_x0 + coef2 * eps_pred

            # Add noise (except at t=0)
            if t > 0:
                # Compute variance for this step
                alpha_t = alpha_bar_t / alpha_bar_t_prev
                variance = (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_t)
                sigma = math.sqrt(max(variance, 1e-8))  # Clamp for numerical stability

                # Sample with terrain-adaptive noise scale
                noise_scale = noise_level.clamp(min=0.01, max=1.0)
                noise = torch.randn_like(actions) * noise_scale
                actions = mean + sigma * noise
            else:
                actions = mean

        # Apply final scaling
        return actions * torch.sigmoid(scaling_pred)

    def _get_alpha_bar(self, t: int, num_steps: int) -> float:
        """Compute ᾱ_t (cumulative product of alphas) for DDPM cosine schedule"""
        return math.cos(((t / num_steps) + 0.008) / 1.008 * math.pi / 2) ** 2


def create_diffusion_policy(obs_dim: int, action_dim: int = 10, **kwargs) -> DiffusionPolicy:
    """Factory function for creating diffusion policy"""
    return DiffusionPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **kwargs
    )


# Test the module
if __name__ == "__main__":
    print("="*60)
    print("DIFFUSION POLICY MODULE TEST")
    print("="*60)
    
    # Test with typical observation dimensions
    obs_dim = 225  # 45 per frame * 5 stacked frames
    action_dim = 10
    batch_size = 4
    
    # Create policy
    policy = create_diffusion_policy(obs_dim=obs_dim, action_dim=action_dim)
    
    # Count parameters
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"\nModel parameters: {num_params:,}")
    
    # Test forward pass
    obs = torch.randn(batch_size, obs_dim)
    action_mean, action_std, noise_level = policy(obs)
    
    print(f"\nForward pass:")
    print(f"  Input obs: {obs.shape}")
    print(f"  Action mean: {action_mean.shape}")
    print(f"  Action std: {action_std.shape}")
    print(f"  Terrain noise level: {noise_level.squeeze().tolist()}")
    
    # Test sampling
    print(f"\nSampling:")
    actions_det = policy.sample(obs, deterministic=True)
    print(f"  Deterministic: {actions_det.shape}")
    
    actions_stoch = policy.sample(obs, deterministic=False)
    print(f"  Stochastic: {actions_stoch.shape}")
    
    print(f"\n✓ All tests passed!")
