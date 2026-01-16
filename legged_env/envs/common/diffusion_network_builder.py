"""
Diffusion Network Builder for rl_games

This module provides a network builder that integrates the terrain-adaptive
diffusion policy with the rl_games framework.

Architecture:
- Actor: DiffusionPolicy (terrain-adaptive diffusion model)
- Critic: Standard MLP value network
"""

import torch
import torch.nn as nn

# Import NetworkBuilder base class
import sys
import os
# Add rl_games to path
rl_games_path = os.path.join(os.path.dirname(__file__), '../../rl_games')
if rl_games_path not in sys.path:
    sys.path.insert(0, rl_games_path)

from rl_games.algos_torch.network_builder import NetworkBuilder
from envs.common.diffusion_policy import DiffusionPolicy


class DiffusionActorCriticBuilder(NetworkBuilder):
    """
    Network builder for Diffusion Actor-Critic architecture.

    The actor uses the terrain-adaptive diffusion policy, while the critic
    uses a standard MLP architecture.
    """

    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = DiffusionActorCriticBuilder.Network(self.params, **kwargs)
        return net

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = kwargs.pop('num_seqs', 1)

            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            # Get observation dimension
            obs_dim = input_shape[0]

            # Create Diffusion Actor
            diffusion_config = self.params.get('diffusion', {})
            self.actor = DiffusionPolicy(
                obs_dim=obs_dim,
                action_dim=actions_num,
                state_latent_dim=diffusion_config.get('state_latent_dim', 64),
                terrain_latent_dim=diffusion_config.get('terrain_latent_dim', 4),
                max_diffusion_steps=diffusion_config.get('max_diffusion_steps', 40),
                min_diffusion_steps=diffusion_config.get('min_diffusion_steps', 20)
            )

            # Create Critic (standard MLP)
            mlp_config = self.params.get('mlp', {})
            mlp_units = mlp_config.get('units', [512, 256, 128])
            mlp_activation = mlp_config.get('activation', 'elu')
            mlp_initializer = mlp_config.get('initializer', {'name': 'default'})

            # Build critic MLP
            critic_layers = []
            in_size = obs_dim

            for unit in mlp_units:
                critic_layers.append(nn.Linear(in_size, unit))
                critic_layers.append(self.activations_factory.create(mlp_activation))
                in_size = unit

            critic_layers.append(nn.Linear(in_size, self.value_size))

            self.critic = nn.Sequential(*critic_layers)

            # Initialize critic weights
            mlp_init = self.init_factory.create(**mlp_initializer)
            for m in self.critic.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

            print(f"Diffusion Actor-Critic Network:")
            print(f"  Obs dim: {obs_dim}")
            print(f"  Action dim: {actions_num}")
            print(f"  Diffusion steps: {diffusion_config.get('min_diffusion_steps', 20)}-{diffusion_config.get('max_diffusion_steps', 40)}")
            print(f"  State latent: {diffusion_config.get('state_latent_dim', 64)}D")
            print(f"  Terrain latent: {diffusion_config.get('terrain_latent_dim', 4)}D")
            print(f"  Critic MLP: {mlp_units}")

        def load(self, params):
            self.separate = params.get('separate', True)
            self.params = params

        def forward(self, obs_dict):
            obs = obs_dict['obs']

            # Get actions from diffusion policy
            # forward() returns (action_mean, action_std, terrain_noise_level)
            mu, sigma, terrain_noise = self.actor(obs)

            # Get value from critic
            value = self.critic(obs)

            return {
                'logits': mu,
                'values': value,
                'sigma': sigma,
                'terrain_noise': terrain_noise
            }

        def is_separate_critic(self):
            return self.separate

        def get_value_layer(self):
            return self.critic


def create_diffusion_network_builder(**kwargs):
    """Factory function for creating diffusion network builder"""
    return DiffusionActorCriticBuilder(**kwargs)
