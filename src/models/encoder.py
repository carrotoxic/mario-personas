"""Grid+state observation encoder shared by the policy and the DRAIL discriminator."""

from __future__ import annotations

import math

import gymnasium as gym
import torch
from torch import nn


def layer_init(
    layer: nn.Module, std: float = math.sqrt(2.0), bias_const: float = 0.0
) -> nn.Module:
    """Orthogonal weight init and constant bias fill; returns the layer for inline use."""
    if hasattr(layer, "weight") and layer.weight is not None:
        nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class ResidualBlock(nn.Module):
    """Two-conv pre-activation residual block."""

    def __init__(self, channels: int):
        super().__init__()
        # Must stay named `net` (checkpoint key contract).
        self.net = nn.Sequential(
            nn.ReLU(),
            layer_init(nn.Conv2d(channels, channels, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(channels, channels, kernel_size=3, padding=1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MarioEncoder(nn.Module):
    """Encodes ``{"grid", "state"}`` Dict observations into a (B, features_dim) vector.
    Submodule names (``embed``, ``grid_torso``, ``grid_head``, ``state_head``,
    ``fusion``) are the state-dict keys of every published checkpoint."""

    def __init__(
        self, obs_space: gym.Space, max_id: int, embed_dim: int, features_dim: int
    ):
        super().__init__()
        if not isinstance(obs_space, gym.spaces.Dict):
            raise TypeError(
                "MarioEncoder expects Dict observations with 'grid' and 'state'."
            )
        self.features_dim = int(features_dim)
        grid_space = obs_space["grid"]
        h, w, c = grid_space.shape
        self.state_dim = int(obs_space["state"].shape[0])
        self.embed_dim = embed_dim
        self.max_id = max_id
        self.embed = nn.Embedding(max_id + 1, embed_dim)
        in_ch = c * embed_dim
        self.grid_torso = nn.Sequential(
            layer_init(nn.Conv2d(in_ch, 64, kernel_size=3, padding=1)),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_grid = torch.zeros(1, h, w, c, dtype=torch.long)
            dummy_embed = (
                self.embed(dummy_grid)
                .reshape(1, h, w, in_ch)
                .permute(0, 3, 1, 2)
                .float()
            )
            grid_flat = self.grid_torso(dummy_embed).shape[1]
        self.grid_head = nn.Sequential(
            layer_init(nn.Linear(grid_flat, features_dim)), nn.ReLU()
        )
        self.state_head = nn.Sequential(
            layer_init(nn.Linear(self.state_dim, 64)),
            nn.LayerNorm(64),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            layer_init(nn.Linear(features_dim + 64, features_dim)),
            nn.ReLU(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        grid = obs["grid"]
        state = obs["state"].float()
        b, h, w, c = grid.shape
        # .long() copies, so the in-place clamp never mutates the input grid.
        grid_ids = grid.long().clamp_(0, self.max_id)
        x = (
            self.embed(grid_ids)
            .reshape(b, h, w, c * self.embed_dim)
            .permute(0, 3, 1, 2)
            .float()
        )
        grid_feat = self.grid_head(self.grid_torso(x))
        state_feat = self.state_head(state)
        return self.fusion(torch.cat([grid_feat, state_feat], dim=-1))
