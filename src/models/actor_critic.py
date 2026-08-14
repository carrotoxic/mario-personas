"""PPO actor-critic over the shared Mario observation encoder."""

from __future__ import annotations

import gymnasium as gym
import torch
from torch import nn
from torch.distributions.categorical import Categorical

from src.models.encoder import MarioEncoder, layer_init


class ActorCritic(nn.Module):
    """Shared-encoder actor-critic with a categorical policy head.
    Construction order encoder -> critic -> actor is load-bearing: it fixes the
    init RNG stream that published checkpoints were drawn from."""

    def __init__(
        self,
        observation_space: gym.Space,
        num_actions: int,
        *,
        max_id: int = 100,
        embed_dim: int = 16,
        features_dim: int = 1024,
        head_hidden_dim: int = 1024,
    ):
        super().__init__()
        self.encoder = MarioEncoder(
            observation_space,
            max_id=max_id,
            embed_dim=embed_dim,
            features_dim=features_dim,
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(features_dim, head_hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(head_hidden_dim, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(features_dim, head_hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(head_hidden_dim, num_actions), std=0.01),
        )

    def get_features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(obs)

    def get_value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.critic(self.get_features(obs))

    def get_action_and_value(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(action, log_prob, entropy, value)``; samples when action is None."""
        hidden = self.get_features(obs)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
