"""DRAIL diffusion discriminator and its conditional MLP noise model."""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn.functional as F
from torch import nn

from src.models.encoder import MarioEncoder


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Clipped cosine noise schedule of Nichol & Dhariwal (2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class MLPConditionDiffusion(nn.Module):
    """Conditional MLP noise predictor with per-block timestep embeddings.
    Depth ``d`` runs ``max(d - 1, 1)`` hidden Linear/ReLU blocks before the output
    projection; one timestep embedding is added per block."""

    def __init__(
        self,
        n_steps: int,
        cond_dim: int,
        data_dim: int,
        num_units: int = 128,
        depth: int = 4,
    ):
        super().__init__()
        self.data_dim = data_dim
        n_blocks = max(depth - 1, 1)
        linears_list: list[nn.Module] = [
            nn.Linear(cond_dim + data_dim, num_units),
            nn.ReLU(),
        ]
        for _ in range(n_blocks - 1):
            linears_list.append(nn.Linear(num_units, num_units))
            linears_list.append(nn.ReLU())
        linears_list.append(nn.Linear(num_units, data_dim))
        self.linears = nn.ModuleList(linears_list)
        self.step_embeddings = nn.ModuleList(
            [nn.Embedding(n_steps, num_units) for _ in range(n_blocks)]
        )

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Predicts the noise added to ``x`` given condition ``c`` at step ``t``."""
        if x.shape[0] != c.shape[0]:
            raise ValueError(
                f"Batch size mismatch: x has {x.shape[0]}, c has {c.shape[0]}"
            )
        if x.shape[0] == 0:
            return torch.zeros((0, self.data_dim), device=x.device)
        x = torch.cat([x, c], dim=1)
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)
        x = self.linears[-1](x)
        return x


class DiffusionDiscriminator(nn.Module):
    """DRAIL discriminator: P(expert | s, a) from conditional diffusion losses.
    Own small MarioEncoder + action embedding; ``compute_disc_val`` is stochastic
    (fresh noise/timesteps every call, even under ``no_grad``)."""

    def __init__(
        self,
        obs_space: gym.Space,
        max_id: int,
        obs_embed_dim: int,
        obs_features_dim: int,
        action_dim: int,
        action_embed_dim: int,
        label_dim: int,
        depth: int,
        hidden_dim: int,
        sample_strategy: str,
        sample_strategy_value: int,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.obs_encoder = MarioEncoder(
            obs_space,
            max_id=max_id,
            embed_dim=obs_embed_dim,
            features_dim=obs_features_dim,
        )
        self.action_dim = int(action_dim)
        self.action_embed = nn.Embedding(self.action_dim, int(action_embed_dim))
        self.label_dim = int(label_dim)
        self.sample_strategy = str(sample_strategy)
        self.sample_strategy_value = int(sample_strategy_value)
        self.n_steps = 1000  # diffusion timesteps; fixed value published checkpoints assume
        betas = cosine_beta_schedule(self.n_steps).to(self.device)
        alphas_prod = torch.cumprod(1 - betas, 0)
        # Deliberately NOT buffers: the schedule tensors are absent from every
        # published state_dict and stay pinned to the constructor device.
        self.alphas_bar_sqrt = torch.sqrt(alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
        self.sa_dim = obs_features_dim + int(action_embed_dim)
        self.diffusion_model = MLPConditionDiffusion(
            self.n_steps,
            self.label_dim,
            self.sa_dim,
            num_units=hidden_dim,
            depth=depth,
        ).to(self.device)

    def _sa_pair(
        self, obs: dict[str, torch.Tensor], action_ids: torch.Tensor
    ) -> torch.Tensor:
        """Encodes a step batch into (B, sa_dim) state-action features."""
        if obs["grid"].ndim != 4 or obs["state"].ndim != 2 or action_ids.ndim != 1:
            raise ValueError(
                "Discriminator expects step inputs: grid(B,H,W,C), state(B,S), "
                "action(B,)."
            )
        obs_feat = self.obs_encoder(obs)
        act_feat = self.action_embed(action_ids.long())
        return torch.cat([obs_feat, act_feat], dim=-1)

    def diffusion_loss(self, label: float, sa_pair: torch.Tensor) -> torch.Tensor:
        """Per-sample (B, 1) noise-prediction MSE for one label branch.
        The default "random" strategy samples timesteps antithetically:
        ``B // 2`` uniform draws paired with mirrors ``n_steps - 1 - t``."""
        b = sa_pair.shape[0]
        if b == 0:
            return torch.zeros((0, 1), device=self.device)
        if self.sample_strategy == "constant":
            step = min(max(0, self.sample_strategy_value), self.n_steps - 1)
            t = torch.full((b,), step, device=self.device, dtype=torch.long)
        else:
            n_samples = max(1, b // 2)
            t = torch.randint(0, self.n_steps, size=(n_samples,), device=self.device)
            t = torch.cat([t, self.n_steps - 1 - t], dim=0)
            if len(t) > b:
                t = t[:b]
            elif len(t) < b:
                t = torch.cat([t, t[: b - len(t)]], dim=0)
        a = self.alphas_bar_sqrt[t].unsqueeze(-1)
        aml = self.one_minus_alphas_bar_sqrt[t].unsqueeze(-1)
        label_in = torch.full((b, self.label_dim), float(label), device=self.device)
        e = torch.randn_like(sa_pair)
        x = sa_pair * a + e * aml
        out = self.diffusion_model(x, label_in, t)
        return (e - out).square().mean(dim=1, keepdim=True)

    def compute_disc_val(
        self, obs: dict[str, torch.Tensor], action_ids: torch.Tensor
    ) -> torch.Tensor:
        """Returns P(expert | s, a) of shape (B,) from two independent loss draws."""
        sa = self._sa_pair(obs, action_ids)
        l1 = self.diffusion_loss(1.0, sa)
        l0 = self.diffusion_loss(0.0, sa)
        stacked = torch.stack([-l1, -l0], dim=0)
        return F.softmax(stacked, dim=0)[0].squeeze(-1)
