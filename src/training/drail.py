"""AIRL-style diffusion-discriminator machinery: reward composition, running
normalizers, and discriminator training. Timing contract: rewards in iteration ``k``
use the discriminator as trained through ``k - 1`` (GAE consumes pre-update rewards)."""

from __future__ import annotations

from typing import Any, NamedTuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from src.data import demonstrations
from src.models import actor_critic
from src.models import discriminator as discriminator_lib
from src.training import config as config_lib
from src.training import rollouts

# Direct DRAIL clips P(expert|s,a) barely away from {0, 1}; post-training uses
# the (clamped) ``drail_d_prob_clip_eps`` config value instead.
DIRECT_D_PROB_CLIP_EPS = 1e-8
_GAP_SAMPLE_SIZE = 512


class RunningMeanStd:
    """Running mean/variance via the parallel (Chan et al.) combine rule: float64
    statistics, an initial pseudo-count of 1e-4, and an element-wise variance
    floor of 1e-8."""

    def __init__(self, shape: tuple[int, ...] = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        """Fold a batch (leading axis) into the running statistics."""
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 0 else 1
        delta = batch_mean - self.mean
        total = self.count + batch_count
        m_2 = (
            self.var * self.count
            + batch_var * batch_count
            + delta**2 * self.count * batch_count / total
        )
        self.mean = self.mean + delta * batch_count / total
        self.var = np.maximum(m_2 / total, 1e-8)
        self.count = total

    @property
    def std(self) -> np.ndarray:
        """Standard deviation of the tracked statistics."""
        return np.sqrt(self.var)


def normalize_state(state: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardize ``state`` and clip to ±10, returning float32."""
    out = (state - mean) / (std + 1e-8)
    return np.clip(out, -10.0, 10.0).astype(np.float32)


def clamp_probability_epsilon(eps: float) -> float:
    """Clamp a discriminator-probability clip epsilon into ``[1e-6, 0.49]``."""
    return float(np.clip(float(eps), 1e-6, 0.49))


def discriminator_reward(d_prob: np.ndarray, reward_type: str) -> np.ndarray:
    """Transform (already clipped) discriminator probabilities P(expert|s,a) into rewards."""
    if reward_type == "airl":
        return np.log(d_prob) - np.log(1.0 - d_prob)
    if reward_type == "gail":
        return np.log(d_prob)
    if reward_type == "raw":
        return d_prob
    if reward_type == "airl-positive":
        return np.log(d_prob) - np.log(1.0 - d_prob) + 20.0
    if reward_type == "revise":
        log_d = np.log(d_prob)
        return log_d + (-1.0 - np.log(-log_d))
    raise ValueError(f"Unrecognized reward_type: {reward_type}")


def make_discriminator(
    obs_space: gym.Space,
    num_actions: int,
    cfg: config_lib.DrailConfig,
    device: torch.device,
) -> discriminator_lib.DiffusionDiscriminator:
    """Build the diffusion discriminator described by a DRAIL config."""
    return discriminator_lib.DiffusionDiscriminator(
        obs_space,
        max_id=int(cfg.max_id),
        obs_embed_dim=int(cfg.embed_dim),
        obs_features_dim=int(cfg.discrim_obs_features_dim),
        action_dim=num_actions,
        action_embed_dim=int(cfg.action_embed_dim),
        label_dim=int(cfg.label_dim),
        depth=int(cfg.discrim_depth),
        hidden_dim=int(cfg.discrim_hidden_dim),
        sample_strategy=str(cfg.sample_strategy),
        sample_strategy_value=int(cfg.sample_strategy_value),
        device=device,
    ).to(device)


def _disc_values(
    disc: discriminator_lib.DiffusionDiscriminator,
    obs: rollouts.ObsArrayDict,
    actions: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    return disc.compute_disc_val(
        rollouts.obs_to_torch(obs, device),
        torch.as_tensor(actions, device=device, dtype=torch.long),
    )


class DrailRewardComposer:
    """Stateful per-step reward hook for ``rollouts.collect_rollout``. Direct mode
    (``drail_lambda=None``) discards the env reward and trains on the (normalized)
    discriminator reward; post-training returns ``env + drail_lambda * drail_term``."""

    def __init__(
        self,
        disc: discriminator_lib.DiffusionDiscriminator,
        *,
        num_envs: int,
        state_dim: int,
        gamma: float,
        reward_type: str,
        reward_scale: float,
        reward_clip: float,
        d_prob_clip_eps: float,
        state_norm: bool,
        reward_norm: bool,
        drail_lambda: float | None,
        device: torch.device,
    ):
        self._disc = disc
        self._gamma = float(gamma)
        self._reward_type = str(reward_type)
        self._reward_scale = float(reward_scale)
        self._reward_clip = float(reward_clip)
        self._d_prob_clip_eps = float(d_prob_clip_eps)
        self._state_norm = bool(state_norm)
        self._reward_norm = bool(reward_norm)
        self._drail_lambda = None if drail_lambda is None else float(drail_lambda)
        self._device = device
        self.state_rms = RunningMeanStd((int(state_dim),))
        self._return_rms = RunningMeanStd(())
        self._returns = np.zeros((int(num_envs),), dtype=np.float32)
        self._level_ids = np.full(
            (int(num_envs),), demonstrations.UNKNOWN_LEVEL_ID, dtype=np.int64
        )
        self._enabled = True
        self._grids: list[np.ndarray] = []
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._levels: list[np.ndarray] = []
        self._raw_rewards: list[np.ndarray] = []
        self._terms: list[np.ndarray] = []
        self._d_prob_sum = 0.0
        self._d_prob_count = 0

    def begin_iteration(self, reward_enabled: bool) -> None:
        """Reset per-iteration buffers; persistent statistics carry over."""
        self._enabled = bool(reward_enabled)
        self._grids, self._states = [], []
        self._actions, self._levels = [], []
        self._raw_rewards, self._terms = [], []
        self._d_prob_sum = 0.0
        self._d_prob_count = 0

    def __call__(
        self,
        obs: rollouts.ObsArrayDict,
        actions: np.ndarray,
        env_reward: np.ndarray,
        dones: np.ndarray,
        infos: dict[str, Any],
    ) -> np.ndarray:
        """RewardFn hook composing this step's training reward from (s_t, a_t)."""
        self._level_ids = demonstrations.extract_level_ids_from_infos(infos, self._level_ids)
        grid = np.asarray(obs["grid"], dtype=np.int16)
        state = np.asarray(obs["state"], dtype=np.float32)
        action_ids = np.asarray(actions).reshape(-1).astype(np.int64)
        disc_state = state
        if self._state_norm:
            # Update-then-normalize, from policy states only (expert states never
            # touch the running statistics); keeps updating during warmup too.
            self.state_rms.update(state)
            disc_state = normalize_state(state, self.state_rms.mean, self.state_rms.std)
        if self._enabled:
            with torch.no_grad():
                d_prob = (
                    self._disc.compute_disc_val(
                        rollouts.obs_to_torch({"grid": grid, "state": disc_state}, self._device),
                        torch.as_tensor(action_ids, device=self._device, dtype=torch.long),
                    )
                    .cpu()
                    .numpy()
                )
            d_prob = np.clip(d_prob, self._d_prob_clip_eps, 1.0 - self._d_prob_clip_eps)
            self._d_prob_sum += float(np.sum(d_prob))
            self._d_prob_count += int(d_prob.size)
            raw_reward = discriminator_reward(d_prob, self._reward_type)
            raw_reward = np.clip(
                raw_reward * self._reward_scale, -self._reward_clip, self._reward_clip
            )
            if self._reward_norm:
                # The mask uses the current step's done flags: the return
                # accumulator is zeroed on the step an episode ends.
                masks = 1.0 - np.asarray(dones, dtype=np.float32)
                self._returns = self._returns * masks * self._gamma + raw_reward.astype(np.float32)
                self._return_rms.update(self._returns)
                drail_term = raw_reward / np.sqrt(float(self._return_rms.var) + 1e-8)
            else:
                drail_term = raw_reward
        else:
            raw_reward = np.zeros((len(action_ids),), dtype=np.float32)
            drail_term = raw_reward
        self._grids.append(grid)
        self._states.append(state)
        self._actions.append(action_ids)
        self._levels.append(self._level_ids)
        self._raw_rewards.append(np.asarray(raw_reward, dtype=np.float32))
        self._terms.append(np.asarray(drail_term, dtype=np.float32))
        if self._drail_lambda is None:
            return drail_term
        return np.asarray(env_reward, dtype=np.float32) + self._drail_lambda * drail_term

    def policy_batch(self) -> tuple[rollouts.ObsArrayDict, np.ndarray, np.ndarray]:
        """Concatenate the iteration's transitions as ``(obs, actions, level_ids)``,
        step-major, with the *raw* (unnormalized) per-step states."""
        obs = {
            "grid": np.concatenate(self._grids, axis=0),
            "state": np.concatenate(self._states, axis=0),
        }
        return obs, np.concatenate(self._actions, axis=0), np.concatenate(self._levels, axis=0)

    @property
    def mean_raw_reward(self) -> float:
        """Mean pre-normalization discriminator reward over the iteration."""
        return float(np.mean(np.concatenate(self._raw_rewards))) if self._raw_rewards else 0.0

    @property
    def mean_normalized_reward(self) -> float:
        """Mean normalized DRAIL term over the iteration."""
        return float(np.mean(np.concatenate(self._terms))) if self._terms else 0.0

    @property
    def d_prob_mean(self) -> float | None:
        """Mean clipped D-probability this iteration; None if never scored."""
        if self._d_prob_count == 0:
            return None
        return self._d_prob_sum / self._d_prob_count


class DiscriminatorUpdateResult(NamedTuple):
    """Mean BCE loss of one discriminator session and the sampled
    ``D(expert) - D(policy)`` gap before/after training."""

    loss: float
    gap_before: float
    gap_after: float


def discriminator_gap(
    disc: discriminator_lib.DiffusionDiscriminator,
    demos: demonstrations.ExpertDemonstrations,
    policy_obs: rollouts.ObsArrayDict,
    policy_actions: np.ndarray,
    policy_levels: np.ndarray,
    device: torch.device,
) -> float:
    """Mean ``D(expert) - D(policy)`` on a level-matched sample of ≤512 pairs.
    Policy states arrive normalized (end-of-rollout statistics); matched expert states
    stay raw. Stochastic per call: ``compute_disc_val`` draws fresh noise/timesteps."""
    sample_size = min(_GAP_SAMPLE_SIZE, len(policy_actions))
    with torch.no_grad():
        indices = np.random.choice(len(policy_actions), size=sample_size, replace=False)
        sampled_obs = {
            "grid": policy_obs["grid"][indices],
            "state": policy_obs["state"][indices],
        }
        expert = demonstrations.sample_matched_by_level(demos, policy_levels[indices])
        expert_val = _disc_values(disc, expert.obs, expert.actions, device).mean()
        policy_val = _disc_values(disc, sampled_obs, policy_actions[indices], device).mean()
    return float((expert_val - policy_val).item())


def update_discriminator(
    disc: discriminator_lib.DiffusionDiscriminator,
    disc_opt: optim.Optimizer,
    demos: demonstrations.ExpertDemonstrations,
    policy_obs: rollouts.ObsArrayDict,
    policy_actions: np.ndarray,
    policy_levels: np.ndarray,
    *,
    state_rms: RunningMeanStd,
    state_norm: bool,
    batch_size: int,
    num_epochs: int,
    device: torch.device,
) -> DiscriminatorUpdateResult:
    """Run one end-of-rollout discriminator training session: rebinds
    ``policy_obs["state"]`` to its normalized version (visible to the caller through
    the shared dict), then trains ``num_epochs * max(1, n // batch_size)`` minibatches."""
    # End-of-rollout running statistics: subtly different from the per-step
    # reward-time normalization of the same data.
    if state_norm:
        policy_obs["state"] = normalize_state(policy_obs["state"], state_rms.mean, state_rms.std)
    gap_before = discriminator_gap(disc, demos, policy_obs, policy_actions, policy_levels, device)

    n_policy = len(policy_actions)
    minibatch = max(1, int(batch_size))
    num_batches = max(1, n_policy // minibatch)
    losses: list[float] = []
    # Uniformly sampled policy negatives paired with level-matched expert
    # positives, summed BCE on both sides, gradient norm clipped at 1.0.
    for _ in range(int(num_epochs)):
        for _ in range(num_batches):
            indices = np.random.choice(n_policy, size=minibatch, replace=n_policy < minibatch)
            batch_obs = {
                "grid": policy_obs["grid"][indices],
                "state": policy_obs["state"][indices],
            }
            expert = demonstrations.sample_matched_by_level(demos, policy_levels[indices])
            expert_obs = expert.obs
            if state_norm:
                expert_obs = {
                    "grid": expert_obs["grid"],
                    "state": normalize_state(expert_obs["state"], state_rms.mean, state_rms.std),
                }
            disc_opt.zero_grad()
            expert_d = _disc_values(disc, expert_obs, expert.actions, device)
            policy_d = _disc_values(disc, batch_obs, policy_actions[indices], device)
            loss = F.binary_cross_entropy(
                expert_d, torch.ones_like(expert_d)
            ) + F.binary_cross_entropy(policy_d, torch.zeros_like(policy_d))
            loss.backward()
            nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            disc_opt.step()
            losses.append(float(loss.item()))
    gap_after = discriminator_gap(disc, demos, policy_obs, policy_actions, policy_levels, device)
    return DiscriminatorUpdateResult(
        loss=float(np.mean(losses)) if losses else 0.0,
        gap_before=gap_before,
        gap_after=gap_after,
    )


def post_update_policy_gap(
    disc: discriminator_lib.DiffusionDiscriminator,
    agent: actor_critic.ActorCritic,
    demos: demonstrations.ExpertDemonstrations,
    policy_obs: rollouts.ObsArrayDict,
    policy_actions: np.ndarray,
    policy_levels: np.ndarray,
    *,
    state_rms: RunningMeanStd,
    state_norm: bool,
    device: torch.device,
) -> float:
    """Post-PPO gap: ``D(expert)`` vs ``D(s, argmax pi_new(s))`` (post-training).
    Known quirk: ``policy_obs["state"]`` is already normalized by
    :func:`update_discriminator` and gets normalized a second time here (metrics only)."""
    sample_size = min(_GAP_SAMPLE_SIZE, len(policy_actions))
    with torch.no_grad():
        indices = np.random.choice(len(policy_actions), size=sample_size, replace=False)
        expert = demonstrations.sample_matched_by_level(demos, policy_levels[indices])
        expert_val = _disc_values(disc, expert.obs, expert.actions, device).mean()
        sampled_state = policy_obs["state"][indices]
        if state_norm:
            sampled_state = normalize_state(sampled_state, state_rms.mean, state_rms.std)
        obs_t = rollouts.obs_to_torch(
            {"grid": policy_obs["grid"][indices], "state": sampled_state}, device
        )
        greedy_actions = agent.actor(agent.get_features(obs_t)).argmax(dim=-1)
        policy_val = disc.compute_disc_val(obs_t, greedy_actions).mean()
    return float((expert_val - policy_val).item())
