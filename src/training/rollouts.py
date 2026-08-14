"""Vectorized rollout collection and advantage estimation for the PPO core.
Shared by the persona PPO and DRAIL trainers; per-variant reward composition is
injected through a ``reward_fn`` callback so there is exactly one rollout loop."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.models import actor_critic
from src.training import config

ObsArrayDict = dict[str, np.ndarray]
ObsTensorDict = dict[str, torch.Tensor]

RewardFn = Callable[
    [ObsArrayDict, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]],
    np.ndarray,
]
"""Maps ``(obs_before_step, actions, env_rewards, dones, infos)`` per vector step to
training rewards. ``obs_before_step`` is the observation the actions were computed
from (the AIRL-aligned ``s_t``); ``dones`` flags terminal-or-truncated transitions."""


class RolloutBatch(NamedTuple):
    """One rollout's tensors, each shaped ``(num_steps, num_envs, ...)``.
    ``dones[t]`` marks that ``obs[t]`` follows a reset; actions are stored as floats
    and cast to long at update time."""

    obs: ObsTensorDict
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor


def obs_to_torch(obs: ObsArrayDict, device: torch.device) -> ObsTensorDict:
    """Convert a batched Dict observation to device tensors (grid keeps its int dtype)."""
    return {
        "grid": torch.as_tensor(obs["grid"], device=device),
        "state": torch.as_tensor(obs["state"], device=device, dtype=torch.float32),
    }


def index_obs(obs: ObsTensorDict, inds: Any) -> ObsTensorDict:
    """Fancy-index every entry of a Dict observation batch."""
    return {key: value[inds] for key, value in obs.items()}


def log_episode_infos(
    writer: SummaryWriter,
    infos: dict[str, Any],
    global_step: int,
    episode_log_interval: int,
) -> None:
    """Write episodic charts, handling both Gymnasium vector-info formats: the
    ``final_info`` list-of-dicts format (gymnasium < 1.0) and the aggregate-array
    format with ``_<key>`` masks; ``charts/episodic_win`` only exists in the first."""
    if not infos:
        return
    # global_step is always a multiple of num_envs, so this gate only ever fires
    # when the interval is divisible by num_envs; a non-positive interval disables it.
    if episode_log_interval > 0 and global_step % episode_log_interval != 0:
        return

    def log_masked(tag: str, key: str) -> None:
        values = infos.get(key)
        if values is None:
            return
        mask = np.asarray(
            infos.get(f"_{key}", np.ones_like(values, dtype=bool)), dtype=bool
        )
        for value, done in zip(np.asarray(values), mask):
            if done:
                writer.add_scalar(f"charts/{tag}", float(value), global_step)

    if "final_info" in infos:
        for info in infos["final_info"]:
            if not info:
                continue
            episode = info.get("episode")
            if episode is not None:
                writer.add_scalar("charts/episodic_return", episode["r"], global_step)
                writer.add_scalar("charts/episodic_length", episode["l"], global_step)
            if "completion" in info:
                writer.add_scalar("charts/episodic_completion", info["completion"], global_step)
            if "kill_ratio" in info:
                writer.add_scalar("charts/episodic_kill_ratio", info["kill_ratio"], global_step)
            if "coin_ratio" in info:
                writer.add_scalar("charts/episodic_coin_ratio", info["coin_ratio"], global_step)
            if "status" in info:
                win = 1.0 if str(info["status"]) == "WIN" else 0.0
                writer.add_scalar("charts/episodic_win", win, global_step)

    episode = infos.get("episode")
    if episode is not None:
        mask = np.asarray(
            infos.get("_episode", np.ones_like(episode.get("r", []), dtype=bool)),
            dtype=bool,
        )
        for value, done in zip(np.asarray(episode.get("r", [])), mask):
            if done:
                writer.add_scalar("charts/episodic_return", float(value), global_step)
        for value, done in zip(np.asarray(episode.get("l", [])), mask):
            if done:
                writer.add_scalar("charts/episodic_length", float(value), global_step)

    log_masked("episodic_completion", "completion")
    log_masked("episodic_kill_ratio", "kill_ratio")
    log_masked("episodic_coin_ratio", "coin_ratio")


def _count_finished_episodes(infos: dict[str, Any], global_step: int) -> int:
    if "final_info" in infos:
        finished = 0
        for info in infos["final_info"]:
            if info and "episode" in info:
                finished += 1
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
        return finished
    if "episode" in infos:
        mask = np.asarray(
            infos.get("_episode", np.ones_like(infos["episode"].get("r", []), dtype=bool)),
            dtype=bool,
        )
        return int(np.count_nonzero(mask))
    return 0


def collect_rollout(
    envs: gym.vector.VectorEnv,
    agent: actor_critic.ActorCritic,
    cfg: config.TrainConfig,
    device: torch.device,
    *,
    next_obs: ObsArrayDict,
    next_done: np.ndarray,
    global_step: int,
    writer: SummaryWriter | None = None,
    reward_fn: RewardFn | None = None,
) -> tuple[RolloutBatch, dict[str, Any]]:
    """Collect ``cfg.num_steps`` vector steps of on-policy experience, returning the
    filled :class:`RolloutBatch` and a stats dict with the carry state
    (``next_obs``, ``next_done``, ``global_step``, ``episodes_finished``)."""
    obs_space = envs.single_observation_space
    num_steps, num_envs = cfg.num_steps, cfg.num_envs
    obs_buf: ObsTensorDict = {
        "grid": torch.zeros(
            (num_steps, num_envs) + obs_space["grid"].shape,
            dtype=torch.int16,
            device=device,
        ),
        "state": torch.zeros(
            (num_steps, num_envs, obs_space["state"].shape[0]),
            dtype=torch.float32,
            device=device,
        ),
    }
    actions_buf = torch.zeros(
        (num_steps, num_envs) + envs.single_action_space.shape, device=device
    )
    logprobs_buf = torch.zeros((num_steps, num_envs), device=device)
    rewards_buf = torch.zeros((num_steps, num_envs), device=device)
    dones_buf = torch.zeros((num_steps, num_envs), device=device)
    values_buf = torch.zeros((num_steps, num_envs), device=device)

    episodes_finished = 0
    for step in range(num_steps):
        # Incremented before the env step: the first logged step is num_envs.
        global_step += num_envs
        obs_t = obs_to_torch(next_obs, device)
        obs_buf["grid"][step] = obs_t["grid"]
        obs_buf["state"][step] = obs_t["state"]
        dones_buf[step] = torch.as_tensor(next_done, device=device, dtype=torch.float32)

        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(obs_t)
            values_buf[step] = value.flatten()
        actions_buf[step] = action
        logprobs_buf[step] = logprob

        action_np = action.cpu().numpy()
        obs_before_step = next_obs
        next_obs, env_reward, terminations, truncations, infos = envs.step(action_np)
        # Truncation counts as termination — no bootstrapping through the
        # truncated state — and the autoreset terminal observation is ignored.
        next_done = np.logical_or(terminations, truncations).astype(np.float32)
        if reward_fn is None:
            reward = env_reward
        else:
            reward = reward_fn(obs_before_step, action_np, env_reward, next_done, infos)
        rewards_buf[step] = torch.as_tensor(reward, device=device).view(-1)

        episodes_finished += _count_finished_episodes(infos, global_step)
        if writer is not None:
            log_episode_infos(writer, infos, global_step, cfg.episode_log_interval)

    batch = RolloutBatch(
        obs=obs_buf,
        actions=actions_buf,
        logprobs=logprobs_buf,
        rewards=rewards_buf,
        dones=dones_buf,
        values=values_buf,
    )
    stats = {
        "next_obs": next_obs,
        "next_done": next_done,
        "global_step": global_step,
        "episodes_finished": episodes_finished,
    }
    return batch, stats


def compute_gae(
    batch: RolloutBatch,
    agent: actor_critic.ActorCritic,
    next_obs: ObsArrayDict,
    next_done: np.ndarray,
    cfg: config.TrainConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns for a rollout under ``no_grad``.
    Bootstraps from the carry observation's value; done flags mask both the bootstrap
    and the recursive accumulator, so truncated episodes count as terminated."""
    device = batch.rewards.device
    with torch.no_grad():
        next_value = agent.get_value(obs_to_torch(next_obs, device)).reshape(1, -1)
        next_done_t = torch.as_tensor(next_done, device=device, dtype=torch.float32)
        advantages = torch.zeros_like(batch.rewards)
        lastgaelam: torch.Tensor | float = 0.0
        for t in reversed(range(cfg.num_steps)):
            if t == cfg.num_steps - 1:
                nextnonterminal = 1.0 - next_done_t
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - batch.dones[t + 1]
                nextvalues = batch.values[t + 1]
            delta = (
                batch.rewards[t]
                + cfg.gamma * nextvalues * nextnonterminal
                - batch.values[t]
            )
            advantages[t] = lastgaelam = (
                delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + batch.values
    return advantages, returns
