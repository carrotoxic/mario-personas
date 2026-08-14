"""Vectorized rollout machinery shared by the evaluation CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

import gymnasium as gym
from gymnasium.vector import AutoresetMode
import numpy as np
import torch
from tqdm import tqdm

from src.evaluation import loading
from src.models import ActorCritic

# Maps a batched observation dict and the active env indices to int64 actions
# for those indices; the third argument is the discrete action-space size.
ActionSource = Callable[[dict, np.ndarray, int], np.ndarray]


class EpisodeRecord(NamedTuple):
    """One finished (or timed-out) evaluation episode."""

    status: str
    reward: float
    length: int
    completion: float
    kill_ratio: float
    coin_ratio: float
    kills: int
    coins: int
    playable: bool


class LevelSummary(NamedTuple):
    """Aggregate of all episodes played on one level."""

    level: str
    playable: bool
    win_count: int
    run_count: int
    win_rate: float
    best_completion: float
    best_kill_ratio: float
    best_coin_ratio: float
    avg_reward: float
    avg_length: float


def make_async_envs(
    thunks: Sequence[Callable[[], gym.Env]], *, disable_autoreset: bool = False
) -> gym.vector.AsyncVectorEnv:
    """Creates an ``AsyncVectorEnv`` with JVM-safe workers: ``context="spawn"`` and
    ``daemon=False`` are hard requirements (a JPype JVM cannot survive fork, and workers
    own JVM threads); ``disable_autoreset`` gives exactly one episode per worker."""
    kwargs: dict = {"context": "spawn", "daemon": False}
    if disable_autoreset:
        kwargs["autoreset_mode"] = AutoresetMode.DISABLED
    return gym.vector.AsyncVectorEnv(list(thunks), **kwargs)


def batch_obs_to_torch(
    obs: dict | np.ndarray, device: torch.device
) -> dict[str, torch.Tensor] | torch.Tensor:
    """Converts an already-batched vector-env observation to torch tensors."""
    if isinstance(obs, dict):
        return {
            "grid": torch.as_tensor(obs["grid"], device=device),
            "state": torch.as_tensor(obs["state"], device=device, dtype=torch.float32),
        }
    return torch.as_tensor(obs, device=device)


def select_actions(
    agent: ActorCritic, obs_t: dict[str, torch.Tensor], *, deterministic: bool
) -> torch.Tensor:
    """Greedy argmax actions if ``deterministic``, else samples from the policy."""
    with torch.no_grad():
        if deterministic:
            return agent.actor(agent.get_features(obs_t)).argmax(dim=-1)
        action, _, _, _ = agent.get_action_and_value(obs_t)
        return action


def final_info_value(infos: dict, key: str, index: int, default: Any) -> Any:
    """Reads a final-info value from gymnasium's per-key array convention
    (``infos[key]`` gated by the boolean mask ``infos["_" + key]``)."""
    vals = infos.get(key)
    if vals is None:
        return default
    mask = infos.get(f"_{key}")
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if index >= len(mask_arr) or not mask_arr[index]:
            return default
    try:
        return vals[index]
    except (IndexError, KeyError, TypeError):
        return default


def final_info_dict(infos: dict, index: int, num_envs: int) -> dict:
    """Returns the ``final_info`` dict for one env, or ``{}`` (list convention)."""
    final_infos = infos.get("final_info")
    mask = np.asarray(infos.get("_final_info", np.zeros((num_envs,), dtype=bool)), dtype=bool)
    if final_infos is not None and index < len(final_infos) and mask[index]:
        fi = final_infos[index]
        if isinstance(fi, dict):
            return fi
    return {}


def episode_record(
    infos: dict, index: int, num_envs: int, *, reward: float, length: int
) -> EpisodeRecord:
    """Builds an ``EpisodeRecord`` from step infos, covering both info conventions."""
    fi = final_info_dict(infos, index, num_envs)
    status = str(fi.get("status", final_info_value(infos, "status", index, "UNKNOWN")))
    return EpisodeRecord(
        status=status,
        reward=float(reward),
        length=int(length),
        completion=float(fi.get("completion", final_info_value(infos, "completion", index, 0.0))),
        kill_ratio=float(fi.get("kill_ratio", final_info_value(infos, "kill_ratio", index, 0.0))),
        coin_ratio=float(fi.get("coin_ratio", final_info_value(infos, "coin_ratio", index, 0.0))),
        kills=int(fi.get("kills", final_info_value(infos, "kills", index, 0))),
        coins=int(fi.get("coins", final_info_value(infos, "coins", index, 0))),
        playable=status == "WIN",
    )


def timeout_record(*, reward: float, length: int) -> EpisodeRecord:
    """Record for a worker that never finished within the step budget; partial progress
    is deliberately zeroed, and downstream ``best_*`` aggregates depend on that."""
    return EpisodeRecord(
        status="TIMEOUT",
        reward=float(reward),
        length=int(length),
        completion=0.0,
        kill_ratio=0.0,
        coin_ratio=0.0,
        kills=0,
        coins=0,
        playable=False,
    )


def summarize_runs(level_path: Path, runs: Sequence[EpisodeRecord]) -> LevelSummary:
    """Aggregates all episodes on one level into a ``LevelSummary``."""
    win_count = sum(1 for run in runs if run.playable)
    run_count = len(runs)
    return LevelSummary(
        level=str(level_path),
        playable=any(run.playable for run in runs),
        win_count=win_count,
        run_count=run_count,
        win_rate=win_count / run_count,
        best_completion=max(run.completion for run in runs),
        best_kill_ratio=max(run.kill_ratio for run in runs),
        best_coin_ratio=max(run.coin_ratio for run in runs),
        avg_reward=sum(run.reward for run in runs) / run_count,
        avg_length=sum(run.length for run in runs) / run_count,
    )


def evaluate_level(
    level_path: Path,
    envs: gym.vector.AsyncVectorEnv,
    agent: ActorCritic,
    *,
    repeats: int,
    deterministic: bool,
    device: torch.device,
    seed_base: int,
    max_steps: int,
) -> LevelSummary:
    """Plays ``repeats`` parallel episodes of one level and summarizes them.  Worker
    ``j`` is reset with seed ``seed_base + j``; autoreset stays enabled, and a
    ``finished`` mask freezes each worker's accounting after its first episode."""
    n_envs = max(1, int(repeats))
    envs.call("set_level_files", [str(level_path)])

    runs: list[EpisodeRecord] = []
    rewards_acc = np.zeros((n_envs,), dtype=np.float32)
    lengths_acc = np.zeros((n_envs,), dtype=np.int32)
    finished = np.zeros((n_envs,), dtype=bool)
    pbar = tqdm(total=n_envs, desc=f"{level_path.stem}", unit="run")
    try:
        seeds = (np.arange(n_envs, dtype=np.int64) + int(seed_base)).tolist()
        obs, _ = envs.reset(seed=seeds)
        step_idx = 0
        while len(runs) < n_envs and step_idx < int(max_steps):
            obs_t = batch_obs_to_torch(obs, device)
            action = select_actions(agent, obs_t, deterministic=deterministic)
            obs, reward, terminations, truncations, infos = envs.step(
                action.detach().cpu().numpy()
            )
            done = np.logical_or(terminations, truncations)
            active = ~finished
            rewards_acc[active] += np.asarray(reward, dtype=np.float32)[active]
            lengths_acc[active] += 1
            for idx in np.where(done & active)[0].tolist():
                runs.append(
                    episode_record(
                        infos, idx, n_envs, reward=rewards_acc[idx], length=lengths_acc[idx]
                    )
                )
                finished[idx] = True
                pbar.update(1)
            step_idx += 1

        for idx in np.where(~finished)[0].tolist():
            runs.append(timeout_record(reward=rewards_acc[idx], length=lengths_acc[idx]))
            pbar.update(1)
    finally:
        pbar.close()
    return summarize_runs(level_path, runs)


def rollout_actions(
    level_paths: Sequence[Path],
    seeds: Sequence[int],
    train_args: loading.TrainConfig,
    action_source: ActionSource,
) -> list[np.ndarray]:
    """Runs one episode per (level, seed) pair, returning each episode's ``(T,)`` int64
    action sequence.  Uses a fresh vector env with autoreset disabled (exactly one
    episode per worker); the action source is called exactly once per step."""
    n = len(level_paths)
    if n == 0:
        return []
    max_steps = int(train_args.max_steps)
    thunks = [loading.make_env_thunk(lp, train_args) for lp in level_paths]
    envs = make_async_envs(thunks, disable_autoreset=True)
    act_lists: list[list[int]] = [[] for _ in range(n)]
    try:
        obs, _ = envs.reset(seed=[int(s) for s in seeds])
        active = np.ones(n, dtype=bool)
        steps = 0
        n_actions = int(envs.single_action_space.n)
        while bool(active.any()) and steps < max_steps:
            idx = np.flatnonzero(active)
            actions_np = np.zeros(n, dtype=np.int64)  # finished workers are fed action 0
            actions_np[idx] = np.asarray(
                action_source(obs, idx, n_actions), dtype=np.int64
            ).reshape(-1)
            obs, _, term, trunc, _ = envs.step(actions_np)
            for i in idx:
                act_lists[i].append(int(actions_np[i]))
            done = np.logical_or(term, trunc)
            active &= ~np.asarray(done, dtype=bool)
            steps += 1
        return [np.asarray(acts, dtype=np.int64) for acts in act_lists]
    finally:
        envs.close()
