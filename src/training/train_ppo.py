"""CLI for persona PPO training on the JVM Mario environment.
Flags are a tyro CLI over :class:`src.training.config.PpoConfig`. Each vector
worker is a spawned process hosting its own JVM."""

from __future__ import annotations

import dataclasses
import multiprocessing as mp
import random
import time
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src import env as src_env
from src.models import actor_critic
from src.training import checkpointing
from src.training import config as config_lib
from src.training import ppo
from src.training import rollouts


def make_env_config(cfg: config_lib.TrainConfig, level_dir: str) -> src_env.EnvConfig:
    """Build the :class:`src.env.EnvConfig` for a training config."""
    return src_env.EnvConfig(
        jar_path=cfg.jar_path,
        level_dir=level_dir,
        user_dir=cfg.user_dir,
        seconds=cfg.seconds,
        mario_mode=cfg.mario_mode,
        max_steps=cfg.max_steps,
        frame_skip=cfg.frame_skip,
        obs_shape=cfg.obs_shape,
        max_id=cfg.max_id,
        lives=cfg.lives,
        death_penalty=cfg.death_penalty,
        completion_reward=cfg.completion_reward,
        win_reward=cfg.win_reward,
        kill_reward=cfg.kill_reward,
        coin_reward=cfg.coin_reward,
        mushroom_reward=cfg.mushroom_reward,
        fireflower_reward=cfg.fireflower_reward,
        brick_reward=cfg.brick_reward,
        segment_reward=cfg.segment_reward,
        include_state_features=cfg.use_state_features,
    )


def make_env_thunk(
    cfg: config_lib.TrainConfig,
    level_dir: str,
    *,
    fixed_level_path: str | None = None,
) -> Callable[[], gym.Env]:
    """Return a vector-env worker thunk building one wrapped Mario env."""

    def thunk() -> gym.Env:
        env_cfg = make_env_config(cfg, level_dir)
        env = src_env.MarioEnv(env_cfg)
        if fixed_level_path is not None:
            env.set_level_files([fixed_level_path])
        return gym.wrappers.RecordEpisodeStatistics(env)

    return thunk


def _holdout_level_paths(level_dir: Path) -> list[Path]:
    if not level_dir.is_dir():
        return []
    return sorted([*level_dir.glob("*.lvl"), *level_dir.glob("*.txt")])


def _make_holdout_env(cfg: config_lib.PpoConfig, level_path: Path) -> src_env.MarioEnv:
    env = src_env.MarioEnv(make_env_config(cfg, str(level_path.parent.resolve())))
    env.set_level_files([str(level_path)])
    return env


def _single_obs_to_batch(
    obs: rollouts.ObsArrayDict, device: torch.device
) -> rollouts.ObsTensorDict:
    return {key: value.unsqueeze(0) for key, value in rollouts.obs_to_torch(obs, device).items()}


def run_holdout_episode(
    agent: actor_critic.ActorCritic,
    env: gym.Env,
    device: torch.device,
    max_steps: int,
    *,
    seed: int | None,
) -> tuple[float, float]:
    """Run one greedy (argmax) episode; returns ``(win as 0/1, final completion)``."""
    agent.eval()
    if seed is not None:
        obs, info = env.reset(seed=int(seed))
    else:
        obs, info = env.reset()
    terminated = False
    truncated = False
    steps = 0
    final_info = info
    while not (terminated or truncated) and steps < int(max_steps):
        obs_t = _single_obs_to_batch(obs, device)
        with torch.no_grad():
            action = agent.actor(agent.get_features(obs_t)).argmax(dim=-1)
        obs, _reward, terminated, truncated, final_info = env.step(int(action.item()))
        steps += 1
    win = 1.0 if str(final_info.get("status", "")) == "WIN" else 0.0
    completion = float(final_info.get("completion", 0.0))
    return win, completion


def evaluate_playable_test_holdout(
    agent: actor_critic.ActorCritic,
    cfg: config_lib.PpoConfig,
    device: torch.device,
    writer: SummaryWriter,
    global_step: int,
) -> None:
    """Evaluate on sampled holdout levels; blocks training in this process.
    Level choice and episode seeds derive from ``cfg.seed`` and ``global_step``, so
    results are deterministic given the pool contents."""
    level_dir = Path(cfg.eval_holdout_level_dir).expanduser()
    pool = _holdout_level_paths(level_dir)
    if not pool:
        print(f"[eval holdout] skip: no .lvl/.txt in {level_dir}")
        return

    # The eval env is built in the main process, which therefore hosts its own
    # JVM from the first eval onward (one JVM per process, never shut down).
    n_levels = max(1, int(cfg.eval_num_levels))
    n_iters = max(1, int(cfg.eval_num_iterations))
    k = min(n_levels, len(pool))
    rng = random.Random(int(cfg.seed) + int(global_step))
    paths = rng.sample(pool, k=k)

    base_env = _make_holdout_env(cfg, paths[0])
    level_win_indicator: list[float] = []
    level_best_completion: list[float] = []
    try:
        for level_idx, level_path in enumerate(paths):
            base_env.set_level_files([str(level_path)])
            seed_base = int(cfg.seed) + level_idx * n_iters + int(global_step)
            any_win = False
            best_completion = 0.0
            for repeat in range(n_iters):
                win, completion = run_holdout_episode(
                    agent, base_env, device, int(cfg.max_steps), seed=seed_base + repeat
                )
                any_win = any_win or (win >= 1.0)
                best_completion = max(best_completion, completion)
            level_win_indicator.append(1.0 if any_win else 0.0)
            level_best_completion.append(best_completion)
    finally:
        base_env.close()

    mean_win = float(np.mean(level_win_indicator))
    mean_completion = float(np.mean(level_best_completion))
    # Restored to train mode only on success; an exception mid-eval leaves eval mode.
    agent.train()

    writer.add_scalar("eval/playable_test_mean_win_rate", mean_win, global_step)
    writer.add_scalar("eval/playable_test_mean_completion", mean_completion, global_step)
    writer.add_scalar("eval/playable_test_num_levels", float(len(paths)), global_step)
    writer.add_scalar("eval/playable_test_eval_num_iterations", float(n_iters), global_step)
    print(
        f"[eval holdout @ step {global_step}] playable_test: "
        f"mean_win_rate={mean_win:.4f} mean_completion={mean_completion:.4f} "
        f"({len(paths)} sampled levels, {n_iters} iters/level)"
    )


def train(cfg: config_lib.PpoConfig) -> None:
    """Train a persona PPO agent and write the run's args-JSON and checkpoint artifacts."""
    run_name = f"mario__{cfg.exp_name}__{cfg.seed}__{int(time.time())}"
    if cfg.track:
        import wandb

        wandb.init(
            project=cfg.wandb_project_name,
            entity=cfg.wandb_entity,
            sync_tensorboard=True,
            config=dataclasses.asdict(cfg),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    run_dir = Path(cfg.runs_dir) / run_name
    writer = SummaryWriter(str(run_dir))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % "\n".join(f"|{key}|{value}|" for key, value in dataclasses.asdict(cfg).items()),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpointing.save_config_json(run_dir, cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    envs = gym.vector.AsyncVectorEnv(
        [make_env_thunk(cfg, cfg.level_dir) for _ in range(cfg.num_envs)],
        context="spawn",
        daemon=False,
    )
    if not isinstance(envs.single_action_space, gym.spaces.Discrete):
        raise TypeError("only discrete action space is supported")

    agent = actor_critic.ActorCritic(
        envs.single_observation_space,
        int(envs.single_action_space.n),
        max_id=cfg.max_id,
        embed_dim=cfg.embed_dim,
        features_dim=cfg.features_dim,
        head_hidden_dim=cfg.head_hidden_dim,
    ).to(device)
    if cfg.pretrained_path:
        pretrained_path = Path(cfg.pretrained_path)
        agent.load_state_dict(checkpointing.load_checkpoint_state_dict(pretrained_path, device))
        print(f"Loaded pretrained weights from: {pretrained_path}")
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    checkpoint_schedule = checkpointing.StepScheduler(cfg.checkpoint_interval)
    holdout_schedule = checkpointing.StepScheduler(cfg.eval_holdout_interval)

    global_step = 0
    episodes_finished = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=cfg.seed)
    next_done = np.zeros(cfg.num_envs, dtype=np.float32)

    pbar = tqdm(total=cfg.total_timesteps, desc="Training", unit="steps")
    for iteration in range(1, cfg.num_iterations + 1):
        if cfg.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        batch, stats = rollouts.collect_rollout(
            envs,
            agent,
            cfg,
            device,
            next_obs=next_obs,
            next_done=next_done,
            global_step=global_step,
            writer=writer,
        )
        next_obs = stats["next_obs"]
        next_done = stats["next_done"]
        global_step = stats["global_step"]
        episodes_finished += stats["episodes_finished"]

        advantages, returns = rollouts.compute_gae(batch, agent, next_obs, next_done, cfg)
        update_stats = ppo.ppo_update(agent, optimizer, batch, advantages, returns, cfg)

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        for key in (
            "value_loss",
            "policy_loss",
            "entropy",
            "old_approx_kl",
            "approx_kl",
            "clipfrac",
            "explained_variance",
        ):
            writer.add_scalar(f"losses/{key}", update_stats[key], global_step)
        sps = int(global_step / max(time.time() - start_time, 1e-8))
        print("SPS:", sps)
        writer.add_scalar("charts/SPS", sps, global_step)
        pbar.update(cfg.batch_size)
        pbar.set_postfix(episodes=episodes_finished, sps=sps)

        for scheduled_step in checkpoint_schedule.due_steps(global_step):
            checkpointing.save_state_dict(
                agent, checkpointing.step_checkpoint_path(run_dir, scheduled_step)
            )
        for _ in holdout_schedule.due_steps(global_step):
            evaluate_playable_test_holdout(agent, cfg, device, writer, global_step)

    envs.close()
    pbar.close()
    checkpointing.save_state_dict(agent, checkpointing.final_checkpoint_path(run_dir))
    checkpointing.save_config_json(run_dir, cfg)
    writer.close()


def main() -> None:
    # A JPype JVM cannot survive fork(); every vector worker must be spawned
    # so it can boot its own JVM.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    cfg = config_lib.with_derived_sizes(config_lib.cli_with_config(config_lib.PpoConfig))
    train(cfg)


if __name__ == "__main__":
    main()
