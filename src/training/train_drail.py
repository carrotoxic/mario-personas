"""CLI for DRAIL imitation training over :class:`src.training.config.DrailConfig`.
Direct DRAIL trains on the (normalized) discriminator reward alone; post-training
(``--init-from``) re-applies the bundle's hyperparameters and mixes in the env reward."""

from __future__ import annotations

import dataclasses
import json
import multiprocessing as mp
import random
import shutil
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import optim
from torch.distributions import Categorical, kl_divergence
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src import paths
from src.data import demonstrations
from src.models import ActorCritic
from src.training import checkpointing
from src.training import config as config_lib
from src.training import drail
from src.training import ppo
from src.training import rollouts
from src.training import train_ppo


def _resolve_repo_path(path_like: str) -> Path:
    path = Path(path_like)
    return path.resolve() if path.is_absolute() else (paths.REPO_ROOT / path).resolve()


def find_pretrained_weights(bundle_dir: Path) -> Path | None:
    """Pick the weights file in a bundle: ``mario_ppo.pt``, persona stems, then ``*.pt``."""
    for name in (
        checkpointing.FINAL_CHECKPOINT_FILENAME,
        # Extensionless ``torch.save`` files shipped in some published bundles.
        "mario_ppo_runner",
        "mario_ppo_collector",
        "mario_ppo_killer",
    ):
        candidate = bundle_dir / name
        if candidate.is_file():
            return candidate
    pt_files = sorted(bundle_dir.glob("*.pt"))
    if len(pt_files) == 1:
        return pt_files[0]
    for candidate in pt_files:
        if "mario" in candidate.name.lower():
            return candidate
    return pt_files[0] if pt_files else None


def resolve_pretrained_bundle(path_like: str) -> tuple[Path, Path]:
    """Resolve ``--init-from`` (checkpoint file or run dir) into ``(bundle_dir, weights)``."""
    path = _resolve_repo_path(path_like)
    if path.is_file():
        return path.parent, path
    if path.is_dir():
        weights = find_pretrained_weights(path)
        if weights is None:
            raise FileNotFoundError(
                f"No pretrained weights found in {path}. "
                f"Expected one of [{checkpointing.FINAL_CHECKPOINT_FILENAME}, "
                "mario_ppo_runner, mario_ppo_collector, mario_ppo_killer] "
                "or a single *.pt file."
            )
        return path, weights
    raise FileNotFoundError(f"--init-from path does not exist: {path}")


def apply_pretrained_run_config(
    cfg: config_lib.DrailConfig, bundle_dir: Path
) -> config_lib.DrailConfig:
    """Re-apply the pretrained run's hyperparameters onto ``cfg``: every shared
    PpoConfig/DrailConfig field (minus the CLI-kept set) comes from the bundle's args
    JSON over the CLI; missing keys fall back, except ``seconds``/``lives`` (keep CLI)."""
    # Run setup, step budget, machine paths, and derived sizes stay CLI-owned.
    keep_cli = frozenset({
        "exp_name", "seed", "torch_deterministic", "cuda", "runs_dir", "pretrained_path",
        "total_timesteps", "num_envs", "checkpoint_interval", "episode_log_interval",
        "level_dir", "jar_path", "user_dir", "batch_size", "minibatch_size", "num_iterations",
    })
    args_json_path = bundle_dir / checkpointing.ARGS_FILENAME
    if args_json_path.is_file():
        raw = json.loads(args_json_path.read_text(encoding="utf-8"))
        bundle = checkpointing.load_config_json(args_json_path, config_lib.DrailConfig)
        print(f"[pretrained] loaded merged-key defaults from: {args_json_path}")
    else:
        raw, bundle = {}, config_lib.DrailConfig()
        print(f"[pretrained] {checkpointing.ARGS_FILENAME} not found at {args_json_path}")
    # Fallbacks for keys missing from the args JSON equal the DrailConfig
    # defaults, except these two.
    if "learning_rate" not in raw:
        bundle = dataclasses.replace(bundle, learning_rate=5e-5)
    if "update_epochs" not in raw:
        bundle = dataclasses.replace(bundle, update_epochs=3)

    print("[pretrained] merged-key resolution:")
    ppo_field_names = {field.name for field in dataclasses.fields(config_lib.PpoConfig)}
    overrides: dict[str, object] = {}
    for field in dataclasses.fields(config_lib.DrailConfig):
        key = field.name
        if key not in ppo_field_names or key in keep_cli:
            continue
        if key in raw:
            overrides[key] = getattr(bundle, key)
            print(f"  {key}: loaded properly")
        elif key in ("seconds", "lives"):
            print(f"  {key}: ! FALLBACK (using CLI/default value)")
        else:
            overrides[key] = getattr(bundle, key)
            print(f"  {key}: ! FALLBACK")
    return dataclasses.replace(cfg, **overrides)


def stage_expert_levels(
    demos: demonstrations.ExpertDemonstrations, level_dir: Path, run_dir: Path
) -> Path:
    """Copy the expert's level files into ``<run_dir>/expert_levels`` (the envs' whole pool)."""
    level_files = demonstrations.get_level_files_for_ids(level_dir, demos.level_ids)
    if not level_files:
        raise ValueError("No expert level files found.")
    expert_level_dir = run_dir / "expert_levels"
    expert_level_dir.mkdir(parents=True, exist_ok=True)
    for level_file in level_files:
        shutil.copy2(level_file, expert_level_dir / level_file.name)
    return expert_level_dir


class ReferenceKlPenalty:
    """PPO extra-loss term ``coef * KL(pi || pi_ref)`` per minibatch.
    The frozen reference is evaluated under ``no_grad``; the last minibatch's
    unscaled KL is kept in ``last_kl`` for logging."""

    def __init__(self, agent: ActorCritic, ref_agent: ActorCritic, coef: float):
        self._agent = agent
        self._ref_agent = ref_agent
        self._coef = float(coef)
        self.last_kl = 0.0

    def __call__(
        self, mb_obs: rollouts.ObsTensorDict, mb_actions: torch.Tensor
    ) -> torch.Tensor:
        logits = self._agent.actor(self._agent.get_features(mb_obs))
        with torch.no_grad():
            ref_logits = self._ref_agent.actor(self._ref_agent.get_features(mb_obs))
        kl = kl_divergence(Categorical(logits=logits), Categorical(logits=ref_logits)).mean()
        self.last_kl = float(kl.detach().item())
        return self._coef * kl


def _batch_of_one(obs: rollouts.ObsArrayDict, device: torch.device) -> rollouts.ObsTensorDict:
    return {key: value.unsqueeze(0) for key, value in rollouts.obs_to_torch(obs, device).items()}


def evaluate_greedy(
    agent: ActorCritic,
    cfg: config_lib.DrailConfig,
    level_paths: list[Path],
    device: torch.device,
) -> tuple[float, float]:
    """Greedy per-level eval returning ``(mean_env_reward, mean_length)`` (direct mode).
    Builds a fresh env — a fresh JVM — per level and accumulates the environment
    reward that direct-DRAIL training discards."""
    rewards: list[float] = []
    lengths: list[int] = []
    for index, level_path in enumerate(level_paths):
        env = train_ppo.make_env_thunk(
            cfg, str(level_path.parent), fixed_level_path=str(level_path)
        )()
        obs, _ = env.reset(seed=cfg.seed + index)
        total_reward = 0.0
        steps = 0
        done = False
        while not done and steps < int(cfg.max_steps):
            with torch.no_grad():
                action = agent.actor(agent.get_features(_batch_of_one(obs, device))).argmax(dim=-1)
            obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            total_reward += float(reward)
            steps += 1
            done = bool(terminated or truncated)
        env.close()
        rewards.append(total_reward)
        lengths.append(steps)
    if not rewards:
        return 0.0, 0.0
    return float(np.mean(rewards)), float(np.mean(lengths))


def train(cfg: config_lib.DrailConfig, init_weights: Path | None) -> None:
    """Run DRAIL training on the shared PPO core; ``init_weights`` selects post-training."""
    posttrain = init_weights is not None
    run_name = f"mario__{cfg.exp_name}__{cfg.seed}__{int(time.time())}"
    run_dir = Path(cfg.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))
    checkpointing.save_config_json(run_dir, cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    demos = demonstrations.load_demonstrations(
        Path(cfg.action_state_dir).resolve(), traj_frac=float(cfg.traj_frac), seed=int(cfg.seed)
    )
    expert_level_dir = stage_expert_levels(demos, Path(cfg.level_dir).resolve(), run_dir)

    envs = gym.vector.AsyncVectorEnv(
        [train_ppo.make_env_thunk(cfg, str(expert_level_dir)) for _ in range(cfg.num_envs)],
        context="spawn",
        daemon=False,
    )
    if not isinstance(envs.single_action_space, gym.spaces.Discrete):
        raise TypeError("only discrete action space is supported")
    num_actions = int(envs.single_action_space.n)

    def make_agent() -> ActorCritic:
        return ActorCritic(
            envs.single_observation_space,
            num_actions,
            max_id=cfg.max_id,
            embed_dim=cfg.embed_dim,
            features_dim=cfg.features_dim,
            head_hidden_dim=cfg.head_hidden_dim,
        ).to(device)

    agent = make_agent()
    ref_agent: ActorCritic | None = None
    if posttrain:
        agent.load_state_dict(checkpointing.load_checkpoint_state_dict(init_weights, device))
        ref_agent = make_agent()
        ref_agent.load_state_dict(agent.state_dict())
        ref_agent.eval()
        ref_agent.requires_grad_(False)
        print("[pretrained] frozen reference agent for KL(pi || pi_ref) regularization")
    elif cfg.pretrained_path:
        agent.load_state_dict(
            checkpointing.load_checkpoint_state_dict(Path(cfg.pretrained_path), device)
        )
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # The discriminator always starts from scratch, even in post-training.
    disc = drail.make_discriminator(envs.single_observation_space, num_actions, cfg, device)
    disc_opt = optim.Adam(disc.parameters(), lr=cfg.discriminator_lr)
    if posttrain:
        print(f"[DRAIL-POSTTRAIN] Using mixed reward: env + {cfg.drail_lambda} * drail")
    else:
        print("[DRAIL] Using discriminator rewards only")

    composer = drail.DrailRewardComposer(
        disc,
        num_envs=cfg.num_envs,
        state_dim=int(envs.single_observation_space["state"].shape[0]),
        gamma=cfg.gamma,
        reward_type=cfg.reward_type,
        reward_scale=cfg.reward_scale,
        reward_clip=cfg.reward_clip,
        d_prob_clip_eps=(
            drail.clamp_probability_epsilon(cfg.drail_d_prob_clip_eps)
            if posttrain
            else drail.DIRECT_D_PROB_CLIP_EPS
        ),
        state_norm=cfg.drail_state_norm,
        reward_norm=cfg.drail_reward_norm,
        drail_lambda=cfg.drail_lambda if posttrain else None,
        device=device,
    )
    kl_penalty: ReferenceKlPenalty | None = None
    if ref_agent is not None and float(cfg.ref_policy_kl_coef) != 0.0:
        kl_penalty = ReferenceKlPenalty(agent, ref_agent, cfg.ref_policy_kl_coef)

    checkpoint_schedule = checkpointing.StepScheduler(cfg.checkpoint_interval)
    eval_schedule = checkpointing.StepScheduler(0 if posttrain else cfg.eval_every_steps)
    eval_levels = sorted([*expert_level_dir.glob("*.lvl"), *expert_level_dir.glob("*.txt")])[
        : max(1, int(cfg.n_eval_levels))
    ]

    warmup_sessions = max(0, int(cfg.drail_warmup_updates))
    disc_sessions = 0
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=cfg.seed)
    next_done = np.zeros(cfg.num_envs, dtype=np.float32)
    pbar = tqdm(total=cfg.total_timesteps, desc="Training", unit="steps")

    for iteration in range(1, cfg.num_iterations + 1):
        if cfg.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        # The warmup gate is evaluated before this iteration's discriminator
        # update: with warmup 30, the reward first applies in iteration 31.
        reward_enabled = True if not posttrain else disc_sessions >= warmup_sessions
        composer.begin_iteration(reward_enabled)
        batch, stats = rollouts.collect_rollout(
            envs,
            agent,
            cfg,
            device,
            next_obs=next_obs,
            next_done=next_done,
            global_step=global_step,
            writer=writer,
            reward_fn=composer,
        )
        next_obs = stats["next_obs"]
        next_done = stats["next_done"]
        global_step = stats["global_step"]

        if posttrain:
            d_prob_mean = composer.d_prob_mean
            if reward_enabled and d_prob_mean is not None:
                writer.add_scalar("drail/d_prob_mean_on_policy", d_prob_mean, global_step)
            writer.add_scalar(
                "drail/shaped_reward_enabled", 1.0 if reward_enabled else 0.0, global_step
            )

        if not posttrain or not reward_enabled:
            should_update_disc = True
        else:
            should_update_disc = (
                int(cfg.drail_update_every_iters) <= 1
                or iteration % int(cfg.drail_update_every_iters) == 0
            )
        disc_result: drail.DiscriminatorUpdateResult | None = None
        policy_obs: rollouts.ObsArrayDict = {}
        policy_actions = policy_levels = np.zeros((0,), dtype=np.int64)
        if should_update_disc:
            policy_obs, policy_actions, policy_levels = composer.policy_batch()
            disc_result = drail.update_discriminator(
                disc,
                disc_opt,
                demos,
                policy_obs,
                policy_actions,
                policy_levels,
                state_rms=composer.state_rms,
                state_norm=cfg.drail_state_norm,
                batch_size=cfg.traj_batch_size,
                num_epochs=cfg.n_drail_epochs,
                device=device,
            )
            disc_sessions += 1

        # GAE consumes the stored rewards, which were computed with the
        # previous iteration's discriminator (pre-update).
        advantages, returns = rollouts.compute_gae(batch, agent, next_obs, next_done, cfg)
        update_stats = ppo.ppo_update(
            agent, optimizer, batch, advantages, returns, cfg, extra_loss_fn=kl_penalty
        )

        after_rl_gap = 0.0
        if posttrain and disc_result is not None:
            after_rl_gap = drail.post_update_policy_gap(
                disc,
                agent,
                demos,
                policy_obs,
                policy_actions,
                policy_levels,
                state_rms=composer.state_rms,
                state_norm=cfg.drail_state_norm,
                device=device,
            )

        sps = int(global_step / max(time.time() - start_time, 1e-8))
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
        if posttrain:
            writer.add_scalar(
                "losses/ref_policy_kl",
                kl_penalty.last_kl if kl_penalty is not None else 0.0,
                global_step,
            )
            writer.add_scalar("drail/lambda", float(cfg.drail_lambda), global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        if disc_result is not None:
            writer.add_scalar("drail/discriminator_loss", disc_result.loss, global_step)
            writer.add_scalar("drail/gap_before", disc_result.gap_before, global_step)
            writer.add_scalar("drail/gap_after_disc", disc_result.gap_after, global_step)
            if posttrain:
                writer.add_scalar("drail/gap_after_rl", after_rl_gap, global_step)
        writer.add_scalar("drail/raw_reward", composer.mean_raw_reward, global_step)
        writer.add_scalar("drail/norm_reward", composer.mean_normalized_reward, global_step)

        for _ in eval_schedule.due_steps(global_step):
            mean_reward, mean_length = evaluate_greedy(agent, cfg, eval_levels, device)
            writer.add_scalar("eval/mean_reward", mean_reward, global_step)
            writer.add_scalar("eval/mean_length", mean_length, global_step)

        pbar.update(cfg.batch_size)
        pbar.set_postfix(sps=sps)
        for scheduled_step in checkpoint_schedule.due_steps(global_step):
            step_path = checkpointing.step_checkpoint_path(run_dir, scheduled_step)
            if posttrain:
                checkpointing.save_state_dict(agent, step_path)
            else:
                torch.save(
                    {
                        "state_dict": agent.state_dict(),
                        "discriminator": {
                            "state_dict": disc.state_dict(),
                            "optimizer_state_dict": disc_opt.state_dict(),
                        },
                    },
                    step_path,
                )

    envs.close()
    pbar.close()
    checkpointing.save_state_dict(agent, checkpointing.final_checkpoint_path(run_dir))
    torch.save(
        {"state_dict": disc.state_dict(), "optimizer_state_dict": disc_opt.state_dict()},
        run_dir / checkpointing.DISCRIMINATOR_CHECKPOINT_FILENAME,
    )
    writer.close()


def main() -> None:
    # A JPype JVM cannot survive fork(); every vector worker must be spawned
    # so it can boot its own JVM.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    cfg = config_lib.cli_with_config(config_lib.DrailConfig)
    init_weights: Path | None = None
    if cfg.init_from:
        bundle_dir, init_weights = resolve_pretrained_bundle(cfg.init_from)
        print(f"[pretrained] bundle directory: {bundle_dir}")
        print(f"[pretrained] weights file: {init_weights}")
        cfg = apply_pretrained_run_config(cfg, bundle_dir)
    cfg = config_lib.with_derived_sizes(cfg)
    train(cfg, init_weights)


if __name__ == "__main__":
    main()
