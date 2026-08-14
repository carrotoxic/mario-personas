"""Frozen training configurations whose field names are the CLI/args-JSON contract.
Published ``mario_ppo_args.json`` files round-trip: matching keys load, unknown keys
are ignored, and missing keys fall back to the field defaults.

Paper presets live under ``main/configs/*.yaml`` and load via ``--config`` on the
training CLIs (CLI flags override YAML values)."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any, TypeVar

import tyro
import yaml

from src import paths

_LEVELS_DIR = paths.DATA_DIR / "human_experiment" / "levels"
_ACTION_STATE_DIR = paths.DATA_DIR / "human_experiment" / "action_state"
_DEFAULT_JAR_PATH = paths.SMB_DIR / "Mario-AI-Interface.jar"


@dataclasses.dataclass(frozen=True)
class PpoConfig:
    """Configuration for persona PPO training (tyro CLI and args-JSON field names)."""

    exp_name: str = "ppo_mario"  # Run-name component (``mario__<exp_name>__<seed>__<time>``).
    seed: int = 1
    torch_deterministic: bool = True  # Sets ``torch.backends.cudnn.deterministic`` only.
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str | None = None
    runs_dir: str = "runs"
    pretrained_path: str | None = None

    total_timesteps: int = 300_000_000
    learning_rate: float = 5e-4
    num_envs: int = 16  # Parallel AsyncVectorEnv workers (one JVM each).
    num_steps: int = 512
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 8
    norm_adv: bool = True
    clip_coef: float = 0.1  # PPO ratio clip range; also reused as the value clip range.
    clip_vloss: bool = True
    ent_coef: float = 0.0005
    vf_coef: float = 0.95
    max_grad_norm: float = 0.7
    # Per-epoch early-stop threshold on the last-minibatch approximate KL;
    # None disables early stopping.
    target_kl: float | None = 0.01

    # Save a state dict every N global steps (``<= 0`` disables checkpointing).
    checkpoint_interval: int = 1_000_000
    # Episodic charts are written only on vector steps where
    # ``global_step % episode_log_interval == 0``; since ``global_step`` is always a
    # multiple of ``num_envs``, the interval must be divisible by ``num_envs`` for the
    # charts to ever fire. ``<= 0`` disables the gate (log every step batch).
    episode_log_interval: int = 10_000
    # Holdout evaluation every N global steps (``<= 0`` disables it).
    eval_holdout_interval: int = 10_000_000
    eval_holdout_level_dir: str = str(_LEVELS_DIR / "playable_test")
    eval_num_levels: int = 10
    eval_num_iterations: int = 10

    features_dim: int = 1024
    embed_dim: int = 16
    head_hidden_dim: int = 1024

    level_dir: str = str(_LEVELS_DIR / "playable_train")
    jar_path: str = str(_DEFAULT_JAR_PATH)
    user_dir: str = str(paths.SMB_DIR)
    seconds: int = 50  # In-game per-segment time budget.
    mario_mode: int = 0  # Starting form: 0 = small, 1 = large, 2 = fire.
    max_steps: int = 3000  # Env truncation step cap; also caps eval episode loops.
    frame_skip: int = 1
    obs_shape: tuple[int, int] = (16, 16)
    lives: int = 0  # Starting lives; 0 means the first death loses.
    max_id: int = 100  # Max tile id; embedding table size is ``max_id + 1``.
    # Must stay True: the encoder only accepts Dict observation spaces.
    use_state_features: bool = True

    death_penalty: float = 0.0  # Reward added (already negative) on losing a life.
    completion_reward: float = 0.0  # Coefficient on the per-step completion delta.
    win_reward: float = 1.0
    kill_reward: float = 0.0
    coin_reward: float = 1.5
    mushroom_reward: float = 1.5
    fireflower_reward: float = 1.5
    brick_reward: float = 0.0
    segment_reward: float = 1.0

    # Derived by :func:`with_derived_sizes`, which overwrites any CLI input:
    # ``batch_size = num_envs * num_steps``, ``minibatch_size = batch_size //
    # num_minibatches``, ``num_iterations = total_timesteps // batch_size``.
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


@dataclasses.dataclass(frozen=True)
class DrailConfig:
    """Configuration for DRAIL imitation training (direct and post-training).
    Fields shared by name with :class:`PpoConfig` keep the same meaning; defaults follow
    direct DRAIL (post-training re-applies the pretrained bundle's hyperparameters)."""

    exp_name: str = "ppo_mario_drail"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    runs_dir: str = "runs"
    pretrained_path: str | None = None
    # Optional persona PPO checkpoint (file or run dir). When set, the trainer runs in
    # post-training mode: bundle hyperparameters are re-applied (``seconds``/``lives``
    # excepted) and the reward is ``env + drail_lambda * drail_term``. When None,
    # direct DRAIL uses the discriminator reward only.
    init_from: str | None = None

    total_timesteps: int = 10_000_000
    learning_rate: float = 1e-4
    num_envs: int = 8
    num_steps: int = 512
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 8
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.0005
    vf_coef: float = 0.95
    max_grad_norm: float = 0.7
    target_kl: float | None = 0.01

    checkpoint_interval: int = 1_000_000
    episode_log_interval: int = 100_000
    eval_every_steps: int = 1_000_000  # Greedy-eval interval in global steps (direct mode).
    n_eval_levels: int = 4  # Number of expert levels used by the greedy eval.

    features_dim: int = 1024
    embed_dim: int = 16
    head_hidden_dim: int = 1024

    level_dir: str = str(_LEVELS_DIR / "expert_levels_runner_stats")
    jar_path: str = str(_DEFAULT_JAR_PATH)
    user_dir: str = str(paths.SMB_DIR)
    seconds: int = 50
    mario_mode: int = 0
    max_steps: int = 3000
    frame_skip: int = 1
    obs_shape: tuple[int, int] = (16, 16)
    lives: int = 5
    max_id: int = 100
    use_state_features: bool = True

    death_penalty: float = 0.0
    completion_reward: float = 0.0
    win_reward: float = 5.0
    kill_reward: float = 0.0
    coin_reward: float = 0.0
    mushroom_reward: float = 0.0
    fireflower_reward: float = 0.0
    brick_reward: float = 0.0
    segment_reward: float = 1.0

    # Expert ``.npz`` transitions that train the discriminator
    # (filenames like ``<uuid>_lvl<id>.npz``).
    action_state_dir: str = str(_ACTION_STATE_DIR / "action_state_runner_stats")
    discriminator_lr: float = 1e-4
    traj_batch_size: int = 96
    traj_frac: float = 1.0  # Fraction of expert transitions kept (shuffled subset).
    n_drail_epochs: int = 1
    drail_state_norm: bool = True
    # Scale discriminator rewards by the running discounted-return standard deviation.
    drail_reward_norm: bool = True
    reward_scale: float = 1.0
    reward_clip: float = 10.0
    # Reward transform of the discriminator probability:
    # ``airl``, ``gail``, ``raw``, ``airl-positive``, or ``revise``.
    reward_type: str = "airl"
    # Width of the constant condition vector fed to the diffusion model
    # (the binary label is tiled into it).
    label_dim: int = 10
    discrim_depth: int = 4  # Diffusion MLP depth: ``max(depth - 1, 1)`` hidden blocks.
    discrim_hidden_dim: int = 128
    discrim_obs_features_dim: int = 64
    action_embed_dim: int = 16
    # Diffusion timestep sampling: ``random`` (antithetic) or ``constant``
    # (fixed at ``sample_strategy_value``).
    sample_strategy: str = "random"
    sample_strategy_value: int = 250

    # Coefficient of the KL penalty against the frozen pretrained policy
    # (post-training; 0 disables the term).
    ref_policy_kl_coef: float = 0.0
    drail_lambda: float = 3.0  # Mixing weight of the DRAIL term (post-training).
    # Discriminator update cadence after warmup (post-training; direct DRAIL
    # updates every iteration).
    drail_update_every_iters: int = 5
    # Discriminator sessions before the DRAIL reward turns on (post-training).
    drail_warmup_updates: int = 30
    # Discriminator-probability clip epsilon used in post-training
    # (direct DRAIL clips at 1e-8).
    drail_d_prob_clip_eps: float = 0.02

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


TrainConfig = PpoConfig | DrailConfig

_ConfigT = TypeVar("_ConfigT", PpoConfig, DrailConfig)


def with_derived_sizes(cfg: _ConfigT) -> _ConfigT:
    """Return a copy of ``cfg`` with the derived batch fields computed, overwriting any
    CLI-provided values. ``num_iterations`` uses floor division, so the trained step
    count is ``num_iterations * batch_size`` (at most ``total_timesteps``)."""
    batch_size = int(cfg.num_envs * cfg.num_steps)
    return dataclasses.replace(
        cfg,
        batch_size=batch_size,
        minibatch_size=int(batch_size // cfg.num_minibatches),
        num_iterations=cfg.total_timesteps // batch_size,
    )


CONFIGS_DIR = paths.PACKAGE_ROOT / "configs"
_REPO_PATH_FIELDS = frozenset({
    "level_dir",
    "eval_holdout_level_dir",
    "jar_path",
    "user_dir",
    "action_state_dir",
    "init_from",
    "pretrained_path",
})


def _resolve_repo_path(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((paths.REPO_ROOT / path).resolve())


def _coerce_config_value(field: dataclasses.Field[Any], value: Any) -> Any:
    if isinstance(field.default, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    merged: dict[str, Any] = {}
    include = raw.pop("_include", None)
    if include is not None:
        includes = [include] if isinstance(include, str) else list(include)
        for item in includes:
            merged.update(_load_yaml_mapping((path.parent / item).resolve()))
    merged.update(raw)
    return merged


def load_yaml_config(path: Path, config_cls: type[_ConfigT]) -> _ConfigT:
    """Load a YAML preset into ``config_cls``; unknown keys are ignored."""
    raw = _load_yaml_mapping(path.resolve())
    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(config_cls):
        if field.name not in raw:
            continue
        value = raw[field.name]
        if field.name in _REPO_PATH_FIELDS and isinstance(value, str):
            value = _resolve_repo_path(value)
        kwargs[field.name] = _coerce_config_value(field, value)
    return dataclasses.replace(config_cls(), **kwargs)


def _pop_config_arg(argv: list[str]) -> Path | None:
    if "--config" not in argv:
        return None
    index = argv.index("--config")
    if index + 1 >= len(argv):
        raise SystemExit("--config requires a path")
    config_path = Path(argv[index + 1]).expanduser()
    del argv[index : index + 2]
    if not config_path.is_file():
        resolved = CONFIGS_DIR / config_path
        if resolved.is_file():
            config_path = resolved
        else:
            raise FileNotFoundError(f"Config not found: {config_path}")
    return config_path


def cli_with_config(config_cls: type[_ConfigT]) -> _ConfigT:
    """Parse ``tyro.cli(config_cls)``, optionally seeded from ``--config <yaml>``."""
    argv = sys.argv[1:]
    config_path = _pop_config_arg(argv)
    default = load_yaml_config(config_path, config_cls) if config_path is not None else config_cls()
    if config_path is not None:
        print(f"[config] loaded preset: {config_path.resolve()}")
    return tyro.cli(config_cls, args=argv, default=default)
