"""Checkpoint, args, agent, and environment loading shared by all evaluation CLIs."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Callable, TypeVar

import torch

from src import paths
from src.env import EnvConfig, MarioEnv
from src.evaluation import CHECKPOINT_STEP_RE, resolve_repo_path
from src.models import ActorCritic
from src.training.config import DrailConfig, PpoConfig

TrainConfig = PpoConfig | DrailConfig
ConfigT = TypeVar("ConfigT", PpoConfig, DrailConfig)


def resolve_model_path(model_path: str | None) -> Path:
    """Resolves a checkpoint argument (file, run dir, or None = newest run) to a ``.pt`` file."""
    if model_path:
        path = Path(model_path)
        if not path.is_absolute():
            path = (paths.REPO_ROOT / path).resolve()
        if path.is_dir():
            candidate = path / "mario_ppo.pt"
            if candidate.exists():
                return candidate
        if path.exists():
            return path
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    candidates = sorted(
        (paths.REPO_ROOT / "runs").glob("mario__ppo_mario__*/mario_ppo.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No mario_ppo.pt checkpoint found under runs/. "
            "Train ppo_mario first or pass --model-path."
        )
    return candidates[0]


def pick_role_checkpoint(role_dir: Path) -> Path:
    """Picks a role's eval checkpoint: highest ``mario_ppo_step_*.pt``, then
    ``mario_ppo.pt``, then a single arbitrary ``*.pt``."""
    step_files = list(role_dir.glob("mario_ppo_step_*.pt"))

    def step_num(p: Path) -> int:
        m = CHECKPOINT_STEP_RE.match(p.name)
        return int(m.group(1)) if m else -1

    if step_files:
        return max(step_files, key=step_num)
    final = role_dir / "mario_ppo.pt"
    if final.is_file():
        return final
    any_pt = sorted(role_dir.glob("*.pt"))
    if len(any_pt) == 1:
        return any_pt[0]
    raise FileNotFoundError(
        f"No usable checkpoint in {role_dir}: "
        "add mario_ppo_step_*.pt, mario_ppo.pt, or a single *.pt file."
    )


def _config_from_args_json(config_type: type[ConfigT], model_path: Path) -> ConfigT:
    """Reads the sibling args JSON, keeping only keys that are config fields; unknown
    keys are ignored, and a missing file yields plain defaults."""
    cfg = config_type()
    args_path = model_path.with_name("mario_ppo_args.json")
    if not args_path.exists():
        return cfg
    raw = json.loads(args_path.read_text(encoding="utf-8"))
    field_names = {f.name for f in dataclasses.fields(config_type)}
    known = {key: value for key, value in raw.items() if key in field_names}
    return dataclasses.replace(cfg, **known)


def load_args_for_checkpoint(model_path: Path) -> PpoConfig:
    """Loads a checkpoint's training args from its sibling ``mario_ppo_args.json``."""
    return _config_from_args_json(PpoConfig, model_path)


def load_drail_args_for_checkpoint(model_path: Path) -> DrailConfig:
    """Same as ``load_args_for_checkpoint`` but with the ``DrailConfig`` field filter."""
    return _config_from_args_json(DrailConfig, model_path)


def apply_eval_env_paths(
    train_args: ConfigT,
    jar_path_cli: str | None,
    user_dir_cli: str | None,
    *,
    quiet: bool = False,
) -> ConfigT:
    """Resolves jar/user-dir for evaluation: CLI value (must exist) > checkpoint
    value if it exists on disk > repo defaults under ``smb/``."""
    updates: dict[str, str] = {}

    if jar_path_cli:
        jp = resolve_repo_path(str(jar_path_cli).strip())
        if not jp.is_file():
            raise FileNotFoundError(f"--jar-path not found: {jp}")
        updates["jar_path"] = str(jp)
    else:
        j = Path(str(train_args.jar_path)).expanduser()
        j = j.resolve() if j.is_absolute() else (paths.REPO_ROOT / j).resolve()
        if not j.is_file():
            for cand in (
                paths.SMB_DIR / "Mario-AI-Interface.jar",
                paths.SMB_DIR / "Mario-AI-Interface_new.jar",
            ):
                if cand.is_file():
                    if not quiet:
                        print(f"  Checkpoint jar missing ({j}); using {cand}")
                    updates["jar_path"] = str(cand.resolve())
                    break
            else:
                raise FileNotFoundError(
                    f"Jar not found at checkpoint path {j}. "
                    f"Pass --jar-path or install a jar under {paths.SMB_DIR}."
                )

    if user_dir_cli:
        ud = resolve_repo_path(str(user_dir_cli).strip())
        if not ud.is_dir():
            raise FileNotFoundError(f"--user-dir not found: {ud}")
        updates["user_dir"] = str(ud)
    else:
        u = Path(str(train_args.user_dir)).expanduser()
        u = u.resolve() if u.is_absolute() else (paths.REPO_ROOT / u).resolve()
        if not u.is_dir():
            if paths.SMB_DIR.is_dir():
                if not quiet:
                    print(f"  Checkpoint user_dir missing ({u}); using {paths.SMB_DIR}")
                updates["user_dir"] = str(paths.SMB_DIR.resolve())
            else:
                raise FileNotFoundError(f"user_dir not found: {u}. Pass --user-dir.")

    return dataclasses.replace(train_args, **updates) if updates else train_args


def build_env(level_path: Path, args: TrainConfig) -> MarioEnv:
    """Builds a single evaluation env locked to one level.  Frame stacking never
    applies to evaluation: published checkpoints assume unstacked observations."""
    cfg = EnvConfig(
        jar_path=args.jar_path,
        level_dir=str(level_path.parent),
        user_dir=args.user_dir,
        seconds=args.seconds,
        mario_mode=args.mario_mode,
        max_steps=args.max_steps,
        frame_skip=args.frame_skip,
        obs_shape=tuple(args.obs_shape),
        max_id=args.max_id,
        lives=args.lives,
        death_penalty=args.death_penalty,
        completion_reward=args.completion_reward,
        win_reward=args.win_reward,
        kill_reward=args.kill_reward,
        coin_reward=args.coin_reward,
        mushroom_reward=args.mushroom_reward,
        fireflower_reward=args.fireflower_reward,
        brick_reward=args.brick_reward,
        segment_reward=args.segment_reward,
        include_state_features=args.use_state_features,
    )
    env = MarioEnv(cfg)
    env.set_level_files([str(level_path)])
    return env


def make_env_thunk(level_path: Path, args: TrainConfig) -> Callable[[], MarioEnv]:
    """Returns a zero-arg env factory for ``AsyncVectorEnv`` workers."""
    resolved = level_path.resolve()

    def thunk() -> MarioEnv:
        return build_env(resolved, args)

    return thunk


def load_agent(model_path: Path, env: MarioEnv, device: torch.device) -> ActorCritic:
    """Builds an ``ActorCritic`` for ``env`` and loads a checkpoint into it."""
    args = load_args_for_checkpoint(model_path)
    agent = ActorCritic(
        env.observation_space,
        int(env.action_space.n),
        max_id=args.max_id,
        embed_dim=args.embed_dim,
        features_dim=args.features_dim,
        head_hidden_dim=args.head_hidden_dim,
    ).to(device)
    try:
        state = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:  # older torch without weights_only
        state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    agent.load_state_dict(state)
    agent.eval()
    return agent
