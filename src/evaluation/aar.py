"""Human-likeness evaluation CLI for killer/collector/runner PPO checkpoints: per-role
AAR on stored human observations plus rollout-based Action JS divergence.  Writes
``aar_*.json`` when ``--output-json`` is set; ``--aar-only`` skips the Java rollouts."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src import paths
from src.evaluation import loading, resolve_repo_path, rollouts
from src.evaluation.similarity import (
    ActionsCache,
    ValidationSample,
    action_match_rate,
    build_validation_samples,
    compute_action_js_metrics,
    rollout_cache_key,
)
from src.models import ActorCritic
from src.training.config import PpoConfig

# Role processing order (also the macro-average order in the output JSON).
ROLES = ("killer", "collector", "runner")


# ---------------------------------------------------------------------------
# AAR on stored human observations.
# ---------------------------------------------------------------------------


def policy_actions_on_human_obs(
    agent: ActorCritic,
    sample: ValidationSample,
    device: torch.device,
    *,
    deterministic: bool,
) -> np.ndarray:
    obs_t = {
        "grid": torch.as_tensor(sample.human_grid, device=device),
        "state": torch.as_tensor(sample.human_state, device=device, dtype=torch.float32),
    }
    action = rollouts.select_actions(agent, obs_t, deterministic=deterministic)
    return action.detach().cpu().numpy().astype(np.int64)


# Stochastic mode samples actions, so mean AAR differs between calls — reported as-is.
def compute_aar_per_sample(
    samples: list[ValidationSample],
    agent: ActorCritic,
    device: torch.device,
    *,
    deterministic: bool,
) -> tuple[float, list[dict]]:
    aar_vals: list[float] = []
    rows: list[dict] = []
    for sample in samples:
        pred = policy_actions_on_human_obs(agent, sample, device, deterministic=deterministic)
        match = action_match_rate(pred, sample.human_actions)
        if match is None:
            continue
        m = int(min(len(pred), len(sample.human_actions)))
        aar_vals.append(match)
        rows.append({"level": sample.level_path.name, "aar": match, "steps": m})
    mean_aar = float(np.mean(aar_vals)) if aar_vals else float("nan")
    return mean_aar, rows


# ---------------------------------------------------------------------------
# Policy rollout cache (one episode per sample, for Action JS).
# ---------------------------------------------------------------------------


def _policy_action_source(
    agent: ActorCritic, device: torch.device, *, deterministic: bool
) -> rollouts.ActionSource:
    def source(obs: dict, active_idx: np.ndarray, n_actions: int) -> np.ndarray:
        del n_actions
        obs_t = {
            "grid": torch.as_tensor(obs["grid"][active_idx], device=device),
            "state": torch.as_tensor(
                obs["state"][active_idx], device=device, dtype=torch.float32
            ),
        }
        action = rollouts.select_actions(agent, obs_t, deterministic=deterministic)
        return action.detach().cpu().numpy().astype(np.int64)

    return source


def build_policy_rollout_cache(
    samples: list[ValidationSample],
    train_args: PpoConfig,
    agent: ActorCritic,
    device: torch.device,
    num_envs: int,
    *,
    deterministic: bool,
) -> ActionsCache:
    """Rolls out the policy once per sample, in consecutive batches of ``num_envs``."""
    source = _policy_action_source(agent, device, deterministic=deterministic)
    cache: ActionsCache = {}
    n_envs = max(1, int(num_envs))
    for start in range(0, len(samples), n_envs):
        batch = samples[start : start + n_envs]
        traces = rollouts.rollout_actions(
            [s.level_path for s in batch],
            [s.eval_seed for s in batch],
            train_args,
            source,
        )
        for sample, actions in zip(batch, traces):
            cache[rollout_cache_key(sample)] = actions
    return cache


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _find_role_subdirs(checkpoint_root: Path) -> dict[str, Path]:
    checkpoint_root = checkpoint_root.resolve()
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_root}")
    by_lower = {d.name.lower(): d for d in checkpoint_root.iterdir() if d.is_dir()}
    out: dict[str, Path] = {}
    missing: list[str] = []
    for role in ROLES:
        d = by_lower.get(role.lower())
        if d is None:
            missing.append(role)
        else:
            out[role] = d
    if missing:
        raise FileNotFoundError(
            f"Missing role subfolder(s) {missing} under {checkpoint_root}. "
            f"Expected directories named {list(ROLES)} (case-insensitive)."
        )
    return out


def _sanitize_json(x: Any) -> Any:
    # NaN/Inf -> None recursively for strict-JSON output.
    if isinstance(x, dict):
        return {k: _sanitize_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_sanitize_json(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AAR and Action JS divergence for killer/collector/runner PPO checkpoints."
        )
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Folder containing killer/, collector/, runner/ each with a PPO .pt checkpoint.",
    )
    parser.add_argument(
        "--action-state-dir",
        type=str,
        default=str(
            paths.DATA_DIR / "human_experiment" / "action_state" / "action_state_test"
        ),
        help="Directory of human *.npz traces (default: action_state_test).",
    )
    parser.add_argument(
        "--level-dir",
        type=str,
        default=str(paths.DATA_DIR / "human_experiment" / "levels" / "expert_levels"),
        help=(
            "Folder of .lvl files used to resolve level paths from npz stems "
            "(default: expert_levels)."
        ),
    )
    parser.add_argument(
        "--lives",
        type=int,
        default=5,
        help="Mario lives for env rollouts (applied after loading checkpoint args; default: 5).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=10,
        metavar="N",
        help="Parallel AsyncVectorEnv workers for rollout collection (default: 10).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Base seed for eval_seed in validation samples."
    )
    parser.add_argument(
        "--jar-path", type=str, default=None, help="Override Mario-AI-Interface jar path."
    )
    parser.add_argument(
        "--user-dir", type=str, default=None, help="Override Java user.dir (usually repo smb/)."
    )
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of CUDA.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use greedy argmax actions; default is stochastic (sample from policy logits).",
    )
    parser.add_argument(
        "--aar-only",
        action="store_true",
        help="Only compute AAR from stored human obs (no Java rollouts, no Action JS). Much faster.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="If set, write full results (per-role metrics + per-npz AAR) to this path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    checkpoint_root = resolve_repo_path(args.checkpoint_dir)
    action_state_dir = resolve_repo_path(args.action_state_dir)
    level_dir = resolve_repo_path(args.level_dir)

    samples = build_validation_samples(action_state_dir, level_dir, seed=int(args.seed))
    if not samples:
        raise RuntimeError(
            f"No validation samples from {action_state_dir} with levels in {level_dir}. "
            "Check npz contents (grid/state/actions) and level naming (_lvlNN in stem)."
        )
    print(
        f"Using lives={int(args.lives)} for rollouts; num_envs={int(args.num_envs)}; "
        f"policy={'deterministic' if args.deterministic else 'stochastic'}; "
        f"{len(samples)} samples; level_dir={level_dir}"
    )

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    role_dirs = _find_role_subdirs(checkpoint_root)
    first_level = samples[0].level_path
    results: dict[str, Any] = {
        "checkpoint_dir": str(checkpoint_root),
        "action_state_dir": str(action_state_dir),
        "level_dir": str(level_dir),
        "lives": int(args.lives),
        "num_envs": int(args.num_envs),
        "n_samples": len(samples),
        "agent_type": "ppo",
        "aar_only": bool(args.aar_only),
        "deterministic": bool(args.deterministic),
        "per_role": {},
    }

    for role, role_dir in role_dirs.items():
        ckpt = loading.pick_role_checkpoint(role_dir)
        train_args = loading.load_args_for_checkpoint(ckpt)
        train_args = loading.apply_eval_env_paths(
            train_args, args.jar_path, args.user_dir, quiet=True
        )
        train_args = dataclasses.replace(train_args, lives=int(args.lives))
        probe = loading.build_env(first_level, train_args)
        n_actions = int(probe.action_space.n)
        agent = loading.load_agent(ckpt, probe, device)
        probe.close()

        mean_aar, details = compute_aar_per_sample(
            samples, agent, device, deterministic=bool(args.deterministic)
        )
        role_payload: dict[str, Any] = {
            "checkpoint": str(ckpt),
            "mean_aar_action": mean_aar,
            "per_npz": details,
        }

        if not args.aar_only:
            policy_cache = build_policy_rollout_cache(
                samples,
                train_args,
                agent,
                device,
                int(args.num_envs),
                deterministic=bool(args.deterministic),
            )

            def sample_rollout(sample: ValidationSample) -> np.ndarray:
                return policy_cache[rollout_cache_key(sample)]

            beh = compute_action_js_metrics(
                samples, action_dim=n_actions, sample_rollout=sample_rollout
            )
            role_payload["behavior_similarity"] = {k: float(v) for k, v in beh.items()}
            del policy_cache

        results["per_role"][role] = role_payload
        # Keep VRAM bounded across the three roles.
        del agent
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if args.aar_only:
            print(f"{role}: mean_aar={mean_aar:.6f}  ({ckpt.name})")
        else:
            beh = role_payload.get("behavior_similarity", {})
            beh_s = ", ".join(f"{k}={beh[k]:.4f}" for k in sorted(beh.keys()))
            print(f"{role}: mean_aar={mean_aar:.6f}  {beh_s}  ({ckpt.name})")

    means = [
        float(results["per_role"][r]["mean_aar_action"]) for r in ROLES if r in results["per_role"]
    ]
    macro = float(np.nanmean(np.asarray(means, dtype=np.float64))) if means else float("nan")
    results["mean_aar_macro_avg_roles"] = macro
    print(f"macro_avg_roles (mean_aar): {macro:.6f}")

    if not args.aar_only:
        sim_keys = sorted(
            {
                k
                for r in ROLES
                for k in ((results["per_role"].get(r) or {}).get("behavior_similarity") or {})
            }
        )
        macro_by: dict[str, float] = {}
        for key in sim_keys:
            vals = [
                float(beh[key])
                for r in ROLES
                for beh in ((results["per_role"].get(r) or {}).get("behavior_similarity") or {},)
                if key in beh
            ]
            macro_by[key] = (
                float(np.nanmean(np.asarray(vals, dtype=np.float64))) if vals else float("nan")
            )
        results["macro_avg_roles_by_metric"] = macro_by
        print("macro_avg_roles (Action JS):")
        for k in sim_keys:
            print(f"  {k}: {macro_by[k]:.6f}")

    if args.output_json:
        out_path = resolve_repo_path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(_sanitize_json(results), indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
