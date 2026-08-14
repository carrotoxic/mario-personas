"""Competence evaluation CLI for a Mario PPO checkpoint: plays every level in
``--level-dir`` for ``--repeats`` episodes and writes ``evaluation_results_*.json``
next to the checkpoint.  One-life protocol by default (``--lives 0``); requires CUDA."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch

from src import paths
from src.evaluation import evaluation_results_output_path, loading, rollouts
from src.evaluation.rollouts import LevelSummary


def print_level_report(result: LevelSummary) -> None:
    level_name = Path(result.level).name
    status = "PLAYABLE" if result.playable else "NOT PLAYABLE"
    print(
        f"  {level_name}: {status} | wins={result.win_count}/{result.run_count} | "
        f"best_completion={result.best_completion:.3f} | "
        f"best_kill_ratio={result.best_kill_ratio:.3f} | "
        f"best_coin_ratio={result.best_coin_ratio:.3f} | "
        f"avg_reward={result.avg_reward:.3f} | avg_steps={result.avg_length:.1f}"
    )


def print_folder_report(results: list[LevelSummary], model_path: Path, level_dir: Path) -> None:
    playable_count = sum(1 for item in results if item.playable)
    n = len(results)
    mean_best_completion = sum(item.best_completion for item in results) / n
    mean_best_kill_ratio = sum(item.best_kill_ratio for item in results) / n
    mean_best_coin_ratio = sum(item.best_coin_ratio for item in results) / n
    avg_reward = sum(item.avg_reward for item in results) / n
    avg_length = sum(item.avg_length for item in results) / n
    print("\nEvaluation Summary")
    print(f"  Model: {model_path}")
    print(f"  Levels directory: {level_dir}")
    print(f"  Levels evaluated: {n}")
    print(f"  Playable levels: {playable_count}/{n} ({playable_count / n:.1%})")
    print(f"  Mean best completion (over levels): {mean_best_completion:.3f}")
    print(f"  Mean best kill ratio (over levels): {mean_best_kill_ratio:.3f}")
    print(f"  Mean best coin ratio (over levels): {mean_best_coin_ratio:.3f}")
    print(f"  Average reward: {avg_reward:.3f}")
    print(f"  Average steps: {avg_length:.1f}")


def save_evaluation_results(
    results: list[LevelSummary],
    model_path: Path,
    level_dir: Path,
    repeats: int,
    deterministic: bool,
    results_name_tag: str = "train",
) -> Path:
    # The payload schema is a downstream contract; win_count is intentionally
    # absent from per-level entries (only win_rate is kept).
    level_summaries = [
        {
            "level": item.level,
            "playable": item.playable,
            "run_count": item.run_count,
            "win_rate": float(item.win_rate),
            "best_completion": float(item.best_completion),
            "best_kill_ratio": float(item.best_kill_ratio),
            "best_coin_ratio": float(item.best_coin_ratio),
            "avg_reward": float(item.avg_reward),
            "avg_length": float(item.avg_length),
        }
        for item in results
    ]
    playable_count = sum(1 for item in results if item.playable)
    n_lv = len(results)
    summary = {
        "model_path": str(model_path),
        "level_dir": str(level_dir),
        "levels_evaluated": n_lv,
        "playable_levels": playable_count,
        "playable_fraction": float(playable_count / n_lv),
        "mean_best_completion": float(sum(item.best_completion for item in results) / n_lv),
        "mean_best_kill_ratio": float(sum(item.best_kill_ratio for item in results) / n_lv),
        "mean_best_coin_ratio": float(sum(item.best_coin_ratio for item in results) / n_lv),
        "mean_reward": float(sum(item.avg_reward for item in results) / n_lv),
        "mean_steps": float(sum(item.avg_length for item in results) / n_lv),
    }
    output_path = evaluation_results_output_path(
        model_path, level_dir, results_name_tag=results_name_tag
    )
    payload = {
        "timestamp": int(time.time()),
        "repeats": int(repeats),
        "deterministic": bool(deterministic),
        "evaluation_results_file": output_path.name,
        "results_name_tag": str(results_name_tag),
        "levels": level_summaries,
        "summary": summary,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Mario PPO checkpoint.")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to mario_ppo.pt or a run directory containing it.",
    )
    parser.add_argument(
        "--level-dir",
        default=str(paths.DATA_DIR / "human_experiment" / "levels" / "generation2"),
        help="Folder of .lvl files for batch evaluation.",
    )
    parser.add_argument(
        "--repeats", type=int, default=10, help="Number of episodes per level."
    )
    parser.add_argument(
        "--lives",
        type=int,
        default=0,
        help="Override number of lives used during evaluation.",
    )
    parser.add_argument(
        "--jar-path",
        default=None,
        help=(
            "Mario-AI-Interface jar (overrides checkpoint path; required if "
            "checkpoint points to another machine)."
        ),
    )
    parser.add_argument(
        "--user-dir",
        default=None,
        help="Java user.dir / Assets root, usually repo smb/ (overrides checkpoint when set).",
    )
    parser.add_argument("--deterministic", action="store_true", help="Use argmax policy actions.")
    parser.add_argument(
        "--results-name-tag",
        type=str,
        default="train",
        metavar="TAG",
        help=(
            "Inserted in saved JSON basename, e.g. ..._collector_train_1e7steps vs "
            "..._collector_test_1e7steps (default: train)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    model_path = loading.resolve_model_path(args.model_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation, but no CUDA device is available.")
    device = torch.device("cuda")

    level_dir = Path(args.level_dir).resolve()
    # Sorted order defines level_idx and thus the seeding below — load-bearing
    # for reproducibility.
    level_paths = sorted([*level_dir.glob("*.lvl"), *level_dir.glob("*.txt")])
    if not level_paths:
        raise FileNotFoundError(f"No .lvl or .txt level files found in {level_dir}")

    train_args = loading.load_args_for_checkpoint(model_path)
    # --lives always overrides checkpoint lives (default 0 = one-life protocol).
    train_args = dataclasses.replace(train_args, lives=int(args.lives))
    print(f"  Using lives override: {train_args.lives}")
    train_args = loading.apply_eval_env_paths(train_args, args.jar_path, args.user_dir)

    n_envs = max(1, int(args.repeats))
    probe_env = loading.build_env(level_paths[0], train_args)
    agent = loading.load_agent(model_path, probe_env, device)
    probe_env.close()

    envs = rollouts.make_async_envs(
        [loading.make_env_thunk(level_paths[0], train_args) for _ in range(n_envs)]
    )
    results: list[LevelSummary] = []
    try:
        for level_idx, level_path in enumerate(level_paths):
            result = rollouts.evaluate_level(
                level_path,
                envs,
                agent,
                repeats=args.repeats,
                deterministic=args.deterministic,
                device=device,
                seed_base=int(train_args.seed) + level_idx * n_envs,
                max_steps=int(train_args.max_steps) if int(train_args.max_steps) > 0 else 3000,
            )
            results.append(result)
            print_level_report(result)
    finally:
        envs.close()

    print_folder_report(results, model_path, level_dir)
    result_path = save_evaluation_results(
        results,
        model_path,
        level_dir,
        args.repeats,
        args.deterministic,
        results_name_tag=args.results_name_tag,
    )
    print(f"  Saved evaluation JSON: {result_path}")


if __name__ == "__main__":
    main()
