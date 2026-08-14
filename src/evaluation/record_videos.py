"""Best-of-N showcase video recorder: per (method, persona, level), N stochastic
attempts run headless, and the best per persona metric is replayed through
``RenderingEnv`` (replay-deterministic sim) into an mp4 plus a global manifest."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

from src import paths
from src.env import EnvConfig, RenderingEnv, jvm
from src.evaluation import loading, rollouts
from src.models import ActorCritic

METHODS = ("ppo_50k", "direct_drail", "ppo_to_drail")
ROLES = ("runner", "killer", "collector")
# One-life protocol for PPO, five-life for the DRAIL variants: a paper-level
# protocol decision, not a tunable.
LIVES_BY_METHOD = {"ppo_50k": 0, "direct_drail": 5, "ppo_to_drail": 5}


def find_model(method: str, role: str, ckpt_dir: Path) -> Path:
    if method == "ppo_50k":
        # Showcase always uses the 200M-step PPO checkpoint of the 50k-level runs.
        p = ckpt_dir / "ppo" / role / "mario_ppo_step_200000000.pt"
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    sub = "drail" if method == "direct_drail" else "ppo_to_drail"
    cands = [p for p in (ckpt_dir / sub / role).glob("*.pt") if "discriminator" not in p.name]
    if len(cands) != 1:
        raise FileNotFoundError(
            f"Expected exactly one policy .pt in {ckpt_dir / sub / role}, found {cands}"
        )
    return cands[0]


def load_method(
    method: str,
    role: str,
    device: torch.device,
    *,
    timer: int,
    max_steps: int,
    probe_level: Path,
    ckpt_dir: Path,
) -> tuple[Path, loading.TrainConfig, ActorCritic]:
    model_path = find_model(method, role, ckpt_dir)
    if method == "ppo_50k":
        train_args: loading.TrainConfig = loading.load_args_for_checkpoint(model_path)
    else:
        train_args = loading.load_drail_args_for_checkpoint(model_path)
    # The same render-capable jar drives scoring workers and replay rendering
    # so dynamics match exactly.
    train_args = dataclasses.replace(
        train_args,
        jar_path=str(paths.RENDER_JAR),
        user_dir=str(paths.SMB_DIR),
        seconds=int(timer),
        max_steps=int(max_steps),
        lives=LIVES_BY_METHOD[method],
    )
    probe_env = loading.build_env(probe_level, train_args)
    agent = loading.load_agent(model_path, probe_env, device)
    probe_env.close()
    return model_path, train_args, agent


def run_attempts(
    envs: gym.vector.AsyncVectorEnv,
    agent: ActorCritic,
    level_path: Path,
    n_envs: int,
    device: torch.device,
    max_steps: int,
) -> tuple[list[list[int]], list[dict]]:
    envs.call("set_level_files", [str(level_path)])
    # Resets are deliberately unseeded (best-of-N is not meant to be reproducible).
    obs, _ = envs.reset()
    actions_by_env: list[list[int]] = [[] for _ in range(n_envs)]
    metrics: list[dict | None] = [None] * n_envs
    finished = np.zeros(n_envs, dtype=bool)
    rewards_acc = np.zeros(n_envs, dtype=np.float32)
    lengths = np.zeros(n_envs, dtype=np.int32)

    # Loop bound: two steps of slack beyond the env step cap.
    for _ in range(int(max_steps) + 2):
        if finished.all():
            break
        obs_t = rollouts.batch_obs_to_torch(obs, device)
        action = rollouts.select_actions(agent, obs_t, deterministic=False)
        act_np = action.detach().cpu().numpy().astype(np.int64)
        for i in np.where(~finished)[0]:
            actions_by_env[i].append(int(act_np[i]))
        obs, rew, term, trunc, infos = envs.step(act_np)
        done = np.logical_or(term, trunc)
        active = ~finished
        rewards_acc[active] += np.asarray(rew, dtype=np.float32)[active]
        lengths[active] += 1

        for i in np.where(done & active)[0].tolist():
            fi = rollouts.final_info_dict(infos, i, n_envs)
            status = str(fi.get("status", rollouts.final_info_value(infos, "status", i, "UNKNOWN")))
            metrics[i] = {
                "status": status,
                "completion": float(
                    fi.get("completion", rollouts.final_info_value(infos, "completion", i, 0.0))
                ),
                "kill_ratio": float(
                    fi.get("kill_ratio", rollouts.final_info_value(infos, "kill_ratio", i, 0.0))
                ),
                "coin_ratio": float(
                    fi.get("coin_ratio", rollouts.final_info_value(infos, "coin_ratio", i, 0.0))
                ),
                "kills": int(fi.get("kills", rollouts.final_info_value(infos, "kills", i, 0))),
                "coins": int(fi.get("coins", rollouts.final_info_value(infos, "coins", i, 0))),
                "length": int(lengths[i]),
                "reward": float(rewards_acc[i]),
            }
            finished[i] = True

    for i in np.where(~finished)[0].tolist():
        metrics[i] = {
            "status": "TIMEOUT",
            "completion": 0.0,
            "kill_ratio": 0.0,
            "coin_ratio": 0.0,
            "kills": 0,
            "coins": 0,
            "length": int(lengths[i]),
            "reward": float(rewards_acc[i]),
        }
    return actions_by_env, metrics


# Best-attempt sort key per persona: win first, then the persona metric.
def attempt_key(role: str, m: dict) -> tuple:
    win = 1 if m["status"] == "WIN" else 0
    if role == "runner":
        return (win, m["completion"], -m["length"], m["reward"])
    if role == "killer":
        return (win, m["kill_ratio"], m["completion"], m["reward"])
    return (win, m["coin_ratio"], m["completion"], m["reward"])


def render_replay(
    actions: list[int], level_file: Path, train_args: loading.TrainConfig
) -> tuple[list[np.ndarray], dict]:
    env_cfg = EnvConfig(
        jar_path=str(paths.RENDER_JAR),
        level_dir=str(level_file.parent),
        user_dir=str(paths.SMB_DIR),
        max_steps=int(train_args.max_steps),
        seconds=int(train_args.seconds),
        mario_mode=int(train_args.mario_mode),
        max_id=int(train_args.max_id),
        obs_shape=tuple(train_args.obs_shape),
        include_state_features=bool(train_args.use_state_features),
        lives=int(train_args.lives),
        frame_skip=int(train_args.frame_skip),
    )
    renv = RenderingEnv(level_file, env_cfg, lives=int(train_args.lives), render=True)
    renv.reset()
    frames: list[np.ndarray] = []
    f0 = renv.render_frame()
    if f0 is not None:
        frames.append(f0)
    info: dict = {}
    for a in actions:
        _obs, _r, term, trunc, info = renv.step(int(a))
        frames.extend(renv.get_step_frames())
        if term or trunc:
            break
    renv.close()
    return frames, dict(info)


def save_video_from_frames(frames: list[np.ndarray], out_path: Path, fps: int) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=int(fps), codec="libx264", quality=8)
    return out_path


def _repo_relative(p: Path) -> str:
    try:
        return str(p.relative_to(paths.REPO_ROOT))
    except ValueError:
        return str(p)


def _load_level_files(levels_json: Path, max_levels: int) -> list[Path]:
    spec = json.loads(levels_json.read_text(encoding="utf-8"))
    level_dir = Path(spec["level_dir"])
    if not level_dir.is_absolute():
        level_dir = (paths.REPO_ROOT / level_dir).resolve()
    level_files: list[Path] = []
    for stem in spec["levels"]:
        for ext in (".lvl", ".txt"):
            p = level_dir / f"{stem}{ext}"
            if p.is_file():
                level_files.append(p)
                break
        else:
            raise FileNotFoundError(f"Level {stem} not found in {level_dir}")
    if max_levels > 0:
        level_files = level_files[:max_levels]
    return level_files


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record best-of-N showcase videos per method/persona/level."
    )
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--roles", default=",".join(ROLES))
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument(
        "--levels-json",
        default=str(paths.REPO_ROOT / "human_like_rl" / "evaluation" / "showcase_levels.json"),
    )
    parser.add_argument("--out-dir", default=str(paths.REPO_ROOT / "results" / "showcase_videos"))
    parser.add_argument("--ckpt-dir", default=str(paths.REPO_ROOT / "human_like_rl" / "checkpoints"))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--timer", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument(
        "--max-levels",
        type=int,
        default=0,
        help="Limit to first N levels (0 = all), for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    ckpt_dir = Path(args.ckpt_dir)
    level_files = _load_level_files(Path(args.levels_json), int(args.max_levels))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest: dict = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[showcase] device={device}, methods={methods}, roles={roles}, "
        f"levels={len(level_files)}, attempts={args.attempts}",
        flush=True,
    )

    # JPype fixes user.dir once per JVM and the first start wins: the main
    # process must start the render-capable jar before any env work, or
    # RenderingEnv breaks. Spawned workers start their own JVMs.
    jvm.start_jvm(str(paths.RENDER_JAR), user_dir=str(paths.SMB_DIR))

    n_envs = max(1, int(args.attempts))
    for method in methods:
        for role in roles:
            todo = [
                lf
                for lf in level_files
                if not (
                    (out_dir / method / role / f"{lf.stem}.mp4").is_file()
                    and manifest.get(method, {}).get(role, {}).get(lf.stem)
                )
            ]
            if not todo:
                print(f"[showcase] {method}/{role}: all levels done, skipping", flush=True)
                continue
            t0 = time.time()
            model_path, train_args, agent = load_method(
                method,
                role,
                device,
                timer=args.timer,
                max_steps=args.max_steps,
                probe_level=level_files[0],
                ckpt_dir=ckpt_dir,
            )
            print(
                f"[showcase] {method}/{role}: model={model_path.name}, lives={train_args.lives}, "
                f"{len(todo)} level(s) to do",
                flush=True,
            )
            envs = rollouts.make_async_envs(
                [loading.make_env_thunk(level_files[0], train_args) for _ in range(n_envs)]
            )
            try:
                for lf in todo:
                    t1 = time.time()
                    actions_by_env, metrics = run_attempts(
                        envs, agent, lf, n_envs, device, args.max_steps
                    )
                    best = max(range(n_envs), key=lambda i: attempt_key(role, metrics[i]))
                    bm = metrics[best]
                    frames, replay_info = render_replay(actions_by_env[best], lf, train_args)
                    video_path = out_dir / method / role / f"{lf.stem}.mp4"
                    save_video_from_frames(frames, video_path, args.fps)

                    # Replay accepted when status matches and completion agrees
                    # within 0.02.
                    replay_status = str(replay_info.get("status", "UNKNOWN"))
                    replay_ok = (
                        replay_status == bm["status"]
                        and abs(float(replay_info.get("completion", 0.0)) - bm["completion"])
                        < 0.02
                    )
                    if not replay_ok:
                        replay_completion = float(replay_info.get("completion", 0.0))
                        print(
                            f"[showcase] WARNING: replay divergence on {lf.stem}: "
                            f"recorded={bm['status']}/{bm['completion']:.3f} "
                            f"replay={replay_status}/{replay_completion:.3f}",
                            flush=True,
                        )

                    manifest.setdefault(method, {}).setdefault(role, {})[lf.stem] = {
                        "video": _repo_relative(video_path),
                        "model": _repo_relative(model_path),
                        "lives": int(train_args.lives),
                        "timer_seconds": int(train_args.seconds),
                        "attempts": metrics,
                        "best_attempt": int(best),
                        "best_metrics": bm,
                        "replay_status": replay_status,
                        "replay_completion": float(replay_info.get("completion", 0.0)),
                        "replay_matches": bool(replay_ok),
                        "n_frames": len(frames),
                    }
                    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
                    wins = sum(1 for m in metrics if m["status"] == "WIN")
                    print(
                        f"[showcase] {method}/{role}/{lf.stem}: wins={wins}/{n_envs} "
                        f"best[{best}]={bm['status']} compl={bm['completion']:.2f} "
                        f"kill={bm['kill_ratio']:.2f} coin={bm['coin_ratio']:.2f} "
                        f"steps={bm['length']} "
                        f"replay_ok={replay_ok} ({time.time() - t1:.0f}s)",
                        flush=True,
                    )
            finally:
                envs.close()
            print(
                f"[showcase] {method}/{role} finished in {(time.time() - t0) / 60:.1f} min",
                flush=True,
            )

    print("[showcase] all done.", flush=True)


if __name__ == "__main__":
    main()
