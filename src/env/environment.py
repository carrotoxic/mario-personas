"""Policy-facing Gymnasium environment over the Java Mario forward model."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import jpype
import numpy as np

from src.env.actions import ActionSpace
from src.env.config import EnvConfig
from src.env.jvm import get_java_classes, start_jvm
from src.env.levels import LevelSampler
from src.env.rewards import (
    STATUS_LOSE,
    STATUS_TIME_OUT,
    STATUS_WIN,
    RewardState,
    compute_reward,
)
from src.env.segments import (
    enemy_counts_per_segment_from_level_text,
    segments_and_checkpoints_from_level_text,
)


def _reset_debug(msg: str) -> None:
    """Print a reset-path trace when the MARIO_ENV_DEBUG env var is set."""
    if not os.environ.get("MARIO_ENV_DEBUG"):
        return
    print(f"[MarioEnv.reset pid={os.getpid()}] {msg}", flush=True)


class MarioEnv(gym.Env):
    """Headless Mario simulator env with segment timers and shaped rewards.
    Levels are drawn per reset via the global ``random`` module (the Gymnasium
    seed has no effect); each 256 px checkpoint re-arms the full timer budget."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig):
        super().__init__()
        self.cfg = cfg
        start_jvm(cfg.jar_path, user_dir=cfg.user_dir)
        MarioWorld, MarioForwardModel, MarioActions = get_java_classes()
        self.MarioWorld = MarioWorld
        self.MarioForwardModel = MarioForwardModel
        self.actions = ActionSpace(MarioActions)
        self.action_space = self.actions.to_gym_space()
        height, width = cfg.obs_shape
        grid_space = gym.spaces.Box(
            low=0, high=cfg.max_id, shape=(height, width, 1), dtype=np.int16
        )
        if cfg.include_state_features:
            self.observation_space = gym.spaces.Dict(
                {
                    "grid": grid_space,
                    "state": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                }
            )
        else:
            self.observation_space = grid_space
        # Deliberately assignable: any object with select_level/read_level works.
        self.level_loader = LevelSampler(cfg.level_dir)
        self._fm = None
        self._world = None
        self._reward_state = RewardState(0.0, 0, 0, 0, 0, 0, 0)
        self._steps = 0
        self._lives = cfg.lives
        self._current_level_file: str | None = None
        self._checkpoints_px: list[int] = []
        self._furthest_seg_cp = -1
        self._seg_timer_ms = 0
        self._enemies_per_segment: list[int] = []
        self._total_enemies = 0
        self._total_coins = 0
        self._kills_rewarded_this_segment = 0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict[str, Any]]:
        """Load a level, prime the Java world, and return the first observation."""
        t0 = time.perf_counter()
        _reset_debug("enter")
        super().reset(seed=seed)
        self._steps = 0
        self._lives = self.cfg.lives
        selected_level = self.level_loader.select_level()
        _reset_debug(f"select_level -> {selected_level!r} ({time.perf_counter() - t0:.3f}s)")
        level_path = Path(selected_level).resolve()
        self._current_level_file = str(level_path)
        level_text = self.level_loader.read_level(str(level_path))
        _, self._checkpoints_px = segments_and_checkpoints_from_level_text(level_text)
        self._enemies_per_segment = enemy_counts_per_segment_from_level_text(level_text)
        self._total_enemies = int(sum(self._enemies_per_segment))
        # Naive count of coin tiles over the whole level text.
        self._total_coins = int(level_text.count("o"))
        self._furthest_seg_cp = -1
        self._kills_rewarded_this_segment = 0
        self._seg_timer_ms = int(self.cfg.seconds) * 1000
        world = self.MarioWorld(None)
        world.visuals = False
        world.initializeLevel(level_text, self._seg_timer_ms)
        world.mario.isLarge = bool(self.cfg.mario_mode > 0)
        world.mario.isFire = bool(self.cfg.mario_mode > 1)
        world.lives = int(self._lives)
        # A priming NOOP update consumes ~1 tick, so the timer is re-armed
        # after it. The JInt wrapper is required: JPype can mis-coerce a plain
        # Python int on this field assignment.
        world.update(self.actions.get_java_action(0))
        world.currentTimer = jpype.JInt(self._seg_timer_ms)
        self._world = world
        self._fm = self.MarioForwardModel(world)
        self._reward_state = RewardState(
            prev_completion=float(self._fm.getCompletionPercentage()),
            prev_lives=int(world.lives),
            prev_kills=0,
            prev_coins=0,
            prev_mushrooms=0,
            prev_flowers=0,
            prev_bricks=0,
        )
        obs = self._get_observation()
        status = str(world.gameStatus)
        info = {
            "action": self.actions.get_action_name(0),
            "status": status,
            "level_file": str(level_path),
            "lives": int(world.lives),
        }
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray | dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Advance ``cfg.frame_skip`` simulator ticks and score the transition."""
        action = max(0, min(action, self.action_space.n - 1))
        java_action = self.actions.get_java_action(action)
        # Blind frame skip: all ticks always run, no early break on gameStatus
        # (unlike RenderingEnv, which breaks to keep frames aligned).
        for _ in range(int(self.cfg.frame_skip)):
            self._fm.advance(java_action)
        prev_cp = self._furthest_seg_cp
        mario_x = float(self._world.mario.x)
        # Only consecutive checkpoints can be claimed; each claim refreshes the
        # world timer to the full per-segment budget.
        for cp_idx in range(self._furthest_seg_cp + 1, len(self._checkpoints_px)):
            if mario_x >= self._checkpoints_px[cp_idx]:
                self._furthest_seg_cp = cp_idx
                self._world.currentTimer = jpype.JInt(self._seg_timer_ms)
            else:
                break
        checkpoints_passed = self._furthest_seg_cp - prev_cp
        if checkpoints_passed > 0:
            # Kills landed in the same step as a crossing draw from the new
            # segment's budget.
            self._kills_rewarded_this_segment = 0
        seg_idx = self._furthest_seg_cp + 1
        enemies_here = (
            self._enemies_per_segment[seg_idx] if seg_idx < len(self._enemies_per_segment) else 0
        )
        kill_credits_left = max(0, enemies_here - self._kills_rewarded_this_segment)
        # The cap only applies to strictly positive kill rewards; a negative
        # kill_reward is passed uncapped and the rewarded count is not tracked.
        kill_cap_arg = kill_credits_left if self.cfg.kill_reward > 0.0 else None
        status = str(self._world.gameStatus)
        self._steps += 1
        terminated = status == STATUS_WIN or status == STATUS_LOSE
        truncated = status == STATUS_TIME_OUT or self._steps >= int(self.cfg.max_steps)
        reward, reward_details, self._reward_state = compute_reward(
            self.cfg,
            self._world,
            self._fm,
            self._reward_state,
            steps=self._steps,
            max_steps=int(self.cfg.max_steps),
            checkpoints_passed_this_step=checkpoints_passed,
            kill_reward_credits_remaining=kill_cap_arg,
        )
        if self.cfg.kill_reward > 0.0:
            self._kills_rewarded_this_segment += int(reward_details["dkills_rewarded"])
        completion = float(reward_details["completion"])
        current_lives = int(reward_details["lives"])
        current_kills = int(reward_details["kills"])
        current_coins = int(reward_details["coins"])
        current_mushrooms = int(reward_details["mushrooms"])
        current_flowers = int(reward_details["flowers"])
        current_bricks = int(reward_details["bricks"])
        kill_ratio = (
            float(current_kills) / float(self._total_enemies) if self._total_enemies > 0 else 0.0
        )
        coin_ratio = (
            float(current_coins) / float(self._total_coins) if self._total_coins > 0 else 0.0
        )
        obs = self._get_observation()
        info = {
            "action": self.actions.get_action_name(action),
            "status": status,
            "completion": completion,
            "kill_ratio": kill_ratio,
            "coin_ratio": coin_ratio,
            "lives": current_lives,
            "kills": current_kills,
            "coins": current_coins,
            "mushrooms": current_mushrooms,
            "flowers": current_flowers,
            "bricks": current_bricks,
            "reward_details": reward_details,
        }
        # Only terminal steps carry the level path; episode-end bookkeeping
        # (e.g. marking a level as won) keys off its presence.
        if terminated or truncated:
            info["level_file"] = self._current_level_file
        return obs, reward, terminated, truncated, info

    def _get_state_features(self, mario_mode: int) -> np.ndarray:
        """Return ``[tanh(vx/8), tanh(vy/8), mode - 1, on_ground]`` (float32)."""
        mario = self._world.mario
        vx = float(getattr(mario, "xa", 0.0))
        vy = float(getattr(mario, "ya", 0.0))
        on_ground = 1.0 if bool(getattr(mario, "onGround", False)) else 0.0
        return np.array(
            [
                float(np.tanh(vx / 8.0)),
                float(np.tanh(vy / 8.0)),
                float(mario_mode - 1),
                on_ground,
            ],
            dtype=np.float32,
        )

    def _get_observation(self) -> np.ndarray | dict[str, np.ndarray]:
        # Detail level 0 = complete tile/enemy ids.
        raw_grid = np.array(self._fm.getMarioCompleteObservation(0, 0), dtype=np.int16)
        # Java grids are [x][y]-indexed; transpose to (H, W), add channel axis.
        grid = raw_grid.T[..., np.newaxis]
        if not self.cfg.include_state_features:
            return grid
        mario = self._world.mario
        mario_mode = 2 if mario.isFire else 1 if mario.isLarge else 0
        return {"grid": grid, "state": self._get_state_features(mario_mode)}

    def set_level_files(self, level_files: list[str]) -> None:
        """Restrict the sampler to ``level_files``."""
        self.level_loader.set_level_files(level_files)

    def render(self) -> None:
        """No-op; use :class:`src.env.rendering.RenderingEnv` for frames."""
        return None

    def close(self) -> None:
        """No-op; the JVM is process-wide and never shut down."""
        return None
