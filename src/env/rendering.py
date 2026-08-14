"""Frame-capturing env using Java ``MarioForwardModel.toImage()``.
Requires a jar shipping ``toImage()`` (see :data:`src.paths.RENDER_JAR`); the
first JVM start pins the classpath, so the first env constructed decides rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import jpype
import numpy as np

from src.env.actions import ActionSpace
from src.env.config import EnvConfig
from src.env.jvm import get_java_classes, start_jvm
from src.env.rewards import STATUS_LOSE, STATUS_RUNNING, STATUS_TIME_OUT, STATUS_WIN

_MS_PER_SECOND = 1000
_NOOP_ACTION = 0
# Detail levels for get*CompleteObservation: 0 = complete tile/enemy ids.
_COMPLETE_DETAIL = 0
_VELOCITY_TANH_SCALE = 8.0
_NUM_STATE_FEATURES = 6
_NUM_GRID_CHANNELS = 2
_CHANNEL_MASK = 0xFF
# Showcase reward terms, deliberately independent of EnvConfig.
_WIN_REWARD = 1.0
_TIMEOUT_PENALTY = 1.0
_DEFAULT_LIVES = 2


def _java_buffered_image_to_numpy(jimg: Any) -> np.ndarray:
    """Convert a Java ``BufferedImage`` to an ``(h, w, 3)`` uint8 RGB array."""
    w, h = int(jimg.getWidth()), int(jimg.getHeight())
    java_int_array = jimg.getRGB(0, 0, w, h, None, 0, w)
    # Per-pixel Python conversion of the ARGB int[] — slow but exact.
    argb = np.array([int(v) for v in java_int_array], dtype=np.int32)
    r = ((argb >> 16) & _CHANNEL_MASK).astype(np.uint8)
    g = ((argb >> 8) & _CHANNEL_MASK).astype(np.uint8)
    b = (argb & _CHANNEL_MASK).astype(np.uint8)
    return np.stack([r, g, b], axis=1).reshape(h, w, 3)


class RenderingEnv(gym.Env):
    """Single-level env that captures an RGB frame per simulated tick.
    Deliberately diverges from :class:`src.env.environment.MarioEnv`: two grid
    channels + 6 state features, one un-refreshed timer budget, its own reward."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        level_file: Path | str,
        cfg: EnvConfig,
        lives: int = _DEFAULT_LIVES,
        render: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.level_file = Path(level_file).resolve()
        self.lives = lives
        self.render_enabled = render
        start_jvm(cfg.jar_path, user_dir=cfg.user_dir)
        MarioWorld, MarioForwardModel, MarioActions = get_java_classes()
        self.MarioWorld = MarioWorld
        self.MarioForwardModel = MarioForwardModel
        self.actions = ActionSpace(MarioActions)
        self.action_space = self.actions.to_gym_space()
        height, width = cfg.obs_shape
        grid_space = gym.spaces.Box(
            low=0, high=cfg.max_id, shape=(height, width, _NUM_GRID_CHANNELS), dtype=np.int16
        )
        if cfg.include_state_features:
            self.observation_space = gym.spaces.Dict(
                {
                    "grid": grid_space,
                    "state": gym.spaces.Box(
                        low=-1.0, high=1.0, shape=(_NUM_STATE_FEATURES,), dtype=np.float32
                    ),
                }
            )
        else:
            self.observation_space = grid_space
        self._world = None
        self._fm = None
        self._prev_completion = 0.0
        self._prev_lives = lives
        self._prev_kills = 0
        self._prev_coins = 0
        self._prev_mushrooms = 0
        self._prev_flowers = 0
        self._prev_bricks = 0
        self._steps = 0
        self._step_frames: list[np.ndarray] = []

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict[str, Any]]:
        """Initialize the world with visuals and return the first observation."""
        super().reset(seed=seed)
        level_str = self.level_file.read_text(encoding="utf-8")
        world = self.MarioWorld(None)
        world.visuals = True
        BufferedImage = jpype.JClass("java.awt.image.BufferedImage")
        GraphicsEnvironment = jpype.JClass("java.awt.GraphicsEnvironment")
        Assets = jpype.JClass("engine.helper.Assets")
        # Prefer the real screen device; on any failure (headless hosts raise
        # assorted Java exceptions through JPype) fall back to an off-screen
        # 1x1 image's device configuration.
        try:
            ge = GraphicsEnvironment.getLocalGraphicsEnvironment()
            graphics_config = ge.getDefaultScreenDevice().getDefaultConfiguration()
        except Exception:
            temp = BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB)
            graphics_config = temp.getGraphics().getDeviceConfiguration()
        # Assets.init resolves sprite images relative to user.dir.
        Assets.init(graphics_config)
        world.initializeVisuals(graphics_config)
        world.initializeLevel(level_str, int(self.cfg.seconds) * _MS_PER_SECOND)
        world.mario.isLarge = bool(self.cfg.mario_mode > 0)
        world.mario.isFire = bool(self.cfg.mario_mode > 1)
        world.lives = int(self.lives)
        # Priming NOOP; unlike MarioEnv the timer is NOT re-armed afterwards.
        world.update(self.actions.get_java_action(_NOOP_ACTION))
        self._world = world
        self._fm = self.MarioForwardModel(world)
        self._prev_completion = float(self._fm.getCompletionPercentage())
        self._prev_lives = int(world.lives)
        self._prev_kills = self._prev_coins = 0
        self._prev_mushrooms = self._prev_flowers = self._prev_bricks = 0
        self._steps = 0
        self._step_frames = []
        obs = self._get_observation()
        info = {
            "status": str(world.gameStatus),
            "lives": int(world.lives),
            "level_file": str(self.level_file),
        }
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray | dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Advance up to ``cfg.frame_skip`` ticks, capturing one frame per tick."""
        action = max(0, min(action, self.action_space.n - 1))
        java_action = self.actions.get_java_action(action)
        self._step_frames = []
        for _ in range(int(self.cfg.frame_skip)):
            # Break before advancing once the game has ended so captured
            # frames stay aligned with live gameplay ticks.
            if str(self._world.gameStatus) != STATUS_RUNNING:
                break
            self._fm.advance(java_action)
            if self.render_enabled:
                frame = self.render_frame()
                if frame is not None:
                    self._step_frames.append(frame)
        status = str(self._world.gameStatus)
        reward = 0.0
        self._steps += 1
        completion = float(self._fm.getCompletionPercentage())
        reward += completion - self._prev_completion
        self._prev_completion = completion
        if self._world.lives < self._prev_lives:
            reward += self.cfg.death_penalty
        self._prev_lives = int(self._world.lives)
        current_kills = int(self._fm.getKillsTotal())
        dkills = current_kills - self._prev_kills
        if dkills > 0:
            reward += self.cfg.kill_reward * dkills
        self._prev_kills = current_kills
        current_coins = int(self._fm.getNumCollectedCoins())
        if current_coins > self._prev_coins:
            reward += self.cfg.coin_reward * (current_coins - self._prev_coins)
        self._prev_coins = current_coins
        current_mushrooms = int(self._fm.getNumCollectedMushrooms())
        if current_mushrooms > self._prev_mushrooms:
            reward += self.cfg.mushroom_reward * (current_mushrooms - self._prev_mushrooms)
        self._prev_mushrooms = current_mushrooms
        current_flowers = int(self._fm.getNumCollectedFireflower())
        if current_flowers > self._prev_flowers:
            reward += self.cfg.fireflower_reward * (current_flowers - self._prev_flowers)
        self._prev_flowers = current_flowers
        current_bricks = int(self._fm.getNumDestroyedBricks())
        if current_bricks > self._prev_bricks:
            reward += self.cfg.brick_reward * (current_bricks - self._prev_bricks)
        self._prev_bricks = current_bricks
        if status == STATUS_WIN:
            reward += _WIN_REWARD
        elif status == STATUS_TIME_OUT:
            reward -= _TIMEOUT_PENALTY
        terminated = status == STATUS_WIN or status == STATUS_LOSE
        truncated = status == STATUS_TIME_OUT or self._steps >= int(self.cfg.max_steps)
        obs = self._get_observation()
        info = {
            "action": self.actions.get_action_name(action),
            "status": status,
            "completion": completion,
            "lives": int(self._world.lives),
            "kills": current_kills,
            "coins": current_coins,
            "mushrooms": current_mushrooms,
            "flowers": current_flowers,
            "bricks": current_bricks,
        }
        if terminated or truncated:
            info["level_file"] = str(self.level_file)
        return obs, reward, terminated, truncated, info

    def render_frame(self) -> Optional[np.ndarray]:
        """Return the current RGB frame, or None when rendering is disabled."""
        if not self.render_enabled:
            return None
        jimg = self._fm.toImage()
        return _java_buffered_image_to_numpy(jimg)

    def get_step_frames(self) -> list[np.ndarray]:
        """Return the frames captured during the most recent :meth:`step`."""
        return self._step_frames.copy()

    def _get_state_features(
        self, mario_screen_pos: tuple[int, int], mario_mode: int
    ) -> np.ndarray:
        """Return ``[rel_x, rel_y, tanh(vx/8), tanh(vy/8), on_ground, mode-1]``."""
        height, width = self.cfg.obs_shape
        mario = self._world.mario
        mx = float(np.clip(mario_screen_pos[0], 0, width - 1))
        my = float(np.clip(mario_screen_pos[1], 0, height - 1))
        vx = float(getattr(mario, "xa", 0.0))
        vy = float(getattr(mario, "ya", 0.0))
        on_ground = 1.0 if bool(getattr(mario, "onGround", False)) else 0.0
        rel_x = 0.0 if width <= 1 else 2.0 * (mx / (width - 1)) - 1.0
        rel_y = 0.0 if height <= 1 else 2.0 * (my / (height - 1)) - 1.0
        return np.array(
            [
                rel_x,
                rel_y,
                float(np.tanh(vx / _VELOCITY_TANH_SCALE)),
                float(np.tanh(vy / _VELOCITY_TANH_SCALE)),
                on_ground,
                float(mario_mode - 1),
            ],
            dtype=np.float32,
        )

    def _get_observation(self) -> np.ndarray | dict[str, np.ndarray]:
        mario_raw = np.array(
            self._fm.getMarioCompleteObservation(_COMPLETE_DETAIL, _COMPLETE_DETAIL),
            dtype=np.int16,
        )
        screen_raw = np.array(
            self._fm.getScreenCompleteObservation(_COMPLETE_DETAIL, _COMPLETE_DETAIL),
            dtype=np.int16,
        )
        mario_pos = self._fm.getMarioScreenTilePos()
        mario_screen_pos = (int(mario_pos[0]), int(mario_pos[1]))
        mario = self._world.mario
        mario_mode = 2 if mario.isFire else 1 if mario.isLarge else 0
        grid = np.concatenate(
            [mario_raw.T[..., np.newaxis], screen_raw.T[..., np.newaxis]], axis=-1
        )
        if not self.cfg.include_state_features:
            return grid
        return {"grid": grid, "state": self._get_state_features(mario_screen_pos, mario_mode)}

    def close(self) -> None:
        """No-op; the JVM is process-wide and never shut down."""
