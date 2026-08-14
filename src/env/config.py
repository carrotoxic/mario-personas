"""Environment configuration shared by the policy and rendering environments."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class EnvConfig:
    """Configuration for :class:`src.env.environment.MarioEnv`."""

    jar_path: str  # Mario-AI-Framework jar put on the JVM classpath
    level_dir: str  # directory of .lvl/.txt level files to sample from
    user_dir: str = ""  # Java user.dir for asset lookup; first JVM start wins
    seconds: int = 50  # per-segment time budget, refreshed at each checkpoint
    mario_mode: int = 0  # starting form: 0 small, 1 large, 2 fire
    max_steps: int = 3000  # env steps (post frame-skip) before truncation
    frame_skip: int = 4  # simulator ticks per step() call
    obs_shape: tuple[int, int] = (16, 16)  # (height, width) of the tile grid
    max_id: int = 100  # tile-id upper bound; encoders clamp ids to [0, max_id]
    lives: int = 0  # starting world.lives; 0 means the first death loses
    death_penalty: float = -0.1  # added (already negative) once per step lives decreased
    completion_reward: float = 0.0  # coefficient on the per-step completion delta
    win_reward: float = 1.0  # added once when the game status becomes WIN
    kill_reward: float = 0.0  # per kill; if > 0, capped per segment by its enemy count
    coin_reward: float = 0.0  # per collected coin
    mushroom_reward: float = 0.0  # per collected mushroom
    fireflower_reward: float = 0.0  # per collected fire flower
    brick_reward: float = 0.0  # per destroyed brick
    segment_reward: float = 1.0  # per segment checkpoint passed in a step
    include_state_features: bool = False  # Dict obs of grid + small state vector

    def with_overrides(self, **kwargs: object) -> "EnvConfig":
        return dataclasses.replace(self, **kwargs)
