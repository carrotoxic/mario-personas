"""Delta-based per-step reward computation for the policy environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.env.config import EnvConfig

STATUS_WIN = "WIN"
STATUS_LOSE = "LOSE"
STATUS_TIME_OUT = "TIME_OUT"
STATUS_RUNNING = "RUNNING"


@dataclass(frozen=True)
class RewardState:
    """Snapshot of the previous step's counters :func:`compute_reward` diffs against.
    Field order is load-bearing: the environment constructs the initial state
    positionally as ``RewardState(0.0, 0, 0, 0, 0, 0, 0)``."""

    prev_completion: float
    prev_lives: int
    prev_kills: int
    prev_coins: int
    prev_mushrooms: int
    prev_flowers: int
    prev_bricks: int


def _delta(current: int | float, previous: int | float) -> float:
    return float(current) - float(previous)


def compute_reward(
    cfg: EnvConfig,
    world: Any,
    forward_model: Any,
    prev: RewardState,
    *,
    steps: int,
    max_steps: int,
    checkpoints_passed_this_step: int = 0,
    kill_reward_credits_remaining: int | None = None,
) -> tuple[float, dict[str, Any], RewardState]:
    """Return step reward ``(reward, details, next_state)`` from deltas vs ``prev``.
    The completion term is unconditional — negative deltas penalize
    backtracking; every other counter term fires only on positive deltas."""
    status = str(world.gameStatus)
    terminated = status == STATUS_WIN or status == STATUS_LOSE
    truncated = status == STATUS_TIME_OUT or steps >= max_steps

    completion = float(forward_model.getCompletionPercentage())
    lives = int(world.lives)
    kills = int(forward_model.getKillsTotal())
    coins = int(forward_model.getNumCollectedCoins())
    mushrooms = int(forward_model.getNumCollectedMushrooms())
    flowers = int(forward_model.getNumCollectedFireflower())
    bricks = int(forward_model.getNumDestroyedBricks())

    dcompletion = _delta(completion, prev.prev_completion)
    dlives = _delta(lives, prev.prev_lives)
    dkills = _delta(kills, prev.prev_kills)
    dcoins = _delta(coins, prev.prev_coins)
    dmushrooms = _delta(mushrooms, prev.prev_mushrooms)
    dflowers = _delta(flowers, prev.prev_flowers)
    dbricks = _delta(bricks, prev.prev_bricks)

    reward = 0.0
    reward += cfg.completion_reward * dcompletion
    if dlives < 0:
        reward += cfg.death_penalty

    dkills_rewarded = 0.0
    if dkills > 0 and cfg.kill_reward != 0.0:
        if kill_reward_credits_remaining is not None:
            cap = max(0, int(kill_reward_credits_remaining))
            dkills_rewarded = float(min(int(dkills), cap))
        else:
            dkills_rewarded = float(dkills)
        reward += cfg.kill_reward * dkills_rewarded
    if dcoins > 0:
        reward += cfg.coin_reward * dcoins
    if dmushrooms > 0:
        reward += cfg.mushroom_reward * dmushrooms
    if dflowers > 0:
        reward += cfg.fireflower_reward * dflowers
    if dbricks > 0:
        reward += cfg.brick_reward * dbricks
    if checkpoints_passed_this_step > 0:
        reward += cfg.segment_reward * float(checkpoints_passed_this_step)
    if status == STATUS_WIN:
        reward += cfg.win_reward

    next_state = RewardState(
        prev_completion=completion,
        prev_lives=lives,
        prev_kills=kills,
        prev_coins=coins,
        prev_mushrooms=mushrooms,
        prev_flowers=flowers,
        prev_bricks=bricks,
    )
    details: dict[str, Any] = {
        "reward_total": reward,
        "completion": completion,
        "lives": lives,
        "kills": kills,
        "coins": coins,
        "mushrooms": mushrooms,
        "flowers": flowers,
        "bricks": bricks,
        "dcompletion": dcompletion,
        "dlives": dlives,
        "dkills": dkills,
        "dkills_rewarded": dkills_rewarded,
        "kill_reward_credits_remaining_before_step": kill_reward_credits_remaining,
        "dcoins": dcoins,
        "dmushrooms": dmushrooms,
        "dflowers": dflowers,
        "dbricks": dbricks,
        "checkpoints_passed_this_step": checkpoints_passed_this_step,
        "segment_reward_step": cfg.segment_reward * float(checkpoints_passed_this_step),
        "status": status,
        "terminated": terminated,
        "truncated": truncated,
        "steps": steps,
    }
    return reward, details, next_state
