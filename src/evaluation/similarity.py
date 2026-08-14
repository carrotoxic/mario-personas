"""Validation-sample loading and the AAR / Action JS computations for the AAR CLI."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Callable

import numpy as np

RolloutKey = tuple[str, int]
ActionsCache = dict[RolloutKey, np.ndarray]


# ---------------------------------------------------------------------------
# Human trajectory samples.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ValidationSample:
    """One human trajectory paired with its level and evaluation seed."""

    level_path: Path  # resolved level file the trajectory was played on
    human_actions: np.ndarray  # (n,) int64 action ids
    human_grid: np.ndarray  # (n, H, W, C) int16 grids (dtype feeds the embedding path)
    human_state: np.ndarray  # (n, state_dim) float32 state vectors
    eval_seed: int  # base_seed + rank in the sorted .npz listing


def _resolve_level_path(level_dir: Path, stem: str) -> Path | None:
    """Maps an ``.npz`` stem (``..._lvlNN``) to a level file, or None."""
    m = re.search(r"_lvl(\d+)$", stem)
    if m is None:
        return None
    level_id = m.group(1)
    for pattern in (f"lvl{level_id}.lvl", f"lvl-{level_id}.lvl", f"Killer-{level_id}.lvl"):
        p = level_dir / pattern
        if p.exists():
            return p
    matches = sorted(level_dir.rglob(f"*{level_id}.lvl"))
    return matches[0] if matches else None


def build_validation_samples(
    data_dir: Path, level_dir: Path, seed: int
) -> list[ValidationSample]:
    """Loads every human ``.npz`` trajectory and resolves its level file.  Sorted order
    is load-bearing: each file's rank defines ``eval_seed = seed + rank`` (skipped files
    still consume their rank) and the ``per_npz`` row order downstream."""
    out: list[ValidationSample] = []
    for rank, fp in enumerate(sorted(data_dir.glob("*.npz"))):
        lvl = _resolve_level_path(level_dir, fp.stem)
        if lvl is None:
            continue
        data = np.load(fp, allow_pickle=True)
        if "grid" not in data.files or "state" not in data.files or "actions" not in data.files:
            continue
        grid = np.asarray(data["grid"], dtype=np.float32)
        state = np.asarray(data["state"], dtype=np.float32)
        acts = np.asarray(data["actions"], dtype=np.int64).reshape(-1)
        n = int(min(grid.shape[0], state.shape[0], acts.shape[0]))
        if n <= 1:
            continue
        out.append(
            ValidationSample(
                level_path=lvl,
                human_actions=acts[:n].astype(np.int64),
                human_grid=grid[:n].astype(np.int16),
                human_state=state[:n].astype(np.float32),
                eval_seed=int(seed) + rank,
            )
        )
    return out


def rollout_cache_key(sample: ValidationSample) -> RolloutKey:
    """Cache key pairing the resolved level path with the sample's eval seed."""
    return (str(sample.level_path.resolve()), int(sample.eval_seed))


# ---------------------------------------------------------------------------
# AAR (action agreement) and Action JS (marginal action-distribution divergence).
# ---------------------------------------------------------------------------


def action_match_rate(pred_actions: np.ndarray, target_actions: np.ndarray) -> float | None:
    """Exact-match rate over the first ``min(len)`` steps; None when empty."""
    pred = np.asarray(pred_actions, dtype=np.int64).reshape(-1)
    target = np.asarray(target_actions, dtype=np.int64).reshape(-1)
    m = int(min(len(pred), len(target)))
    if m <= 0:
        return None
    return float(np.mean(pred[:m] == target[:m]))


def _safe_prob(counts: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(counts, dtype=np.float64).reshape(-1)
    arr = np.maximum(arr, 0.0) + float(eps)
    return arr / max(arr.sum(), float(eps))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats between two (smoothed) count vectors."""
    p = _safe_prob(p)
    q = _safe_prob(q)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)))
    kl_qm = np.sum(q * (np.log(q) - np.log(m)))
    return float(0.5 * (kl_pm + kl_qm))


def _action_hist(actions: np.ndarray, action_dim: int) -> np.ndarray:
    a = np.asarray(actions, dtype=np.int64).reshape(-1)
    if a.size == 0:
        return np.zeros((int(action_dim),), dtype=np.float64)
    return np.bincount(
        np.clip(a, 0, int(action_dim) - 1), minlength=int(action_dim)
    ).astype(np.float64)


def compute_action_js_metrics(
    samples: list[ValidationSample],
    *,
    action_dim: int,
    sample_rollout: Callable[[ValidationSample], np.ndarray],
) -> dict[str, float]:
    """Action JS divergence, policy-rollout vs human marginal action distributions.
    Sequences truncate to their common prefix length per sample; the output is the
    unweighted mean over samples (``{}`` when no sample has overlapping steps)."""
    js_action: list[float] = []
    for sample in samples:
        pred_actions = sample_rollout(sample)
        human_actions = np.asarray(sample.human_actions, dtype=np.int64).reshape(-1)
        m = int(min(len(pred_actions), len(human_actions)))
        if m <= 0:
            continue
        js_action.append(
            _js_divergence(
                _action_hist(pred_actions[:m], action_dim),
                _action_hist(human_actions[:m], action_dim),
            )
        )
    if not js_action:
        return {}
    return {"action_js_divergence": float(np.mean(js_action))}
