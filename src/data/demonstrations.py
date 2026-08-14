"""Expert demonstration loading and level-matched sampling for DRAIL.
Trajectories are ``<uuid>_lvl<id>.npz`` with aligned grid/state/actions arrays;
sorted read order and the private shuffle fix the published expert subsets."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np

ObsArrayDict = dict[str, np.ndarray]

UNKNOWN_LEVEL_ID = -1


class TransitionBatch(NamedTuple):
    obs: ObsArrayDict  # "grid" (int16) and "state" (float32), shared batch dim
    actions: np.ndarray  # int64 action ids, shape (N,)


@dataclasses.dataclass(frozen=True)
class ExpertDemonstrations:
    by_level: dict[int, TransitionBatch]  # level id -> transitions on that level
    flat: TransitionBatch  # shuffled fallback pool for unknown level ids

    @property
    def level_ids(self) -> list[int]:
        return sorted(self.by_level)


def parse_level_id(stem: str) -> int | None:
    """Extract the level id from a stem like ``<uuid>_lvl42`` (None if absent)."""
    match = re.search(r"_lvl(\d+)$", stem)
    if match is None:
        match = re.search(r"lvl[-_]?(\d+)", stem, re.IGNORECASE)
    return int(match.group(1)) if match else None


def get_level_files_for_ids(level_dir: Path, ids: Sequence[int]) -> list[Path]:
    """Resolve at most one existing level file per id, in id order.
    The glob fallback is a substring match (id 1 can match ``lvl11.lvl``);
    ids without any match are silently skipped."""
    out: list[Path] = []
    for level_id in ids:
        found: Path | None = None
        for template in (
            "lvl{id}.lvl",
            "lvl-{id}.lvl",
            "Collector-{id}.lvl",
            "Runner-{id}.lvl",
            "Killer-{id}.lvl",
        ):
            candidate = level_dir / template.format(id=level_id)
            if candidate.exists():
                found = candidate
                break
        if found is None:
            matches = sorted(
                list(level_dir.glob(f"*{level_id}.lvl"))
                + list(level_dir.glob(f"*{level_id}.txt"))
            )
            if matches:
                found = matches[0]
        if found is not None:
            out.append(found)
    return out


def load_expert_by_level(action_state_dir: Path) -> dict[int, TransitionBatch]:
    """Load expert ``.npz`` transitions grouped by level id.
    Dict insertion order is first-seen file order, which flattening depends on;
    bad files are skipped and arrays truncated to their common length."""
    files = sorted(action_state_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files in {action_state_dir}")
    grouped: dict[int, list[TransitionBatch]] = {}
    for path in files:
        level_id = parse_level_id(path.stem)
        if level_id is None:
            continue
        data = np.load(path, allow_pickle=True)
        if not {"grid", "state", "actions"}.issubset(data.files):
            continue
        grid = np.asarray(data["grid"], dtype=np.float32)
        state = np.asarray(data["state"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.int64).reshape(-1)
        length = int(min(grid.shape[0], state.shape[0], actions.shape[0]))
        if length <= 0:
            continue
        grouped.setdefault(level_id, []).append(
            TransitionBatch(
                obs={
                    "grid": grid[:length].astype(np.int16),
                    "state": state[:length].astype(np.float32),
                },
                actions=actions[:length],
            )
        )
    out = {
        level_id: TransitionBatch(
            obs={
                "grid": np.concatenate([b.obs["grid"] for b in batches], axis=0),
                "state": np.concatenate([b.obs["state"] for b in batches], axis=0),
            },
            actions=np.concatenate([b.actions for b in batches], axis=0),
        )
        for level_id, batches in grouped.items()
    }
    if not out:
        raise ValueError("No expert data loaded after level filtering.")
    return out


def flatten_expert_transitions(
    by_level: dict[int, TransitionBatch], traj_frac: float, seed: int
) -> TransitionBatch:
    """Concatenate all levels (insertion order); keep a shuffled ``traj_frac`` subset.
    The shuffle uses a private ``np.random.default_rng(seed)`` independent of
    the global NumPy stream; at least one transition is always kept."""
    grid = np.concatenate([b.obs["grid"] for b in by_level.values()], axis=0)
    state = np.concatenate([b.obs["state"] for b in by_level.values()], axis=0)
    actions = np.concatenate([b.actions for b in by_level.values()], axis=0).astype(np.int64)
    total = actions.shape[0]
    keep = max(1, min(total, int(round(float(traj_frac) * total))))
    indices = np.arange(total)
    np.random.default_rng(int(seed)).shuffle(indices)
    selected = indices[:keep]
    return TransitionBatch(
        obs={"grid": grid[selected], "state": state[selected]},
        actions=actions[selected],
    )


def load_demonstrations(
    action_state_dir: Path, *, traj_frac: float, seed: int
) -> ExpertDemonstrations:
    """Load the discriminator's expert dataset (per-level pools + flat fallback)."""
    by_level = load_expert_by_level(action_state_dir)
    return ExpertDemonstrations(
        by_level=by_level, flat=flatten_expert_transitions(by_level, traj_frac, seed)
    )


def extract_level_ids_from_infos(
    infos: dict[str, Any], prev_level_ids: np.ndarray
) -> np.ndarray:
    """Track per-env level ids from vector-env ``level_file`` infos.
    Envs without an update this step keep their previous id; a missing
    ``_level_file`` mask treats every entry as fresh. Returns a new array."""
    out = np.asarray(prev_level_ids, dtype=np.int64).copy()
    level_values = infos.get("level_file")
    if level_values is None:
        return out
    mask_values = infos.get("_level_file")
    if mask_values is None:
        mask = np.ones((len(out),), dtype=bool)
    else:
        mask = np.asarray(mask_values, dtype=bool)
    for i in range(min(len(out), len(mask))):
        if not mask[i]:
            continue
        try:
            raw = level_values[i]
        except (IndexError, KeyError, TypeError):
            continue
        if raw is None:
            continue
        level_id = parse_level_id(Path(str(raw)).stem)
        if level_id is not None:
            out[i] = int(level_id)
    return out


def sample_matched_by_level(
    demos: ExpertDemonstrations, policy_level_ids: np.ndarray
) -> TransitionBatch:
    """Draw one expert transition per policy transition, matched by level id.
    Draws use the global NumPy stream; unknown ids fall back to the flat pool.
    Expert states are returned raw — normalization is the caller's concern."""
    grids: list[np.ndarray] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for level_id in np.asarray(policy_level_ids, dtype=np.int64).tolist():
        pool = demos.by_level.get(int(level_id), demos.flat)
        index = int(np.random.randint(0, len(pool.actions)))
        grids.append(pool.obs["grid"][index])
        states.append(pool.obs["state"][index])
        actions.append(np.asarray(pool.actions[index], dtype=np.int64))
    return TransitionBatch(
        obs={"grid": np.stack(grids, axis=0), "state": np.stack(states, axis=0)},
        actions=np.asarray(actions, dtype=np.int64),
    )
