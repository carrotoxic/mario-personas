"""Run-artifact I/O: args JSON, state-dict checkpoints, and step schedules.
File formats are pinned by the published runs: ``mario_ppo_args.json`` is an
``indent=2`` JSON dump of the config fields; checkpoints hold raw ``state_dict``s."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, TypeVar

import torch
from torch import nn

ARGS_FILENAME = "mario_ppo_args.json"
FINAL_CHECKPOINT_FILENAME = "mario_ppo.pt"
DISCRIMINATOR_CHECKPOINT_FILENAME = "drail_discriminator.pt"

_ConfigT = TypeVar("_ConfigT")


def save_config_json(run_dir: Path, cfg: Any) -> Path:
    """Write the args snapshot for a config dataclass into ``run_dir``
    (tuples serialize as JSON lists)."""
    path = run_dir / ARGS_FILENAME
    path.write_text(json.dumps(dataclasses.asdict(cfg), indent=2), encoding="utf-8")
    return path


def load_config_json(args_path: Path, config_cls: type[_ConfigT]) -> _ConfigT:
    """Load an args JSON into ``config_cls``: keys matching fields are kept, unknown
    keys are ignored, and missing keys fall back to the field defaults. List values
    are coerced to tuples for tuple-typed fields (the ``obs_shape`` round-trip)."""
    raw = json.loads(args_path.read_text(encoding="utf-8"))
    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(config_cls):
        if field.name not in raw:
            continue
        value = raw[field.name]
        if isinstance(field.default, tuple) and isinstance(value, list):
            value = tuple(value)
        kwargs[field.name] = value
    return config_cls(**kwargs)


def step_checkpoint_path(run_dir: Path, step: int) -> Path:
    """Return the periodic checkpoint path for a scheduled global step."""
    return run_dir / f"mario_ppo_step_{step}.pt"


def final_checkpoint_path(run_dir: Path) -> Path:
    """Return the final-model checkpoint path for a run directory."""
    return run_dir / FINAL_CHECKPOINT_FILENAME


def save_state_dict(module: nn.Module, path: Path) -> None:
    """Save a module's raw ``state_dict`` (no wrapper dict, no metadata)."""
    torch.save(module.state_dict(), path)


def load_checkpoint_state_dict(
    path: Path, device: torch.device | str
) -> dict[str, torch.Tensor]:
    """Load a checkpoint, unwrapping an optional ``{"state_dict": ...}`` payload."""
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # torch < 1.13 has no ``weights_only`` keyword.
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


class StepScheduler:
    """Interval trigger firing at exact multiples of ``interval`` steps; a
    non-positive interval disables it. When one iteration crosses several boundaries,
    one step is returned per boundary (exact interval-multiple checkpoint names)."""

    def __init__(self, interval: int):
        self._interval = int(interval)
        self._next: int | None = self._interval if self._interval > 0 else None

    def due_steps(self, global_step: int) -> list[int]:
        """Pop and return every scheduled step at or below ``global_step``."""
        due: list[int] = []
        while self._next is not None and global_step >= self._next:
            due.append(self._next)
            self._next += self._interval
        return due
