"""Evaluation tooling for trained Mario persona agents.  Defines the evaluation-JSON
naming contract (``evaluation_results_<levels>_<role>_<tag>_<steps>steps.json``) in
pure stdlib, so consumers can import it without pulling in torch."""

from __future__ import annotations

import re
from pathlib import Path

from src import paths

__all__ = [
    "CHECKPOINT_STEP_RE",
    "RESULTS_STEP_RE",
    "evaluation_results_output_path",
    "infer_results_levels_and_role",
    "parse_result_steps",
    "resolve_repo_path",
    "results_glob_pattern",
    "sanitize_filename_part",
    "training_steps_tag_from_checkpoint",
]

CHECKPOINT_STEP_RE = re.compile(r"^mario_ppo_step_(\d+)\.pt$", re.IGNORECASE)
RESULTS_STEP_RE = re.compile(r"_(\d+)steps\.json$", re.IGNORECASE)


def resolve_repo_path(path_like: str) -> Path:
    """Resolves a path, treating relative paths as repo-root relative."""
    p = Path(path_like).expanduser()
    return p.resolve() if p.is_absolute() else (paths.REPO_ROOT / p).resolve()


def sanitize_filename_part(name: str) -> str:
    """Maps runs of filename-unsafe characters to ``_``; empty results become ``unknown``."""
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(name).strip()).strip("_")
    return s or "unknown"


def training_steps_tag_from_checkpoint(model_path: Path) -> str:
    """Steps tag from a checkpoint name: ``mario_ppo_step_<N>.pt`` -> ``"<N>"``,
    ``mario_ppo.pt`` -> ``"final"``, else the sanitized stem."""
    m = CHECKPOINT_STEP_RE.match(model_path.name)
    if m:
        return m.group(1)
    if model_path.name.lower() == "mario_ppo.pt":
        return "final"
    return sanitize_filename_part(model_path.stem)


def infer_results_levels_and_role(model_path: Path) -> tuple[str | None, str | None]:
    """Extracts (levels_dir, role) from a ``.../results/<levels>/checkpoints/<role>/...`` path."""
    parts = model_path.resolve().parts
    for i, part in enumerate(parts):
        if part.lower() != "results":
            continue
        if i + 4 < len(parts) and parts[i + 2].lower() == "checkpoints":
            return parts[i + 1], parts[i + 3]
        if i + 3 < len(parts) and parts[i + 2].lower() == "checkpoints":
            return parts[i + 1], None
    return None, None


def evaluation_results_output_path(
    model_path: Path, level_dir: Path, results_name_tag: str = "train"
) -> Path:
    """Evaluation JSON path for a checkpoint, written next to it.  Downstream tooling
    globs on this naming pattern — preserve it byte-for-byte."""
    step_tag = training_steps_tag_from_checkpoint(model_path)
    eval_set = sanitize_filename_part(level_dir.name)
    tag = sanitize_filename_part(results_name_tag)
    levels_dir, role = infer_results_levels_and_role(model_path)
    if levels_dir is not None:
        role_part = sanitize_filename_part(role) if role else "model"
        name = (
            f"evaluation_results_{sanitize_filename_part(levels_dir)}_{role_part}"
            f"_{tag}_{step_tag}steps.json"
        )
    else:
        # Outside a results tree, key on the eval level folder and checkpoint
        # stem so runs on different --level-dir do not collide.
        ck = sanitize_filename_part(model_path.stem)
        name = f"evaluation_results_{eval_set}_{ck}_{tag}_{step_tag}steps.json"
    return model_path.parent / name


def results_glob_pattern(levels_name: str, role: str, tag: str) -> str:
    """Glob matching evaluation JSONs for one (levels suite, role, tag)."""
    return f"evaluation_results_{levels_name}_{role}_{tag}_*steps.json"


def parse_result_steps(path: Path) -> int | None:
    """Parses the training timestep from an evaluation JSON file name, or None."""
    m = RESULTS_STEP_RE.search(path.name)
    return int(m.group(1)) if m else None
