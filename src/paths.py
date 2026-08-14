"""Package-relative paths for data, simulator assets, and repo layout."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT: Path = Path(__file__).resolve().parents[1]  # main/
REPO_ROOT: Path = PACKAGE_ROOT.parent

# Vendored simulator (main/smb); falls back to the repo-root copy.
_VENDORED_SMB = PACKAGE_ROOT / "smb"
SMB_DIR: Path = _VENDORED_SMB if _VENDORED_SMB.is_dir() else REPO_ROOT / "smb"
DATA_DIR: Path = PACKAGE_ROOT / "data"

# Only this jar exposes MarioForwardModel.toImage(); the first JVM start in a
# process pins the jar (see src.env.jvm.start_jvm).
RENDER_JAR: Path = SMB_DIR / "Mario-AI-Interface_new.jar"
