"""JVM bootstrap for the Java Mario simulator via JPype."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

import jpype
from jpype import JClass

# A Java class proxy returned by jpype.JClass.
JavaClass = Any


def start_jvm(jar_path: str, user_dir: str | None = None) -> None:
    """Start the process-wide JVM with ``jar_path`` on the classpath.
    One JVM per process, so the first call wins: it pins classpath and
    ``user.dir``; later calls are silent no-ops (the first env decides rendering)."""
    if jpype.isJVMStarted():
        return
    jar = Path(jar_path).resolve()
    if not jar.exists():
        raise FileNotFoundError(f"Jar not found: {jar}")
    xmx = os.environ.get("MARIO_JVM_XMX", "512m")
    args = [f"-Djava.class.path={jar}", f"-Xmx{xmx}"]
    # macOS: AWT must not be headless or BufferedImage rendering breaks.
    if platform.system() == "Darwin":
        args.append("-Djava.awt.headless=false")
    jpype.startJVM(jpype.getDefaultJVMPath(), *args)
    if user_dir:
        JClass("java.lang.System").setProperty("user.dir", str(Path(user_dir).resolve()))


def get_java_classes() -> tuple[JavaClass, JavaClass, JavaClass]:
    """Return the ``(MarioWorld, MarioForwardModel, MarioActions)`` class proxies."""
    return (
        JClass("engine.core.MarioWorld"),
        JClass("engine.core.MarioForwardModel"),
        JClass("engine.helper.MarioActions"),
    )
