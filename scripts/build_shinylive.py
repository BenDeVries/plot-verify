#!/usr/bin/env python3
"""Build the static shinylive bundle for PlotVerify (manual-only).

Stages the canonical sources (shiny_app/app.py, shiny_app/figures.py,
axis_pipeline/, plotverify_core/) plus shinylive_app/requirements.txt into
dist/shinylive/staging/, then runs `shinylive export` to produce a static
deployable site at dist/shinylive/site/.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "dist" / "shinylive" / "staging"
SITE = ROOT / "dist" / "shinylive" / "site"

FILES = [
    (ROOT / "shiny_app" / "app.py", "app.py"),
    (ROOT / "shiny_app" / "figures.py", "figures.py"),
    (ROOT / "shiny_app" / "user_manual.py", "user_manual.py"),
    (ROOT / "shinylive_app" / "requirements.txt", "requirements.txt"),
]
PACKAGES = [
    ROOT / "axis_pipeline",
    ROOT / "plotverify_core",
]


def _check_shinylive() -> None:
    if shutil.which("shinylive") is None:
        sys.exit(
            "shinylive CLI not found. Install it with:\n"
            "    pip install shinylive"
        )


def _stage() -> None:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    for src, name in FILES:
        if not src.exists():
            sys.exit(f"missing source file: {src}")
        shutil.copy2(src, STAGING / name)

    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for pkg in PACKAGES:
        if not pkg.is_dir():
            sys.exit(f"missing source package: {pkg}")
        shutil.copytree(pkg, STAGING / pkg.name, ignore=ignore)


def _export() -> None:
    SITE.parent.mkdir(parents=True, exist_ok=True)
    if SITE.exists():
        shutil.rmtree(SITE)
    subprocess.run(
        ["shinylive", "export", str(STAGING), str(SITE)],
        check=True,
    )


def main() -> None:
    _check_shinylive()
    _stage()
    _export()
    print(f"Built shinylive bundle at: {SITE}")
    print(f"Serve locally with:  python -m http.server -d {SITE} 8000")


if __name__ == "__main__":
    main()
