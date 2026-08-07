"""Runtime feature flags for the Shiny app.

No shiny imports — safe to import from anywhere, including tests and the
shinylive (Pyodide) bundle where ``shiny_app`` modules are staged flat at the
bundle root.
"""
from __future__ import annotations

import os
import sys


def json_only_mode() -> bool:
    """True when the app should run as a JSON-only verification surface.

    JSON-only mode hides the Calibrate tab and the image/CSV uploads: the
    Agent JSON supplies the image, data rows, calibration, plot type and
    orientation. It activates automatically under Pyodide (the shinylive
    GitHub Pages deploy) and can be forced either way locally with
    ``PLOTVERIFY_JSON_ONLY=1`` / ``PLOTVERIFY_JSON_ONLY=0``.
    """
    env = os.environ.get("PLOTVERIFY_JSON_ONLY", "").strip().lower()
    if env in ("1", "true", "yes"):
        return True
    if env in ("0", "false", "no"):
        return False
    return sys.platform == "emscripten"
