"""Shared helpers for Migas-1.5 notebooks.

Provides repo-root resolution and data-path validation so notebooks work
regardless of the working directory (repo root, notebooks/, etc.).
"""

import os
import sys


def find_repo_root() -> str:
    """Walk up from the caller's file to find the directory containing pyproject.toml."""
    # Use the caller's __file__ if available, otherwise cwd
    frame = sys._getframe(1)
    caller_file = frame.f_globals.get("__file__")
    if caller_file:
        d = os.path.dirname(os.path.abspath(caller_file))
    else:
        d = os.getcwd()

    for _ in range(10):
        if os.path.isfile(os.path.join(d, "pyproject.toml")):
            return d
        d = os.path.dirname(d)
    raise FileNotFoundError(
        "Could not find repo root (no pyproject.toml found). "
        "Run this script from within the synthefy-migas repository."
    )


def require_data(path: str, download_cmd: str) -> str:
    """Assert that *path* exists (file or directory); raise with download instructions if not.

    Returns the path unchanged for convenient inline use:
        csv_path = require_data("/abs/path/to/file.csv", "uv run ... --csvs")
    """
    if not os.path.exists(path):
        kind = "File" if "." in os.path.basename(path) else "Directory"
        raise FileNotFoundError(
            f"{kind} not found: {path}\n\n"
            f"Download it first:\n  {download_cmd}"
        )
    return path
