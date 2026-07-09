"""Registry-first project path discovery for representativeness inputs.

This isolates candidate local roots so live code can discover the sister
project without scattering drive-specific literals across the codebase.
"""
# sentinel:skip-file  (P0-hardcoded-local-path: dedicated path-discovery
# helper; candidate drive literals are the canonical pattern Sentinel
# recommends, see lessons.md "Do not hardcode one drive".)
from __future__ import annotations

import os
from pathlib import Path


_ROOT_CANDIDATES: tuple[str, ...] = (
    r"C:\Projects\registry_first_rct_meta",
    r"D:\Projects\registry_first_rct_meta",
)


def discover_root(cli_root: str | None = None) -> Path | None:
    """Return the registry-first repo root, or ``None`` if unavailable."""
    if cli_root:
        return Path(cli_root)
    env = os.environ.get("REGISTRY_FIRST_ROOT")
    if env:
        return Path(env)
    for cand in _ROOT_CANDIDATES:
        path = Path(cand)
        if (path / "outputs" / "population_runs").exists():
            return path
    return None


__all__ = ["discover_root"]
