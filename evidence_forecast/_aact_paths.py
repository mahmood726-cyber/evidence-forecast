"""AACT snapshot path discovery — single source of truth.

Hardcoded candidate roots are isolated here so the rest of the codebase
doesn't contain literal drive letters; callers go through ``discover_root``
or ``discover_file`` which honour ``AACT_ROOT`` / ``AACT_PATH`` env vars
before falling back to candidates.

Per ``lessons.md`` CT.gov rule: "Do not hardcode one drive. Large local
snapshots such as AACT may live on C: or D:. Use config, candidate-root
discovery, or explicit path inputs and fail closed if no snapshot is found."

This file intentionally contains the drive-specific candidate list; Sentinel's
P0-hardcoded-local-path rule is a false positive here because candidate-root
discovery is precisely what the rule *recommends* — the detector cannot tell
the difference. Keep the candidate list short and update via session memory
(``~/.claude/projects/.../memory/aact_storage_location.md``).
"""
from __future__ import annotations

import os
from pathlib import Path


# Ordered candidates; first match wins. Prefer D (current authoritative
# storage after 2026-04-15 migration); keep C entries for compat with older
# clones and machines where AACT was never moved.
_RAW_ROOT_CANDIDATES: tuple[str, ...] = (
    r"D:\AACT-storage\AACT\2026-04-12",
    r"D:\AACT\2026-04-12",
    r"C:\Users\user\AACT\2026-04-12",
)

_STUDIES_FILE_CANDIDATES: tuple[str, ...] = tuple(
    str(Path(r) / "studies.txt") for r in _RAW_ROOT_CANDIDATES
)


def discover_root(cli_root: str | None = None) -> Path:
    """Return the AACT raw-tables root directory.

    Resolution order:
      1. ``cli_root`` argument (from ``--aact-root``)
      2. ``AACT_ROOT`` environment variable
      3. First candidate in ``_RAW_ROOT_CANDIDATES`` whose ``studies.txt`` exists

    Raises ``SystemExit`` with the searched candidate list if nothing resolves.
    """
    if cli_root:
        return Path(cli_root)
    env = os.environ.get("AACT_ROOT")
    if env:
        return Path(env)
    for cand in _RAW_ROOT_CANDIDATES:
        p = Path(cand)
        if (p / "studies.txt").exists():
            return p
    raise SystemExit(
        "AACT raw root not found. Set --aact-root or AACT_ROOT env var. "
        f"Searched: {_RAW_ROOT_CANDIDATES}"
    )


def discover_file(
    cli_path: str | None = None,
    project_cache: Path | None = None,
) -> Path:
    """Return an AACT path for a consumer: joined CSV, or raw ``studies.txt``.

    Resolution order:
      1. ``cli_path`` argument (from ``--aact``)
      2. ``AACT_PATH`` environment variable
      3. ``project_cache`` (joined CSV in the project's ``cache/`` dir), if set and exists
      4. First ``studies.txt`` in ``_STUDIES_FILE_CANDIDATES`` that exists
      5. Fall back to the project cache path even if missing — callers that
         require a file should check ``.exists()`` on the returned path

    The fallback lets downstream scripts surface a specific "run
    build_aact_cache.py first" error rather than a generic traceback.
    """
    if cli_path:
        return Path(cli_path)
    env = os.environ.get("AACT_PATH")
    if env:
        return Path(env)
    if project_cache is not None and project_cache.exists():
        return project_cache
    for cand in _STUDIES_FILE_CANDIDATES:
        p = Path(cand)
        if p.exists():
            return p
    return project_cache if project_cache is not None else Path(_STUDIES_FILE_CANDIDATES[0])


__all__ = ["discover_root", "discover_file"]
