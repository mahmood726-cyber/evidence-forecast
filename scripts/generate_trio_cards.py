"""Phase-1 release driver: emits all three Forecast Cards."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PICOS = ["sglt2i_hfpef", "tirzepatide_hfpef_acm", "empareg_t2dm"]


def main() -> int:
    rc = 0
    for pico in PICOS:
        print(f"--- {pico} ---")
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_forecast.py"), "--pico", pico],
            check=False,
        )
        rc = rc or result.returncode
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
