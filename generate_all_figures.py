#!/usr/bin/env python3
"""Regenerate all 12 publication figures across Tracks I, II, and III."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate all 12 publication figures across Tracks I-III.")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "generated_figures",
        help="Output root directory (creates track1/, track2/, track3/ subdirectories).",
    )
    args = ap.parse_args()

    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    for track_name, fn_name in [
        ("track1", "build_track1_figures"),
        ("track2", "build_track2_figures"),
        ("track3", "build_track3_figures"),
    ]:
        tdir = REPO_ROOT / track_name
        sys.path.insert(0, str(tdir))
        try:
            import importlib
            mod = importlib.import_module("build_figures")
            importlib.reload(mod)
            fn = getattr(mod, fn_name)
            generated.extend(fn(out_root / track_name))
        finally:
            sys.path.remove(str(tdir))
            sys.modules.pop("build_figures", None)

    print(f"\nGenerated {len(generated)}/12 publication figures cleanly under {out_root}")


if __name__ == "__main__":
    main()
