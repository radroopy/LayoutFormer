#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATTERN_ROOT = ROOT / "pattern" / "pattern"
DEFAULT_XLSX = ROOT / "级放数据.xlsx"


def run_step(cmd: list[str], title: str):
    print(f"\n== {title} ==")
    print("cmd:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"step failed: {title}")


def main():
    parser = argparse.ArgumentParser(description="Run all preprocessing steps in order.")
    parser.add_argument(
        "--pattern-root",
        default=str(DEFAULT_PATTERN_ROOT) if DEFAULT_PATTERN_ROOT.is_dir() else str(ROOT),
        help="Root directory for pattern json files (default: <project>/pattern/pattern).",
    )
    parser.add_argument(
        "--xlsx",
        default=str(DEFAULT_XLSX) if DEFAULT_XLSX.is_file() else None,
        help="Path to input xlsx (default: <project>/级放数据.xlsx if exists).",
    )
    parser.add_argument(
        "--sdf-base",
        type=int,
        default=512,
        help="Base size for SDF maps (default: 512).",
    )
    args = parser.parse_args()

    pattern_root = Path(args.pattern_root)
    if not pattern_root.is_dir():
        raise SystemExit(f"pattern root not found: {pattern_root}")

    if args.xlsx is None:
        raise SystemExit("xlsx path is required (use --xlsx).")
    xlsx_path = Path(args.xlsx)
    if not xlsx_path.is_file():
        raise SystemExit(f"xlsx not found: {xlsx_path}")

    py = sys.executable
    here = Path(__file__).resolve().parent

    run_step([py, str(here / "normalize_points.py"), str(pattern_root)], "1) normalize points (-n.json)")
    run_step([py, str(here / "normalize_xlsx.py"), str(xlsx_path), "--root", str(pattern_root)], "2) normalize xlsx (-n.xlsx)")
    run_step([py, str(here / "build_pattern_piece_embeddings.py"), str(pattern_root)], "3) build pattern boundary embeddings")
    run_step([py, str(here / "build_element_embeddings.py"), str(xlsx_path), "--root", str(pattern_root)], "4) build element embeddings")
    run_step([py, str(here / "analyze_shape_element_mapping.py"), str(xlsx_path), "--root", str(pattern_root)], "5) build shape/size mapping")
    run_step([py, str(here / "compute_piece_size_scales.py")], "6) compute size scale factors")
    run_step([py, str(here / "build_pair_splits.py")], "7) build pair splits")
    run_step([py, str(here / "build_sdf_maps.py"), "--base", str(args.sdf_base)], "8) build SDF maps")

    print("\nAll preprocessing steps completed.")


if __name__ == "__main__":
    main()
