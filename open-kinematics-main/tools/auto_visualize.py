#!/usr/bin/env python3
"""
Overwrite geometry.yaml with different wheel offsets, run visualizer, and restore original.
Usage:
    python tools/auto_visualize.py --geometry tests/data/geometry.yaml --offsets -500,-100,-38,0,100,500
"""
from pathlib import Path
import argparse
import yaml
import subprocess
import os
import sys


def run_visualizer(geometry_path: Path, output_path: Path) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    cmd = ["uv", "run", "kinematics", "visualize", "--geometry", str(geometry_path), "--output", str(output_path)]
    print("Running:", " ".join(cmd))
    try:
        p = subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
        print(p.stdout)
        return 0
    except subprocess.CalledProcessError as e:
        print("Visualizer failed:", e.stderr or e.stdout)
        return e.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", default="tests/data/geometry.yaml")
    parser.add_argument("--offsets", default="-500,-100,-38,0,100,500")
    parser.add_argument("--outdir", default="plots_auto")
    args = parser.parse_args()

    geom_path = Path(args.geometry)
    if not geom_path.exists():
        print(f"Geometry file not found: {geom_path}")
        sys.exit(1)

    offsets = [float(x) for x in args.offsets.split(",")]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Backup original
    orig_text = geom_path.read_text(encoding="utf-8")

    try:
        for off in offsets:
            print(f"\n---\nApplying offset {off} and running visualizer")
            data = yaml.safe_load(orig_text)
            if data is None:
                data = {}
            # Ensure nested keys exist
            config = data.setdefault("config", {})
            wheel = config.setdefault("wheel", {})
            wheel["offset"] = float(off)

            # Write modified geometry (overwrite)
            geom_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

            out_file = outdir / f"plot_offset_{int(off)}.png"
            rc = run_visualizer(geom_path, out_file)
            if rc != 0:
                print(f"Visualizer returned non-zero exit {rc} for offset {off}")

    finally:
        # Restore original file
        print("\nRestoring original geometry file")
        geom_path.write_text(orig_text, encoding="utf-8")

    print("Done. Plots saved to:", outdir.resolve())


if __name__ == "__main__":
    main()
