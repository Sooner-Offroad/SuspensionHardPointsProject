from pathlib import Path
import subprocess
from kinematics.io.geometry_loader import load_geometry
from kinematics.points.derived.definitions import get_wheel_center
from kinematics.core.enums import PointID

# Offsets to test
offsets = [-500, -100, -38, 0, 100, 500]

p = Path("tests/data/geometry.yaml")
s = load_geometry(p)
positions = s.get_hardpoints_copy()

print(f"Using geometry file: {p.resolve()}\n")
for off in offsets:
    wc = get_wheel_center(positions, off)
    print(f"offset {off:6} -> wheel_center: {wc}")

# Optional: try to run visualiser for each offset (may fail if 'uv' is not installed)
run_visualizer = False
if run_visualizer:
    for off in offsets:
        out = Path(f"plot_offset_{off}.png")
        cmd = ["uv", "run", "kinematics", "visualize", "--geometry", str(p), "--output", str(out), "--override-offset", str(off)]
        # Note: the visualize CLI likely doesn't accept --override-offset; this is illustrative.
        try:
            subprocess.run(cmd, check=True)
            print(f"Wrote {out}")
        except Exception as e:
            print(f"Visualizer run failed for offset {off}: {e}")
