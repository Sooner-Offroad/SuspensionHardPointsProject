from pathlib import Path
from kinematics.io.geometry_loader import load_geometry
from kinematics.points.derived.definitions import get_wheel_center
from kinematics.core.enums import PointID

p = Path("tests/data/geometry.yaml")
s = load_geometry(p)
positions = s.get_hardpoints_copy()

offset = s.config.wheel.offset if s.config is not None else None

wc = get_wheel_center(positions, offset)
ax_in = positions[PointID.AXLE_INBOARD]
ax_out = positions[PointID.AXLE_OUTBOARD]

print(f"wheel offset: {offset}")
print(f"axle_inboard: {ax_in}")
print(f"axle_outboard: {ax_out}")
print(f"computed wheel_center: {wc}")
