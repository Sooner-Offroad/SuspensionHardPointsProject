"""
Double Wishbone Kinematic Solver
Baja SAE / Suspension Hardpoint Design Tool

Coordinate system (SAE):
  X = forward (positive toward front of car)
  Y = left (positive toward driver's left)
  Z = up (positive upward)

All units: mm and degrees unless noted.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Hardpoints:
    """
    All 3D hardpoint coordinates for one corner (front left by convention).
    Mirror Y for right side.
    """
    # Upper control arm
    # UBJ slightly inboard of LBJ → negative camber as intended for Baja
    uca_inboard_front: np.ndarray = field(default_factory=lambda: np.array([ 60.0,  155.0, 295.0]))
    uca_inboard_rear:  np.ndarray = field(default_factory=lambda: np.array([-60.0,  155.0, 295.0]))
    uca_outboard:      np.ndarray = field(default_factory=lambda: np.array([  0.0,  297.0, 280.0]))  # UBJ

    # Lower control arm
    lca_inboard_front: np.ndarray = field(default_factory=lambda: np.array([ 80.0,  120.0, 175.0]))
    lca_inboard_rear:  np.ndarray = field(default_factory=lambda: np.array([-80.0,  120.0, 175.0]))
    lca_outboard:      np.ndarray = field(default_factory=lambda: np.array([  0.0,  300.0, 150.0]))  # LBJ

    # Steering (rack behind axle, tie rod roughly parallel to LCA)
    tie_rod_inboard:   np.ndarray = field(default_factory=lambda: np.array([-30.0,  160.0, 195.0]))
    tie_rod_outboard:  np.ndarray = field(default_factory=lambda: np.array([-30.0,  295.0, 195.0]))

    # Shock absorber (upper on chassis, lower near LCA outboard)
    shock_upper:       np.ndarray = field(default_factory=lambda: np.array([  0.0,  195.0, 345.0]))
    shock_lower:       np.ndarray = field(default_factory=lambda: np.array([  0.0,  270.0, 195.0]))

    # Wheel center at static ride height (hub center)
    wheel_center:      np.ndarray = field(default_factory=lambda: np.array([  0.0,  300.0, 215.0]))

    def copy(self) -> "Hardpoints":
        return Hardpoints(**{k: v.copy() for k, v in self.__dict__.items()})

    def to_dict(self) -> dict:
        return {k: v.tolist() for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Hardpoints":
        return cls(**{k: np.array(v) for k, v in d.items()})


@dataclass
class KinematicTargets:
    """Design targets — solver scores hardpoints against these."""
    camber_gain_per_mm:    float = -0.05   # deg/mm (negative = gain toward neg camber in bump)
    roll_center_height:    float = 50.0    # mm above ground
    bump_steer_per_mm:     float = 0.002   # deg/mm max toe change
    motion_ratio:          float = 0.65    # shock travel / wheel travel
    anti_squat_pct:        float = 50.0    # % (rear parameter, placeholder)


@dataclass
class KinematicResult:
    """Results of a full travel sweep for one corner."""
    travel_mm:        np.ndarray = field(default_factory=lambda: np.zeros(1))
    camber_deg:       np.ndarray = field(default_factory=lambda: np.zeros(1))
    toe_deg:          np.ndarray = field(default_factory=lambda: np.zeros(1))
    caster_deg:       np.ndarray = field(default_factory=lambda: np.zeros(1))
    roll_center_z:    np.ndarray = field(default_factory=lambda: np.zeros(1))
    roll_center_y:    np.ndarray = field(default_factory=lambda: np.zeros(1))
    motion_ratio:     np.ndarray = field(default_factory=lambda: np.zeros(1))
    track_change_mm:  np.ndarray = field(default_factory=lambda: np.zeros(1))
    shock_length_mm:  np.ndarray = field(default_factory=lambda: np.zeros(1))

    # Static values
    static_camber:    float = 0.0
    static_toe:       float = 0.0
    static_caster:    float = 0.0
    static_rc_height: float = 0.0
    static_motion_ratio: float = 0.65


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def rotation_matrix_x(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_matrix_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def angle_from_vertical_yz(p1: np.ndarray, p2: np.ndarray) -> float:
    """Camber-style angle: signed angle of (p2-p1) from vertical in Y-Z plane, degrees."""
    d = p2 - p1
    angle = np.degrees(np.arctan2(d[1], d[2]))
    return angle - 90.0  # relative to vertical


def compute_camber(ubj: np.ndarray, lbj: np.ndarray) -> float:
    """
    Camber = lean of wheel from vertical in front (Y-Z) plane.
    Angle of (UBJ - LBJ) vector from the Z axis.
    Negative = top of wheel leans inward (normal Baja setup).
    """
    d = ubj - lbj
    return np.degrees(np.arctan2(d[1], d[2]))


def compute_caster(ubj: np.ndarray, lbj: np.ndarray) -> float:
    """
    Caster = angle of kingpin axis in side (X-Z) plane.
    Positive caster = upper pivot behind lower pivot.
    """
    d = ubj - lbj
    return np.degrees(np.arctan2(-d[0], d[2]))


def compute_toe(wheel_center: np.ndarray, tie_outboard: np.ndarray,
                tie_inboard: np.ndarray) -> float:
    """
    Approximate toe from the steering arm direction.
    Positive = toe-in (front of wheel toward center).
    """
    steer_vec = tie_outboard - tie_inboard
    return np.degrees(np.arctan2(steer_vec[0], steer_vec[1]))


def compute_roll_center(ubj_l: np.ndarray, lbj_l: np.ndarray,
                        ubj_r: np.ndarray, lbj_r: np.ndarray,
                        track_half: float) -> tuple[float, float]:
    """
    Front-view geometry roll center via instant center method.
    Projects UBJ and LBJ into Y-Z plane, finds instant center per side,
    draws line to contact patch, intersects at vehicle centerline.

    Returns (rc_y, rc_z) — lateral position and height.
    """
    # Left side instant center (Y-Z plane)
    ic_l = _instant_center_yz(ubj_l, lbj_l)
    # Right side (mirror Y)
    ubj_r_m = ubj_r.copy(); ubj_r_m[1] *= -1
    lbj_r_m = lbj_r.copy(); lbj_r_m[1] *= -1
    ic_r = _instant_center_yz(ubj_r_m, lbj_r_m)

    # Contact patches at ground (Z=0)
    cp_l = np.array([0.0, track_half, 0.0])
    cp_r = np.array([0.0, -track_half, 0.0])

    # Roll center = intersection of (IC_l -> CP_l) and (IC_r -> CP_r) in Y-Z
    rc = _line_intersect_yz(ic_l[:2], cp_l[1:3], ic_r[:2], cp_r[1:3])
    return float(rc[0]), float(rc[1])


def _instant_center_yz(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    For a control arm defined by two points, find where the perpendicular
    bisector line (in Y-Z plane) intersects X=0. Simplified: we just return
    the midpoint projected — for a proper IC we use the arm axis intersection.
    """
    # IC is at infinity along the arm axis direction projected to Y-Z
    d = p2[[1, 2]] - p1[[1, 2]]
    perp = np.array([-d[1], d[0]])  # perpendicular in Y-Z
    mid = (p1[[1, 2]] + p2[[1, 2]]) / 2.0
    # IC = midpoint + t*perp; project to large distance for near-parallel arms
    t = 500.0 / (np.linalg.norm(perp) + 1e-9)
    return np.array([0.0, mid[0] + perp[0] * t, mid[1] + perp[1] * t])


def _line_intersect_yz(p1: np.ndarray, p2: np.ndarray,
                       p3: np.ndarray, p4: np.ndarray) -> np.ndarray:
    """
    Intersect two 2D lines defined by point pairs (y,z).
    Falls back to midpoint average if parallel.
    """
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-9:
        return (p1 + p3) / 2.0
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    return p1 + t * d1


# ---------------------------------------------------------------------------
# Wheel position solver
# ---------------------------------------------------------------------------

def move_wheel_vertically(hp: Hardpoints, dz: float) -> Hardpoints:
    """
    Compute new hardpoint positions when the wheel moves dz mm vertically.

    Strategy:
      - LCA rotates about its inboard axis (front-rear pivot line)
      - UCA rotates about its inboard axis
      - Outboard points constrained to their arm lengths
      - Tie rod outboard follows the knuckle
      - Shock lower follows LCA outboard
    """
    moved = hp.copy()

    # LCA inboard axis
    lca_axis = unit(hp.lca_inboard_rear - hp.lca_inboard_front)
    lca_pivot = hp.lca_inboard_front
    lca_orig = hp.lca_outboard - lca_pivot

    # UCA inboard axis
    uca_axis = unit(hp.uca_inboard_rear - hp.uca_inboard_front)
    uca_pivot = hp.uca_inboard_front
    uca_orig = hp.uca_outboard - uca_pivot

    # Find rotation angle for LCA such that outboard point rises by dz
    # Use Newton iteration on the Z component of the rotated vector
    lca_angle = _solve_arm_angle(lca_axis, lca_pivot, lca_orig, hp.lca_outboard, dz)
    uca_angle = _solve_arm_angle(uca_axis, uca_pivot, uca_orig, hp.uca_outboard, dz)

    # Apply rotations
    moved.lca_outboard = lca_pivot + _rodrigues(lca_orig, lca_axis, lca_angle)
    moved.uca_outboard = uca_pivot + _rodrigues(uca_orig, uca_axis, uca_angle)

    # Wheel center tracks LCA outboard Z change (simplified)
    moved.wheel_center = hp.wheel_center + np.array([0.0, 0.0, dz])

    # Tie rod outboard: translate with knuckle (simplified — no bump steer yet)
    knuckle_dz = (moved.lca_outboard[2] - hp.lca_outboard[2] +
                  moved.uca_outboard[2] - hp.uca_outboard[2]) / 2.0
    moved.tie_rod_outboard = hp.tie_rod_outboard + np.array([0.0, 0.0, knuckle_dz])

    # Shock lower follows LCA outboard
    moved.shock_lower = hp.shock_lower + (moved.lca_outboard - hp.lca_outboard)

    return moved


def _rodrigues(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate vector v around unit axis by angle_rad (Rodrigues formula)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


def _solve_arm_angle(axis: np.ndarray, pivot: np.ndarray,
                     arm_vec: np.ndarray, outboard: np.ndarray,
                     target_dz: float) -> float:
    """
    Newton's method: find rotation angle about axis such that
    outboard Z increases by target_dz.
    """
    angle = 0.0
    for _ in range(30):
        rotated = _rodrigues(arm_vec, axis, angle)
        current_dz = (pivot + rotated)[2] - outboard[2]
        error = current_dz - target_dz
        if abs(error) < 1e-6:
            break
        # Derivative: d(Z)/d(angle)
        d_rotated = _rodrigues(arm_vec, axis, angle + 1e-5)
        dzdangle = ((pivot + d_rotated)[2] - (pivot + rotated)[2]) / 1e-5
        if abs(dzdangle) < 1e-10:
            break
        angle -= error / dzdangle
    return angle


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------

def run_sweep(hp: Hardpoints,
              bump_mm: float = 50.0,
              droop_mm: float = 50.0,
              steps: int = 41,
              track_half: Optional[float] = None) -> KinematicResult:
    """
    Sweep wheel travel from -droop_mm to +bump_mm and compute
    kinematic outputs at each step.
    """
    travel = np.linspace(-droop_mm, bump_mm, steps)

    if track_half is None:
        track_half = hp.wheel_center[1]

    # Mirror right side
    hp_r = hp.copy()
    hp_r.uca_inboard_front[1] *= -1
    hp_r.uca_inboard_rear[1]  *= -1
    hp_r.uca_outboard[1]      *= -1
    hp_r.lca_inboard_front[1] *= -1
    hp_r.lca_inboard_rear[1]  *= -1
    hp_r.lca_outboard[1]      *= -1

    camber    = np.zeros(steps)
    toe       = np.zeros(steps)
    caster    = np.zeros(steps)
    rc_z      = np.zeros(steps)
    rc_y      = np.zeros(steps)
    mot_ratio = np.zeros(steps)
    track_chg = np.zeros(steps)
    shock_len = np.zeros(steps)

    static_shock_len = np.linalg.norm(hp.shock_upper - hp.shock_lower)

    for i, dz in enumerate(travel):
        m  = move_wheel_vertically(hp,   dz)
        mr = move_wheel_vertically(hp_r, dz)

        camber[i]    = compute_camber(m.uca_outboard, m.lca_outboard)
        toe[i]       = compute_toe(m.wheel_center, m.tie_rod_outboard, m.tie_rod_inboard)
        caster[i]    = compute_caster(m.uca_outboard, m.lca_outboard)

        rc_y_v, rc_z_v = compute_roll_center(
            m.uca_outboard, m.lca_outboard,
            mr.uca_outboard, mr.lca_outboard,
            track_half
        )
        rc_z[i] = rc_z_v
        rc_y[i] = rc_y_v

        sl = np.linalg.norm(m.shock_upper - m.shock_lower)
        shock_len[i] = sl
        if i > 0:
            d_wheel = dz - travel[i - 1]
            d_shock = sl - shock_len[i - 1]
            mot_ratio[i] = abs(d_shock / d_wheel) if abs(d_wheel) > 1e-9 else mot_ratio[i-1]

        track_chg[i] = m.wheel_center[1] - hp.wheel_center[1]

    mot_ratio[0] = mot_ratio[1] if steps > 1 else 0.65

    # Static index
    si = steps // 2

    return KinematicResult(
        travel_mm       = travel,
        camber_deg      = camber,
        toe_deg         = toe,
        caster_deg      = caster,
        roll_center_z   = rc_z,
        roll_center_y   = rc_y,
        motion_ratio    = mot_ratio,
        track_change_mm = track_chg,
        shock_length_mm = shock_len,
        static_camber       = float(camber[si]),
        static_toe          = float(toe[si]),
        static_caster       = float(caster[si]),
        static_rc_height    = float(rc_z[si]),
        static_motion_ratio = float(mot_ratio[si]),
    )


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_hardpoints_csv(hp: Hardpoints, path: str) -> None:
    lines = ["name,X_mm,Y_mm,Z_mm"]
    for name, val in hp.__dict__.items():
        lines.append(f"{name},{val[0]:.3f},{val[1]:.3f},{val[2]:.3f}")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def export_kinematics_csv(result: KinematicResult, path: str) -> None:
    header = "travel_mm,camber_deg,toe_deg,caster_deg,rc_height_mm,rc_lateral_mm,motion_ratio,track_change_mm"
    rows = []
    for i in range(len(result.travel_mm)):
        rows.append(
            f"{result.travel_mm[i]:.2f},{result.camber_deg[i]:.4f},"
            f"{result.toe_deg[i]:.4f},{result.caster_deg[i]:.4f},"
            f"{result.roll_center_z[i]:.3f},{result.roll_center_y[i]:.3f},"
            f"{result.motion_ratio[i]:.4f},{result.track_change_mm[i]:.3f}"
        )
    with open(path, "w") as f:
        f.write(header + "\n" + "\n".join(rows))