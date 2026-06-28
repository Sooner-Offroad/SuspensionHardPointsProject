"""
solver.py — Double Wishbone Kinematic Solver
Sooner Offroad Baja SAE — Teddy 2025-26

Coordinate system: wheel center at origin
  X = forward   (+toward front)
  Y = outboard  (+away from centerline)
  Z = up        (+upward)

Units: inches and degrees throughout.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Vehicle constants (Teddy) ────────────────────────────────────────────────
TIRE_RADIUS_IN = 11.5       # 23in OD
HALF_TRACK_IN  = 26.0       # 52in front track / 2 — confirmed real
WHEELBASE_IN   = 52.0       # 1320.8mm — per explicit team direction, confirmed
                             # after the conflict with the 1250mm config above.
                             # NOTE: two different sources gave 49.21in vs 52.0in
                             # for this number over the course of this project;
                             # if anything Ackermann-related looks off later,
                             # this is worth double-checking against the real car.
CHASSIS_HALF   = 6.125      # 12.25in footbox width / 2 — real, at the chassis
                             # station nearest the front suspension mounts.
                             # NOTE: also not currently used in any calculation
                             # below. If mounts are further back (nearer the
                             # driver bars), real half-width there is 7.2075in
                             # (14.415in / 2) instead.
RIDE_HEIGHT_IN = 13.25      # real (New Car spec sheet), was placeholder 16.0


# ── Hardpoints ───────────────────────────────────────────────────────────────
@dataclass
class Hardpoints:
    """
    10 pickup points, all relative to wheel center origin.
    Inboard chassis points: negative Y, small X spread (front/rear legs).
    Outboard knuckle points: Y near 0, above/below wheel center.
    """
    # REAL MEASURED TEDDY FRONT-LEFT GEOMETRY — transformed from the team's
    # CAD coordinates (Points_Front_Left.csv, CHAS_/UPRI_ block, origin at
    # chassis centerline, X=fore Y=outboard Z=up) into this tool's
    # wheel-center origin. Wheel center placed using TWO real, confirmed
    # numbers: ~2in spindle length (rough but real measurement, not a
    # curve-fit) and 13.25in ride height (team's New Car spec sheet) ->
    # wheel center = [0, 26.17, 13.25] in the team frame. This gives caster
    # 8.0° (target 7.5, close) and scrub 0.193in (target 0.125, in the
    # right range but not exact — the 2in spindle number is a rough
    # recollection, not a precise measurement, so don't expect a perfect
    # match). KPI comes out 10° vs the team's 14° target, and camber comes
    # out -10° vs the team's 0° spec — BOTH of these are NOT fixable by
    # adjusting spindle length or wheel center position (verified directly:
    # changing spindle length leaves camber/KPI/caster completely
    # unchanged, since they only depend on the upright's own ball-joint
    # geometry, not where the wheel center sits). The real cause is a
    # solver simplification: camber/KPI here come from the kingpin-line
    # tilt between the two ball joints, not a true wheel-spindle axis —
    # this is the same gap that made Lotus read 0° camber when its Wheel
    # Spindle Point was degenerate. Fixing this needs a proper rigid-
    # knuckle solver with a real spindle axis, not different input numbers.
    uca_inboard_front : np.ndarray = field(default_factory=lambda: np.array([ 2.000, -19.500, 12.250]))
    uca_inboard_rear  : np.ndarray = field(default_factory=lambda: np.array([-9.000, -18.250,  8.500]))
    uca_outboard      : np.ndarray = field(default_factory=lambda: np.array([-0.562,  -2.617,  2.250]))
    lca_inboard_front : np.ndarray = field(default_factory=lambda: np.array([ 2.750, -19.500,  5.000]))
    lca_inboard_rear  : np.ndarray = field(default_factory=lambda: np.array([-8.250, -18.250,  3.000]))
    lca_outboard      : np.ndarray = field(default_factory=lambda: np.array([ 0.422,  -1.383, -4.750]))
    tie_rod_inboard   : np.ndarray = field(default_factory=lambda: np.array([ 0.300, -19.500,  7.250]))
    tie_rod_outboard  : np.ndarray = field(default_factory=lambda: np.array([ 2.800,  -1.523, -1.500]))
    # PLACEHOLDER — the team CAD's coilover points (NSMA/CHAS_AttPnt) weren't
    # in the front-left points export, and these need to fit the real Fox
    # 885-06-161 shock (14.31-20.95in). Update with real mount locations
    # once the coilover pickups for this shock are set.
    shock_upper       : np.ndarray = field(default_factory=lambda: np.array([ 0.0, -8.0, 12.0]))
    shock_lower       : np.ndarray = field(default_factory=lambda: np.array([ 0.0, -2.0, -5.0]))

    # As-DESIGNED static camber, degrees (negative = top of wheel leans in).
    # This is the camber the hub is machined to at ride height, and it is
    # SEPARATE from the kingpin (steering) axis. A real upright can have a
    # tilted kingpin line (for caster/KPI) while the wheel still sits at
    # whatever camber the spindle was machined to. Teddy's spec sheet says
    # static camber = 0, so that's the default. The solver uses this to
    # define a true wheel-spindle axis distinct from the kingpin line, then
    # carries that axis rigidly through suspension travel — so static camber
    # equals this design value and camber CHANGE through travel comes from
    # real knuckle kinematics, instead of the old approximation that wrongly
    # equated camber with the kingpin-line Y-tilt.
    static_design_camber : float = 0.0

    def copy(self):
        return Hardpoints(**{k: (v.copy() if isinstance(v, np.ndarray) else v)
                             for k, v in self.__dict__.items()})

    def to_dict(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: (np.array(v) if isinstance(v, (list, tuple)) else v)
                      for k, v in d.items()})

    def point_names(self):
        return [k for k, v in self.__dict__.items() if isinstance(v, np.ndarray)]

    def all_points(self):
        return {k: v for k, v in self.__dict__.items() if isinstance(v, np.ndarray)}


POINT_LABELS = {
    "uca_inboard_front" : "UCA Inboard Front",
    "uca_inboard_rear"  : "UCA Inboard Rear",
    "uca_outboard"      : "UCA Outboard (UBJ)",
    "lca_inboard_front" : "LCA Inboard Front",
    "lca_inboard_rear"  : "LCA Inboard Rear",
    "lca_outboard"      : "LCA Outboard (LBJ)",
    "tie_rod_inboard"   : "Tie Rod Inboard",
    "tie_rod_outboard"  : "Tie Rod Outboard",
    "shock_upper"       : "Shock Upper",
    "shock_lower"       : "Shock Lower",
}


# ── Geometry math ────────────────────────────────────────────────────────────
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def _rodrigues(v, axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    return v*c + np.cross(axis, v)*s + axis*np.dot(axis, v)*(1-c)


def _solve_arm_angle(axis, pivot, arm_vec, outboard, target_dz):
    angle = 0.0
    for _ in range(50):
        rot = _rodrigues(arm_vec, axis, angle)
        err = (pivot + rot)[2] - outboard[2] - target_dz
        if abs(err) < 1e-7:
            break
        rot2 = _rodrigues(arm_vec, axis, angle + 1e-5)
        dz_da = ((pivot + rot2)[2] - (pivot + rot)[2]) / 1e-5
        if abs(dz_da) < 1e-10:
            break
        angle -= err / dz_da
    return angle


def _line_intersect_2d(p1, p2, p3, p4):
    d1, d2 = p2-p1, p4-p3
    cross = d1[0]*d2[1] - d1[1]*d2[0]
    # Scale-relative parallel check: an absolute 1e-9 threshold lets near-
    # parallel (but not exactly parallel) lines slip through and divide by
    # a tiny number, blowing the intersection out to absurd distances
    # instead of correctly falling back to "treat as parallel".
    scale = np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-12
    if abs(cross) / scale < 1e-6:
        return (p1+p3)/2
    t = ((p3[0]-p1[0])*d2[1] - (p3[1]-p1[1])*d2[0]) / cross
    return p1 + t*d1


# ── Static geometry calculations ─────────────────────────────────────────────
def calc_camber(ubj, lbj):
    """LEGACY camber approximation: treats the kingpin line tilt AS camber.
    This is geometrically wrong for a real knuckle (the wheel plane and the
    steering axis are different things) and is kept only for backward
    compatibility / comparison. The sweep now uses the spindle-axis model
    below (camber_from_spindle) instead."""
    d = ubj - lbj
    return float(np.degrees(np.arctan2(d[1], d[2])))


def spindle_axis_static(ubj, lbj, design_camber_deg):
    """Build the wheel SPINDLE axis (the axle the wheel spins about) at the
    static position, distinct from the kingpin axis.

    The spindle points outboard (+Y) and its tilt in the front (Y-Z) plane
    is set by the design camber: a wheel with camber θ has its spin axis
    tilted θ from horizontal (top of wheel leaning in for negative camber).
    The wheel plane is perpendicular to this axis, so its camber reads as the
    design value by construction — this is what the old model got wrong.

    Returned in the GLOBAL frame at the static pose; the sweep then carries
    it rigidly with the knuckle so camber CHANGE comes from real kinematics."""
    # Camber θ: spindle axis tilts up-inboard by θ. In Y-Z, spindle direction
    # = (cosθ outboard, sinθ up) with sign so negative camber leans wheel top in.
    th = np.radians(design_camber_deg)
    # spindle points outboard (−Y is outboard in left-side tool convention since
    # outboard ball joints sit at negative Y relative to wheel center… but the
    # wheel center is outboard of them, so outboard = +Y away from chassis).
    # Use +Y as outboard for the spindle direction.
    spindle = np.array([0.0, np.cos(th), np.sin(th)])
    return unit(spindle)


def camber_from_spindle(spindle_axis):
    """Camber = tilt of the wheel plane from vertical = tilt of the spindle
    axis from horizontal, measured in the front (Y-Z) plane. Negative =
    top-in."""
    s = spindle_axis
    return float(np.degrees(np.arctan2(s[2], abs(s[1]))))


def calc_caster(ubj, lbj):
    d = ubj - lbj
    return float(np.degrees(np.arctan2(-d[0], d[2])))

def calc_kpi(ubj, lbj):
    d = ubj - lbj
    return float(abs(np.degrees(np.arctan2(abs(d[1]), d[2]))))

def calc_scrub(ubj, lbj):
    d = ubj - lbj
    if abs(d[2]) < 1e-9: return 0.0
    t = (-TIRE_RADIUS_IN - lbj[2]) / d[2]
    return float(0.0 - (lbj[1] + t*d[1]))

def calc_mechanical_trail(ubj, lbj):
    d = ubj - lbj
    if abs(d[2]) < 1e-9: return 0.0
    t = (-TIRE_RADIUS_IN - lbj[2]) / d[2]
    return float((lbj[0] + t*d[0]))

def calc_toe(tie_out, tie_in):
    d = tie_out - tie_in
    return float(np.degrees(np.arctan2(d[0], d[1])))

def _fvic(ubj, lbj, uca_mid, lca_mid):
    p1 = np.array([uca_mid[1], uca_mid[2]])
    p2 = np.array([ubj[1],     ubj[2]])
    p3 = np.array([lca_mid[1], lca_mid[2]])
    p4 = np.array([lbj[1],     lbj[2]])
    ic = _line_intersect_2d(p1, p2, p3, p4)
    return float(ic[0]), float(ic[1])

def calc_swing_arm(ubj, lbj, uca_mid, lca_mid):
    iy, iz = _fvic(ubj, lbj, uca_mid, lca_mid)
    return float(np.sqrt(iy**2 + iz**2))

def calc_roll_center(ubj_l, lbj_l, um_l, lm_l,
                     ubj_r, lbj_r, um_r, lm_r):
    ic_l = np.array(_fvic(ubj_l, lbj_l, um_l, lm_l))
    ic_r = np.array(_fvic(ubj_r, lbj_r, um_r, lm_r))
    cp_l = np.array([ HALF_TRACK_IN, -TIRE_RADIUS_IN])
    cp_r = np.array([-HALF_TRACK_IN, -TIRE_RADIUS_IN])
    rc = _line_intersect_2d(ic_l, cp_l, ic_r, cp_r)
    return float(rc[0]), float(rc[1])

def ackermann_from_steer(hp: Hardpoints) -> float:
    """Real Ackermann percentage via the standard static convergence-point
    method (this replaces an earlier attempt that simulated a mirrored
    rack input — that approach was proven mathematically incapable of ever
    detecting real Ackermann on mirror-symmetric hardware: applying a
    mirrored rack displacement to mirrored geometry always yields exactly
    equal-and-opposite wheel angles by construction, regardless of the
    real steering-arm geometry, so it could only ever report 0%).

    The standard method: extend a line from the kingpin axis's ground
    contact point through the steering arm pivot (tie_rod_inboard). For
    100% (true) Ackermann, this line — extended inward and rearward —
    should cross the vehicle centerline exactly at the rear axle. If it
    crosses closer to the front axle, that's less than 100% (toward
    parallel steer / 0%); if it crosses behind the rear axle, that's over
    100%. This is a static construction — no turn or rack simulation
    needed, since it depends only on the as-built steering arm geometry."""
    d = hp.uca_outboard - hp.lca_outboard
    if abs(d[2]) < 1e-9:
        return 0.0
    t = (-TIRE_RADIUS_IN - hp.lca_outboard[2]) / d[2]
    kpi_x = hp.lca_outboard[0] + t*d[0]
    kpi_y = hp.lca_outboard[1] + t*d[1]

    tx, ty = hp.tie_rod_inboard[0], hp.tie_rod_inboard[1]
    dx, dy = tx - kpi_x, ty - kpi_y
    centerline_y = -HALF_TRACK_IN   # vehicle centerline, in this wheel's frame
    if abs(dy) < 1e-9:
        return 0.0   # arm points straight fore/aft, line never reaches centerline
    s = (centerline_y - kpi_y) / dy
    x_cross = kpi_x + s*dx   # X position where the line crosses centerline

    # Front axle is at X=0 (wheel center origin); rear axle is WHEELBASE_IN
    # behind that, at X = -WHEELBASE_IN. 100% = crosses exactly at the rear
    # axle. Percentage scales linearly with how far back the crossing is,
    # relative to the front-to-rear-axle distance.
    return float((-x_cross / WHEELBASE_IN) * 100.0)


def calc_ackermann(tie_out, tie_in, ubj, lbj):
    """LEGACY: static-offset approximation. This was found to be a broken
    formula — it compares the kingpin's ground-intersection Y against the
    tie-rod INBOARD point's Y (a chassis-mounted point ~20in away), which
    has no real geometric relationship to Ackermann percentage. On real
    Teddy geometry this produced 618% before being silently clamped to
    150%, which is why every single optimizer run showed exactly 150% —
    not a real result, a clamp ceiling hiding a wrong formula. Kept only
    for reference; ackermann_from_steer() above is the real calculation."""
    d = ubj - lbj
    if abs(d[2]) < 1e-9: return 0.0
    t = (-TIRE_RADIUS_IN - lbj[2]) / d[2]
    kpi_y = lbj[1] + t*d[1]
    offset = kpi_y - tie_in[1]
    ideal = HALF_TRACK_IN * 0.12
    return float(np.clip((offset/(ideal+1e-9))*100, 0, 150))

def calc_motion_ratio(hp: Hardpoints, dz=0.5):
    """Numerical derivative of shock length vs wheel travel."""
    m1 = _move(hp,  dz/2)
    m2 = _move(hp, -dz/2)
    sl1 = np.linalg.norm(m1.shock_upper - m1.shock_lower)
    sl2 = np.linalg.norm(m2.shock_upper - m2.shock_lower)
    return float(abs(sl1 - sl2) / dz)


# ── Wheel travel ─────────────────────────────────────────────────────────────
def _knuckle_frame(ubj, lbj):
    """Local frame fixed to the knuckle: z along the kingpin axis (LBJ->UBJ),
    x as close to vehicle-forward as possible while staying perpendicular
    to z, y completing a right-handed frame. Used to carry rigidly-attached
    points (like the tie rod outboard point) through a knuckle rotation."""
    z = unit(ubj - lbj)
    global_x = np.array([1.0, 0.0, 0.0])
    x = global_x - np.dot(global_x, z) * z
    if np.linalg.norm(x) < 1e-6:
        global_y = np.array([0.0, 1.0, 0.0])
        x = global_y - np.dot(global_y, z) * z
    x = unit(x)
    y = np.cross(z, x)
    return x, y, z


def _carry_with_knuckle(old_ubj, old_lbj, new_ubj, new_lbj, point_old):
    """Move `point_old` as if it were rigidly bolted to the knuckle defined
    by old_ubj/old_lbj, which has now moved to new_ubj/new_lbj."""
    x_old, y_old, z_old = _knuckle_frame(old_ubj, old_lbj)
    x_new, y_new, z_new = _knuckle_frame(new_ubj, new_lbj)
    rel = point_old - old_lbj
    local = np.array([np.dot(rel, x_old), np.dot(rel, y_old), np.dot(rel, z_old)])
    return new_lbj + local[0]*x_new + local[1]*y_new + local[2]*z_new


def _move(hp: Hardpoints, dz: float) -> Hardpoints:
    m = hp.copy()
    lca_ax  = unit(hp.lca_inboard_rear - hp.lca_inboard_front)
    lca_piv = hp.lca_inboard_front
    lca_vec = hp.lca_outboard - lca_piv
    uca_ax  = unit(hp.uca_inboard_rear - hp.uca_inboard_front)
    uca_piv = hp.uca_inboard_front
    uca_vec = hp.uca_outboard - uca_piv

    la = _solve_arm_angle(lca_ax, lca_piv, lca_vec, hp.lca_outboard, dz)
    ua = _solve_arm_angle(uca_ax, uca_piv, uca_vec, hp.uca_outboard, dz)

    m.lca_outboard = lca_piv + _rodrigues(lca_vec, lca_ax, la)
    m.uca_outboard = uca_piv + _rodrigues(uca_vec, uca_ax, ua)

    m.tie_rod_outboard = _carry_with_knuckle(
        hp.uca_outboard, hp.lca_outboard, m.uca_outboard, m.lca_outboard,
        hp.tie_rod_outboard)
    m.shock_lower = hp.shock_lower + (m.lca_outboard - hp.lca_outboard)
    return m


def _steer(hp: Hardpoints, steer_deg: float) -> Hardpoints:
    """Steer the knuckle by rotating the upright about its kingpin axis
    (UBJ->LBJ line). Approximates rack input: a positive steer angle rotates
    the wheel/knuckle (and the tie rod outboard, spindle, etc.) about the
    steering axis. Used to compute true CV joint angle and bump-steer-while-
    cornering — the solver's vertical-only sweep can't see these.

    This is a kinematic approximation (rotates directly about the kingpin
    axis by the commanded angle) rather than solving the full rack-and-tie-
    rod linkage, which is enough to estimate articulation but won't capture
    exact rack-driven toe curves."""
    if abs(steer_deg) < 1e-9:
        return hp.copy()
    m = hp.copy()
    axis = unit(hp.uca_outboard - hp.lca_outboard)   # kingpin axis
    pivot = hp.lca_outboard
    ang = np.radians(steer_deg)
    # Rotate the outboard points that turn with the wheel about the kingpin.
    # UBJ and LBJ are ON the axis, so they don't move; the tie rod outboard
    # and the wheel/spindle do.
    m.tie_rod_outboard = pivot + _rodrigues(hp.tie_rod_outboard - pivot, axis, ang)
    return m


def _mirror_right(hp: Hardpoints) -> Hardpoints:
    hp_r = hp.copy()
    for k in hp_r.__dict__:
        v = getattr(hp_r, k)
        if not isinstance(v, np.ndarray):
            continue
        v2 = v.copy(); v2[1] *= -1
        setattr(hp_r, k, v2)
    return hp_r


# ── Full sweep result ────────────────────────────────────────────────────────
@dataclass
class SweepResult:
    travel        : np.ndarray
    camber        : np.ndarray
    toe           : np.ndarray
    caster        : np.ndarray
    rc_height     : np.ndarray
    rc_lateral    : np.ndarray
    motion_ratio  : np.ndarray
    track_change  : np.ndarray
    shock_length  : np.ndarray
    frames        : list        # list of Hardpoints at each step

    # Static values
    s_camber      : float = 0.0
    s_toe         : float = 0.0
    s_caster      : float = 0.0
    s_kpi         : float = 0.0
    s_scrub       : float = 0.0
    s_mech_trail  : float = 0.0
    s_rc_height   : float = 0.0
    s_motion_ratio: float = 0.0
    s_swing_arm   : float = 0.0
    s_ackermann   : float = 0.0
    s_fvic_y      : float = 0.0
    s_fvic_z      : float = 0.0


def run_sweep(hp: Hardpoints,
              bump_in:  float = 5.0,
              droop_in: float = 5.0,
              steps:    int   = 41) -> SweepResult:

    travel = np.linspace(-droop_in, bump_in, steps)
    hp_r   = _mirror_right(hp)

    camber = np.zeros(steps); toe    = np.zeros(steps)
    caster = np.zeros(steps); rc_h   = np.zeros(steps)
    rc_lat = np.zeros(steps); mr     = np.zeros(steps)
    trkchg = np.zeros(steps); slen   = np.zeros(steps)
    frames = []

    # Build the static wheel spindle axis (distinct from the kingpin line),
    # tilted per the design camber. As the knuckle moves through travel, this
    # axis is carried rigidly with it (rotated by the same knuckle rotation
    # that moves the ball joints), and camber is read from the MOVED axis.
    design_camber = getattr(hp, "static_design_camber", 0.0)
    spindle_static = spindle_axis_static(hp.uca_outboard, hp.lca_outboard,
                                         design_camber)

    for i, dz in enumerate(travel):
        ml = _move(hp,   dz)
        mr_hp = _move(hp_r, dz)
        frames.append(ml)

        # Carry the spindle axis rigidly with the knuckle: express it as a
        # direction in the static knuckle frame, then re-express in the moved
        # knuckle frame. (Vector carry: a point at lbj + spindle, carried,
        # minus the carried lbj.)
        carried_tip = _carry_with_knuckle(
            hp.uca_outboard, hp.lca_outboard, ml.uca_outboard, ml.lca_outboard,
            hp.lca_outboard + spindle_static)
        spindle_now = unit(carried_tip - ml.lca_outboard)
        camber[i] = camber_from_spindle(spindle_now)

        toe[i]    = calc_toe(ml.tie_rod_outboard, ml.tie_rod_inboard)
        caster[i] = calc_caster(ml.uca_outboard, ml.lca_outboard)

        um_l = (ml.uca_inboard_front + ml.uca_inboard_rear) / 2
        lm_l = (ml.lca_inboard_front + ml.lca_inboard_rear) / 2
        um_r = (mr_hp.uca_inboard_front + mr_hp.uca_inboard_rear) / 2
        lm_r = (mr_hp.lca_inboard_front + mr_hp.lca_inboard_rear) / 2

        ry, rz = calc_roll_center(ml.uca_outboard, ml.lca_outboard, um_l, lm_l,
                                   mr_hp.uca_outboard, mr_hp.lca_outboard, um_r, lm_r)
        rc_lat[i] = ry; rc_h[i] = rz

        sl = np.linalg.norm(ml.shock_upper - ml.shock_lower)
        slen[i] = sl
        if i > 0:
            dw = dz - travel[i-1]
            ds = sl - slen[i-1]
            mr[i] = abs(ds/dw) if abs(dw)>1e-9 else mr[i-1]

        trkchg[i] = ml.uca_outboard[1] - hp.uca_outboard[1]

    mr[0] = mr[1] if steps > 1 else 0.65
    si = steps // 2
    um_s = (hp.uca_inboard_front + hp.uca_inboard_rear) / 2
    lm_s = (hp.lca_inboard_front + hp.lca_inboard_rear) / 2
    fy, fz = _fvic(hp.uca_outboard, hp.lca_outboard, um_s, lm_s)

    return SweepResult(
        travel=travel, camber=camber, toe=toe, caster=caster,
        rc_height=rc_h, rc_lateral=rc_lat, motion_ratio=mr,
        track_change=trkchg, shock_length=slen, frames=frames,
        s_camber      = float(camber[si]),
        s_toe         = float(toe[si]),
        s_caster      = float(caster[si]),
        s_kpi         = calc_kpi(hp.uca_outboard, hp.lca_outboard),
        s_scrub       = calc_scrub(hp.uca_outboard, hp.lca_outboard),
        s_mech_trail  = calc_mechanical_trail(hp.uca_outboard, hp.lca_outboard),
        s_rc_height   = float(rc_h[si]),
        s_motion_ratio= float(mr[si]),
        s_swing_arm   = calc_swing_arm(hp.uca_outboard, hp.lca_outboard, um_s, lm_s),
        s_ackermann   = ackermann_from_steer(hp),
        s_fvic_y=fy, s_fvic_z=fz,
    )


# ── Spring / ride frequency ──────────────────────────────────────────────────
def spring_for_frequency(hz, sprung_lb, motion_ratio):
    m = sprung_lb * 0.4536
    k_nm = (hz * 2 * np.pi)**2 * m
    k_lbf = k_nm / 175.127
    return k_lbf / (motion_ratio**2 + 1e-9)

def wheel_rate(spring_lbf_in, motion_ratio):
    return spring_lbf_in * motion_ratio**2

def ride_frequency(wheel_rate_lbf_in, sprung_lb):
    k = wheel_rate_lbf_in * 175.127
    m = sprung_lb * 0.4536
    return (1/(2*np.pi)) * np.sqrt(k/(m+1e-9))


# ── CSV export ───────────────────────────────────────────────────────────────
def export_hardpoints_csv(hp: Hardpoints, path: str):
    lines = ["name,X_in,Y_in,Z_in"]
    for name, val in hp.__dict__.items():
        if not isinstance(val, np.ndarray):
            continue
        lines.append(f"{name},{val[0]:.4f},{val[1]:.4f},{val[2]:.4f}")
    open(path,"w").write("\n".join(lines))

def export_kinematics_csv(r: SweepResult, path: str):
    hdr = "travel_in,camber_deg,toe_deg,caster_deg,rc_height_in,rc_lat_in,motion_ratio,track_chg_in"
    rows = [f"{r.travel[i]:.3f},{r.camber[i]:.4f},{r.toe[i]:.4f},{r.caster[i]:.4f},"
            f"{r.rc_height[i]:.4f},{r.rc_lateral[i]:.4f},{r.motion_ratio[i]:.4f},{r.track_change[i]:.4f}"
            for i in range(len(r.travel))]
    open(path,"w").write(hdr+"\n"+"\n".join(rows))