"""
optimizer.py — Suspension Hardpoint Optimizer
Sooner Offroad Baja SAE — Teddy 2025-26

Two-layer system:
  Knowns  → hard constraints  (violations = rejected via constraint-domination)
  Goals   → soft targets      (multi-objective Pareto search via NSGA-II)

v2: Replaces the single-objective scipy differential_evolution search with a
self-contained NSGA-II multi-objective genetic algorithm (no extra deps beyond
numpy — pymoo was considered but isn't worth the dependency weight for this
problem size). Produces a Pareto front of candidate designs rather than one
"optimal" point baked from arbitrary weights, then picks a recommended
compromise from that front using your Goals weights — matching the
"optimizer -> are the points good? -> add generations" loop in the project
flowchart.

Backward compatible: run_optimizer() keeps the same call signature and
returns an OptResult with the same fields the GUI (app.py) already reads
(final_score, n_evals, converged, violations, score_breakdown, hardpoints,
sweep). It now ALSO attaches `.pareto_front`, a list of OptResult-like
candidates spanning the trade-off space, for future GUI work (e.g. letting
you click through alternative designs instead of only seeing one).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
from solver import (
    Hardpoints, SweepResult, run_sweep,
    calc_camber, calc_caster, calc_kpi, calc_scrub,
    calc_mechanical_trail, calc_ackermann, unit, _steer,
    TIRE_RADIUS_IN, HALF_TRACK_IN,
)


POINT_NAMES = [
    "uca_inboard_front","uca_inboard_rear","uca_outboard",
    "lca_inboard_front","lca_inboard_rear","lca_outboard",
    "tie_rod_inboard","tie_rod_outboard",
    "shock_upper","shock_lower",
]


# ── Knowns (hard constraints) ─────────────────────────────────────────────────
@dataclass
class Knowns:
    """
    Physical facts about Teddy that the optimizer must respect.
    All units: inches.
    """
    chassis_half_width    : float = 6.0
    wheelbase             : float = 64.0
    ride_height           : float = 16.0

    uca_inboard_z_min     : float = 0.5
    uca_inboard_z_max     : float = 7.0
    lca_inboard_z_min     : float = -7.0
    lca_inboard_z_max     : float = 1.0

    uca_inboard_y_min     : float = -12.0
    uca_inboard_y_max     : float = -3.0
    lca_inboard_y_min     : float = -12.0
    lca_inboard_y_max     : float = -3.0

    ubj_z_min             : float = 0.5
    ubj_z_max             : float = 6.0
    lbj_z_min             : float = -6.0
    lbj_z_max             : float = -0.5
    bj_y_min              : float = -0.75     # knuckle thickness — reverted
    bj_y_max              : float = 0.75      # after testing showed loosening this to help KPI
                                                # made camber gain worse (same Y-freedom that helps
                                                # KPI also lets camber swing more during travel)

    shock_stroke_min      : float = 14.31     # Fox 885-06-161 compressed length, real spec
    shock_stroke_max      : float = 20.95     # Fox 885-06-161 extended length, real spec
                                                # (was 3.0/14.0 placeholder)

    bump_in               : float = 5.0
    droop_in              : float = 5.0

    min_jounce_clearance  : float = 0.5

    never_negative_caster : bool  = True
    never_negative_trail  : bool  = True

    # Physical sanity bounds — catch geometrically "feasible" knuckle
    # configurations that are nonetheless not realistic (e.g. UBJ/LBJ
    # spread sideways enough to imply a tilted knuckle). The solver doesn't
    # model the knuckle as a true rigid body, so without these bounds the
    # optimizer can occasionally find degenerate corners (huge caster/KPI/
    # camber) that score well on one goal by accident while every other
    # goal is wrecked.
    caster_max_deg        : float = 12.0
    kpi_max_deg            : float = 18.0
    camber_max_deg         : float = 8.0
    rc_height_max_in       : float = 15.0     # roll center / FVIC math can go
    swing_arm_max_in       : float = 400.0    # unstable near-parallel arms

    # Minimum front-view inclination for each control arm (angle, in the
    # Y-Z plane, between the inboard-midpoint-to-outboard line and
    # horizontal). Real control arms are essentially never designed
    # perfectly flat in front view — that's a degenerate case for the
    # front-view-instant-center roll center calculation (a near-horizontal
    # arm sends the instant center out to an extreme, unstable distance).
    # This keeps the search out of that region rather than trying to make
    # the roll-center math itself immune to it.
    min_arm_angle_deg      : float = 6.0

    # Minimum angular SEPARATION between the UCA and LCA front-view lines
    # (not just each one vs. horizontal). Two arms can each be reasonably
    # inclined and still be nearly PARALLEL TO EACH OTHER — that's the
    # actual condition that sends the front-view instant center (and
    # therefore roll center / swing arm) to an unstable extreme. The
    # min_arm_angle_deg check above misses this case entirely — confirmed
    # directly: a real run produced UCA at -9.98°, LCA at -11.26° (both
    # comfortably past min_arm_angle_deg) but only 1.28° apart from each
    # other, and that candidate had a 216in swing arm.
    min_arm_separation_deg : float = 4.0

    # ── 4WD front halfshaft / CV joint (next car) ──────────────────────
    # Outboard CV joint: fixed-angle, max articulation per spec.
    # Inboard joint: plunging (telescoping), at the diff output.
    cv_max_angle_deg       : float = 42.0     # real spec, confirmed
    max_steer_deg          : float = 27.0     # real (Teddy spec sheet: steering angle 27°)

    # Diff output location (X,Y,Z, our convention: origin=wheel center,
    # X=forward, Y=outboard, Z=up), inches. None = not yet decided — still
    # an open architecture question (possible integration into the
    # steering housing), and a wrong guess here wouldn't just be "off by a
    # bit" the way other placeholders have been; an integrated-housing
    # diff vs. a traditional centrally-mounted one are fundamentally
    # different halfshaft geometries. The CV angle check below is SKIPPED
    # entirely while this is None, rather than running against a
    # placeholder that could look meaningful but isn't. Set this once the
    # diff placement decision is made.
    # THEORETICAL diff output position — built from real diff housing
    # dimensions (9.71in long, 4.98in tall, per team sketch), but the
    # PLACEMENT itself rests on assumptions the sketch doesn't specify:
    # fore/aft position (assumed aligned with front axle, X=0), ground
    # clearance (assumed 4in under the housing), and that the output stub
    # sits at half the housing's long dimension from centerline. This is
    # for EARLY sanity-checking only — it will NOT match the real
    # SolidWorks-derived position that comes after hardpoints are
    # finalized. Treat any CV violation caught here as "worth a second
    # look," not as definitive — and treat a clean pass here as "no gross
    # mismatch found yet," not "confirmed fits."
    diff_output_position   : Optional[np.ndarray] = field(
        default_factory=lambda: np.array([0.0, -21.14, -5.01]))

    # Spindle length (upright ball joints to wheel center), inches. The
    # team's own measurement was rough ("roughly 2in, didn't really care
    # when designing"), so this is a SEARCH RANGE, not a fixed value — the
    # optimizer picks the best spindle length within this range and reports
    # it as an output (see implied_spindle_length()), rather than the tool
    # assuming a single guessed number. Centered on the real ~2in
    # recollection with real margin on both sides for genuine uncertainty.
    spindle_length_min      : float = 1.0
    spindle_length_max      : float = 3.5

    # ── Curve smoothness / stability across the travel sweep ───────────
    # The optimizer evaluates goals at discrete sample points, so a
    # geometry can score well at those points while having violently
    # unstable behavior BETWEEN them — the front-view-instant-center
    # instability shows up as huge spikes in roll center / bump steer /
    # camber over a tiny slice of travel. A real car moves through the
    # whole range constantly, so a spiky curve is not a usable suspension
    # even if its static value looks perfect. These limits reject any
    # geometry whose curves jump more than a sane amount between adjacent
    # travel steps. Calibrated from real data: a good baseline geometry
    # has roll-center jumps ~7.6 in/in and bump-steer jumps ~0.10 deg/in;
    # a degenerate spiky one (which previously passed) had ~518 in/in and
    # ~706 deg/in. Thresholds sit well above "good" but far below
    # "degenerate" so they cleanly separate the two.
    max_rc_jump_per_in      : float = 50.0    # roll center, in per in of travel
    max_bumpsteer_jump      : float = 5.0     # toe, deg per in of travel
    max_camber_jump         : float = 20.0    # camber, deg per in of travel
    max_motion_ratio_jump   : float = 2.0     # motion ratio change per in


# ── Goals (soft targets) ──────────────────────────────────────────────────────
@dataclass
class Goals:
    """
    Kinematic targets from Teddy geometry doc.
    Each has a target value and a weight (0-100).
    Higher weight = optimizer prioritizes it more (and is more likely to be
    picked as one of the explicit Pareto axes — see _primary_and_secondary).
    """
    bump_steer_target     : float = 0.020
    bump_steer_weight     : float = 90.0      # Will: ~90% priority among the 4 primary goals

    roll_center_target    : float = 3.7       # real (New Car spec)
    roll_center_weight    : float = 2.5       # Will: even split with camber gain, ~2.5% each

    camber_gain_target    : float = -0.30
    camber_gain_weight    : float = 2.5       # Will: even split with roll center, ~2.5% each

    kpi_target            : float = 14.0      # real (New Car spec)
    kpi_weight            : float = 5.0       # Will: ~5% priority among the 4 primary goals

    scrub_target          : float = 0.125     # real (New Car spec)
    scrub_weight          : float = 10.0

    motion_ratio_target   : float = 0.60      # was 0.65 — lowered so the real Fox
                                                # shock (6.64in stroke) can deliver the
                                                # full 10in wheel travel within its stroke
    motion_ratio_weight   : float = 10.0

    ackermann_target      : float = -17.5     # ANTI-Ackermann per team decision (mild,
                                                # -15% to -20% range) — NOT the spec
                                                # sheet's +89%, which was true Ackermann.
                                                # Negative = outer wheel turns more than
                                                # inner (per ackermann_from_steer's sign
                                                # convention). Verify this sign matches
                                                # your team's intent before trusting it.
    ackermann_weight      : float = 8.0

    swing_arm_target      : float = 75.0
    swing_arm_weight      : float = 5.0

    caster_target         : float = 7.5       # real (New Car spec)
    caster_weight         : float = 5.0

    mech_trail_target     : float = 0.8
    mech_trail_weight     : float = 5.0

    static_camber_target  : float = 0.0       # real (New Car spec: static camber 0°)
    static_camber_weight  : float = 5.0


# Goal terms always evaluated in this order. The N highest-weighted goals
# become true Pareto objectives in NSGA-II; the rest are folded into one
# weighted "secondary" objective. This keeps the search in the 4-6 objective
# range where NSGA-II actually works well, instead of Pareto-optimizing all
# 11 goals at once (which makes nearly every candidate "non-dominated").
GOAL_TERMS = [
    "bump_steer", "roll_center", "camber_gain", "kpi", "scrub",
    "motion_ratio", "ackermann", "swing_arm", "caster", "mech_trail",
    "static_camber",
]
N_PRIMARY_OBJECTIVES = 4


# ── Point bounds ──────────────────────────────────────────────────────────────
@dataclass
class PointBounds:
    x_min: float = -5.0; x_max: float = 5.0; x_locked: bool = False
    y_min: float = -10.0; y_max: float = 1.0; y_locked: bool = False
    z_min: float = -5.0; z_max: float = 8.0; z_locked: bool = False


def make_bounds_from_knowns(hp: Hardpoints, knowns: Knowns) -> dict:
    """
    Auto-generate point bounds RELATIVE TO EACH POINT'S ACTUAL POSITION.

    Built around wherever each hardpoint currently sits, with generous
    freedom in every direction. This matters because the real Teddy
    geometry has the UCA inboard pivots mounted high (Z~8-12in) and far
    inboard (Y~-19in) — fixed absolute bounds sized for the old placeholder
    geometry put those real points OUTSIDE the allowed range, which clipped
    and distorted the seed. Relative bounds track the real scale
    automatically.

    Critically, the inboard-pivot Z freedom is wide (the arms can go
    anywhere — they are NOT pinned to the current near-parallel layout, per
    the team). Inboard-pivot Z is what sets each arm's front-view angle, so
    giving it room is exactly what lets the optimizer fix the near-parallel
    instability by spreading the UCA and LCA angles apart.

    NOTE: these ranges are NOT verified against the SAE Baja rule book
    (ground clearance, frame envelope, etc.) — sanity-check the optimizer's
    final mounting locations against your frame CAD and current rules before
    committing.
    """
    b = {}

    # Inboard pivots: wide freedom. X (fore/aft along the chassis rail),
    # Y (how far inboard the mount sits), Z (mount height — sets arm angle).
    DX_IN = 5.0   # fore/aft mount freedom
    DY_IN = 5.0   # inboard/outboard mount freedom
    DZ_IN = 6.0   # height freedom — generous, this is what frees the arm angle

    for name in ["uca_inboard_front", "uca_inboard_rear",
                 "lca_inboard_front", "lca_inboard_rear",
                 "tie_rod_inboard"]:
        v = getattr(hp, name)
        b[name] = PointBounds(
            x_min=v[0]-DX_IN, x_max=v[0]+DX_IN,
            y_min=v[1]-DY_IN, y_max=v[1]+DY_IN,
            z_min=v[2]-DZ_IN, z_max=v[2]+DZ_IN)

    # Outboard ball joints sit on the upright — much less free. They must
    # stay near the wheel (near the knuckle) and keep the knuckle physically
    # sane. Bounds are relative to each real point with TIGHT freedom: the
    # knuckle is a rigid machined part, so the joints can shift a little
    # (design refinement) but not roam. Y freedom here is now intentionally
    # tiny (manufacturing tolerance only) — SPINDLE LENGTH is handled as one
    # shared search variable in _pack/_unpack instead, which shifts all
    # three outboard points together and preserves the knuckle's rigid
    # internal shape, rather than letting each point's Y drift independently
    # (which would distort the knuckle rather than represent a real
    # spindle-length design choice).
    DX_OUT = 2.5
    DY_OUT = 0.15  # manufacturing tolerance only — spindle length is separate
    DZ_OUT = 3.0
    for name in ["uca_outboard", "lca_outboard", "tie_rod_outboard"]:
        v = getattr(hp, name)
        b[name] = PointBounds(
            x_min=v[0]-DX_OUT, x_max=v[0]+DX_OUT,
            y_min=v[1]-DY_OUT, y_max=v[1]+DY_OUT,
            z_min=v[2]-DZ_OUT, z_max=v[2]+DZ_OUT)

    # Shock mounts: relative, with enough Z range to reach the real Fox
    # shock's 14.31-20.95in length window through travel.
    v = hp.shock_upper
    b["shock_upper"] = PointBounds(
        x_min=v[0]-3.0, x_max=v[0]+3.0,
        y_min=v[1]-3.0, y_max=v[1]+3.0,
        z_min=v[2]-3.0, z_max=v[2]+3.0)

    v = hp.shock_lower
    b["shock_lower"] = PointBounds(
        x_min=v[0]-3.0, x_max=v[0]+3.0,
        y_min=v[1]-3.0, y_max=v[1]+3.0,
        z_min=v[2]-3.0, z_max=v[2]+3.0)

    return b


# ── Pack / unpack parameter vector ───────────────────────────────────────────
SPINDLE_OUTBOARD_NAMES = ["uca_outboard", "lca_outboard", "tie_rod_outboard"]


def _pack(hp: Hardpoints, bounds: dict, spindle_bounds=None):
    """Pack hardpoints into a flat optimization vector. If spindle_bounds is
    given (lo, hi) in inches, spindle length is packed as ONE shared
    variable that shifts all three outboard points together — preserving
    the knuckle's rigid internal geometry — instead of letting each
    outboard point's Y drift independently (which would distort the
    knuckle rather than represent a real spindle-length design choice).
    The current spindle length (= -mean Y of the three outboard points,
    since wheel center is Y=0 by definition in this tool's frame) becomes
    the starting value; it's then a real, reportable OUTPUT of the search,
    not a fixed input."""
    x0, sp_bounds, idx_map = [], [], []
    skip_y = set()
    if spindle_bounds is not None:
        cur_spindle = -np.mean([getattr(hp, n)[1] for n in SPINDLE_OUTBOARD_NAMES])
        x0.append(np.clip(cur_spindle, spindle_bounds[0], spindle_bounds[1]))
        sp_bounds.append(spindle_bounds)
        idx_map.append(("__spindle__", 1))
        skip_y = set(SPINDLE_OUTBOARD_NAMES)

    for name in POINT_NAMES:
        val = getattr(hp, name)
        b = bounds[name]
        for ai, (locked, lo, hi, v) in enumerate([
            (b.x_locked, b.x_min, b.x_max, val[0]),
            (b.y_locked, b.y_min, b.y_max, val[1]),
            (b.z_locked, b.z_min, b.z_max, val[2]),
        ]):
            if ai == 1 and name in skip_y:
                continue  # handled by the shared spindle variable instead
            if not locked:
                x0.append(np.clip(v, lo, hi))
                sp_bounds.append((lo, hi))
                idx_map.append((name, ai))
    return x0, sp_bounds, idx_map


def _unpack(x, base_hp: Hardpoints, idx_map) -> Hardpoints:
    hp = base_hp.copy()
    spindle_val = None
    for i, (name, axis) in enumerate(idx_map):
        if name == "__spindle__":
            spindle_val = x[i]
            continue
        arr = getattr(hp, name).copy()
        arr[axis] = x[i]
        setattr(hp, name, arr)
    if spindle_val is not None:
        for n in SPINDLE_OUTBOARD_NAMES:
            arr = getattr(hp, n).copy()
            arr[1] = -spindle_val
            setattr(hp, n, arr)
    return hp


def implied_spindle_length(hp: Hardpoints) -> float:
    """Read the spindle length implied by a Hardpoints set — the distance
    from the upright's ball joints out to wheel center (Y=0 in this tool's
    frame). Real, reportable output, not an assumed input."""
    return float(-np.mean([getattr(hp, n)[1] for n in SPINDLE_OUTBOARD_NAMES]))


# ── Constraint checker ────────────────────────────────────────────────────────
def check_constraints(hp: Hardpoints, knowns: Knowns, result: SweepResult) -> list:
    """Returns list of violated constraint strings. Empty = all satisfied."""
    violations = []

    if knowns.never_negative_caster and result.s_caster < 0:
        violations.append(f"negative caster ({result.s_caster:.2f}°)")

    if knowns.never_negative_trail and result.s_mech_trail < 0:
        violations.append(f"negative mech trail ({result.s_mech_trail:.3f}in)")

    sl_min = np.min(result.shock_length)
    sl_max = np.max(result.shock_length)
    if sl_min < knowns.shock_stroke_min:
        violations.append(f"shock too compressed ({sl_min:.2f}in)")
    if sl_max > knowns.shock_stroke_max:
        violations.append(f"shock too extended ({sl_max:.2f}in)")

    if hp.uca_outboard[2] <= hp.lca_outboard[2]:
        violations.append("UBJ below LBJ")

    uca_z = (hp.uca_inboard_front[2] + hp.uca_inboard_rear[2]) / 2
    lca_z = (hp.lca_inboard_front[2] + hp.lca_inboard_rear[2]) / 2
    if uca_z <= lca_z:
        violations.append("UCA inboard below LCA inboard")

    if abs(result.s_caster) > knowns.caster_max_deg:
        violations.append(f"caster out of physical range ({result.s_caster:.2f}°)")
    if abs(result.s_kpi) > knowns.kpi_max_deg:
        violations.append(f"KPI out of physical range ({result.s_kpi:.2f}°)")
    if abs(result.s_camber) > knowns.camber_max_deg:
        violations.append(f"camber out of physical range ({result.s_camber:.2f}°)")
    if abs(result.s_rc_height) > knowns.rc_height_max_in:
        violations.append(f"roll center height out of physical range ({result.s_rc_height:.1f}in)")
    if abs(result.s_swing_arm) > knowns.swing_arm_max_in:
        violations.append(f"swing arm length unstable/out of range ({result.s_swing_arm:.1f}in)")

    uca_mid_in = (hp.uca_inboard_front + hp.uca_inboard_rear) / 2
    lca_mid_in = (hp.lca_inboard_front + hp.lca_inboard_rear) / 2
    uca_angle_signed = np.degrees(np.arctan2(
        hp.uca_outboard[2] - uca_mid_in[2], hp.uca_outboard[1] - uca_mid_in[1]))
    lca_angle_signed = np.degrees(np.arctan2(
        hp.lca_outboard[2] - lca_mid_in[2], hp.lca_outboard[1] - lca_mid_in[1]))
    uca_angle = abs(uca_angle_signed)
    lca_angle = abs(lca_angle_signed)
    if uca_angle < knowns.min_arm_angle_deg:
        violations.append(f"UCA too close to flat in front view ({uca_angle:.2f}°)")
    if lca_angle < knowns.min_arm_angle_deg:
        violations.append(f"LCA too close to flat in front view ({lca_angle:.2f}°)")

    angle_sep = abs(uca_angle_signed - lca_angle_signed)
    if angle_sep < knowns.min_arm_separation_deg:
        violations.append(f"UCA/LCA nearly parallel to each other ({angle_sep:.2f}° apart)")

    # 4WD front halfshaft / CV joint check — only runs once a real diff
    # output location is set (see Knowns.diff_output_position docstring).
    # Approximation: uses the UBJ/LBJ midpoint at each travel step as a
    # stand-in for the hub/axle centerline (the solver doesn't separately
    # track a wheel-center point through travel), and measures how much
    # the halfshaft direction swings away from its static (ride-height)
    # direction across the bump/droop sweep — now INCLUDING steering input
    # at each travel step, since the solver can steer the knuckle. The worst
    # case is checked over the grid of (travel × steer), capturing the real
    # combined articulation (bumping while turning) rather than the old
    # vertical-only lower bound.
    if knowns.diff_output_position is not None:
        mid = result.frames[len(result.frames)//2]
        hub_static = (mid.uca_outboard + mid.lca_outboard) / 2
        dir_static = unit(hub_static - knowns.diff_output_position)
        max_swing = 0.0
        steer_angles = [-knowns.max_steer_deg, 0.0, knowns.max_steer_deg]
        for frame in result.frames:
            for steer in steer_angles:
                sf = _steer(frame, steer) if abs(steer) > 1e-9 else frame
                hub = (sf.uca_outboard + sf.lca_outboard) / 2
                dir_now = unit(hub - knowns.diff_output_position)
                ang = np.degrees(np.arccos(np.clip(np.dot(dir_static, dir_now), -1.0, 1.0)))
                max_swing = max(max_swing, ang)
        if max_swing > knowns.cv_max_angle_deg:
            violations.append(
                f"CV joint articulation exceeds spec ({max_swing:.1f}° > {knowns.cv_max_angle_deg}°, "
                f"travel + steering combined)")

    # Curve smoothness / stability across the travel sweep. Reject
    # geometries whose key curves spike violently between adjacent travel
    # steps — these are near-degenerate (front-view instant center
    # blowing up mid-travel) and not real usable suspensions, even when
    # their static sample-point values look fine. See the threshold
    # docstrings in Knowns.
    if len(result.travel) > 2:
        dt = np.diff(result.travel)
        dt_safe = np.where(np.abs(dt) > 1e-9, dt, 1e-9)
        rc_jump = float(np.max(np.abs(np.diff(result.rc_height) / dt_safe)))
        toe_jump = float(np.max(np.abs(np.diff(result.toe) / dt_safe)))
        cam_jump = float(np.max(np.abs(np.diff(result.camber) / dt_safe)))
        mr_jump = float(np.max(np.abs(np.diff(result.motion_ratio) / dt_safe)))
        if rc_jump > knowns.max_rc_jump_per_in:
            violations.append(f"roll center unstable through travel ({rc_jump:.0f} in/in jump)")
        if toe_jump > knowns.max_bumpsteer_jump:
            violations.append(f"bump steer unstable through travel ({toe_jump:.1f} deg/in jump)")
        if cam_jump > knowns.max_camber_jump:
            violations.append(f"camber unstable through travel ({cam_jump:.1f} deg/in jump)")
        if mr_jump > knowns.max_motion_ratio_jump:
            violations.append(f"motion ratio unstable through travel ({mr_jump:.2f}/in jump)")

    return violations


# ── Per-goal normalized residuals ────────────────────────────────────────────
def _goal_terms(result: SweepResult, goals: Goals):
    """
    Each entry: normalized residual (0 = perfect, ~1 = one target-width of
    error). Shared math behind both the legacy weighted score() and the
    multi-objective NSGA-II search.
    """
    if len(result.travel) > 1:
        dt = np.diff(result.travel)
        bs = float(np.max(np.abs(np.diff(result.toe) / (dt + 1e-9))))
    else:
        bs = 0.0

    if len(result.travel) > 3:
        mid = len(result.travel) // 2
        dt2 = np.diff(result.travel[mid-3:mid+3])
        dc  = np.diff(result.camber[mid-3:mid+3])
        cam_rate = float(np.mean(dc / (dt2 + 1e-9)))
    else:
        cam_rate = 0.0

    terms = {
        "bump_steer":   abs(bs) / (goals.bump_steer_target + 1e-9),
        "roll_center":  abs(result.s_rc_height - goals.roll_center_target) / 2.0,
        "camber_gain":  abs(cam_rate - goals.camber_gain_target) / 0.05,
        "kpi":          abs(result.s_kpi - goals.kpi_target) / 3.0,
        "scrub":        abs(result.s_scrub - goals.scrub_target) / 0.3,
        "motion_ratio": abs(result.s_motion_ratio - goals.motion_ratio_target) / 0.15,
        "ackermann":    abs(result.s_ackermann - goals.ackermann_target) / 20.0,
        "swing_arm":    abs(result.s_swing_arm - goals.swing_arm_target) / 25.0,
        "caster":       abs(result.s_caster - goals.caster_target) / 2.0,
        "mech_trail":   abs(result.s_mech_trail - goals.mech_trail_target) / 0.4,
        "static_camber":abs(result.s_camber - goals.static_camber_target) / 1.0,
    }
    display = {
        "Bump steer":   (bs, goals.bump_steer_target, "deg/in"),
        "Roll center":  (result.s_rc_height, goals.roll_center_target, "in"),
        "Camber gain":  (cam_rate, goals.camber_gain_target, "deg/in"),
        "KPI":          (result.s_kpi, goals.kpi_target, "deg"),
        "Scrub":        (result.s_scrub, goals.scrub_target, "in"),
        "Motion ratio": (result.s_motion_ratio, goals.motion_ratio_target, "—"),
        "Ackermann":    (result.s_ackermann, goals.ackermann_target, "%"),
        "Swing arm":    (result.s_swing_arm, goals.swing_arm_target, "in"),
        "Caster":       (result.s_caster, goals.caster_target, "deg"),
        "Mech trail":   (result.s_mech_trail, goals.mech_trail_target, "in"),
        "Static camber":(result.s_camber, goals.static_camber_target, "deg"),
    }
    return terms, display


def _goal_weight(goals: Goals, name: str) -> float:
    return getattr(goals, f"{name}_weight")


PRIMARY_GOAL_NAMES = ["bump_steer"]


def _primary_and_secondary(goals: Goals):
    """The goal(s) always treated as true Pareto objectives in NSGA-III —
    fixed by engineering priority, NOT by weight magnitude (see below for
    why those are different things). Currently just bump steer: Will's
    priority split (bump steer ~90%, KPI ~5%, roll center/camber gain
    ~2.5% each) is a single dominant priority, not 4 comparably-important
    ones. NSGA-III explores every primary axis as EQUALLY important
    during the actual search — weight only affects which point on the
    resulting front gets picked afterward. So 4 primary axes can't
    represent "one goal dominates," no matter how skewed their weights
    are; this was tested directly and confirmed broken (bump steer came
    back as the WORST of all 11 goals despite 90% weight, because the
    search spent equal effort spreading across all 4 "primary" axes
    instead of concentrating on bump steer). One primary objective +
    everything else folded into the weighted secondary composite actually
    makes the search chase bump steer hard, with roll center/camber
    gain/KPI trading off underneath it via their (much smaller) weights,
    same as every other secondary goal."""
    primary = PRIMARY_GOAL_NAMES
    secondary = [n for n in GOAL_TERMS if n not in PRIMARY_GOAL_NAMES]
    return primary, secondary


def score(result: SweepResult, goals: Goals) -> float:
    """Legacy single weighted-sum score (kept for display / final ranking)."""
    terms, _ = _goal_terms(result, goals)
    total_w = sum(_goal_weight(goals, n) for n in GOAL_TERMS) + 1e-9
    return sum(_goal_weight(goals, n) * terms[n] for n in GOAL_TERMS) / total_w


def score_breakdown(result: SweepResult, goals: Goals) -> dict:
    """Return per-goal display values: {name: (got, target, unit)}."""
    _, display = _goal_terms(result, goals)
    return display


def objective_vector(result: SweepResult, goals: Goals, primary, secondary) -> np.ndarray:
    """Build the NSGA-II objective vector: [primary goals..., secondary composite]."""
    terms, _ = _goal_terms(result, goals)
    obj = [terms[n] for n in primary]
    if secondary:
        sec_w = sum(_goal_weight(goals, n) for n in secondary) + 1e-9
        sec = sum(_goal_weight(goals, n) * terms[n] for n in secondary) / sec_w
        obj.append(sec)
    return np.array(obj, dtype=float)


# ── Optimizer result ──────────────────────────────────────────────────────────
@dataclass
class OptResult:
    hardpoints      : Hardpoints
    sweep           : SweepResult
    final_score     : float
    n_evals         : int
    converged       : bool
    violations      : list
    score_breakdown : dict
    pareto_front    : list = field(default_factory=list)
    engine          : str = "built-in"   # which optimizer engine ran: "pymoo" or "built-in"
    spindle_length  : float = 0.0        # OUTPUT, not input — see implied_spindle_length()


# ── NSGA-II (self-contained, numpy only) ─────────────────────────────────────
def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def _fast_nondominated_sort(objs: np.ndarray, feasible: np.ndarray):
    n = len(objs)
    feas_idx = [i for i in range(n) if feasible[i]]
    infeas_idx = [i for i in range(n) if not feasible[i]]

    fronts = []
    if feas_idx:
        S = {i: [] for i in feas_idx}
        dom_count = {i: 0 for i in feas_idx}
        for i in feas_idx:
            for j in feas_idx:
                if i == j:
                    continue
                if _dominates(objs[i], objs[j]):
                    S[i].append(j)
                elif _dominates(objs[j], objs[i]):
                    dom_count[i] += 1
        cur = [i for i in feas_idx if dom_count[i] == 0]
        while cur:
            fronts.append(cur)
            nxt = []
            for i in cur:
                for j in S[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0:
                        nxt.append(j)
            cur = nxt

    if infeas_idx:
        fronts.append(infeas_idx)
    return fronts


def _crowding_distance(objs: np.ndarray, front: list) -> dict:
    m = objs.shape[1]
    dist = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front:
            dist[i] = float("inf")
        return dist
    for k in range(m):
        vals = sorted(front, key=lambda i: objs[i, k])
        lo, hi = objs[vals[0], k], objs[vals[-1], k]
        rng = hi - lo if hi > lo else 1e-9
        dist[vals[0]] = float("inf")
        dist[vals[-1]] = float("inf")
        for p in range(1, len(vals)-1):
            dist[vals[p]] += (objs[vals[p+1], k] - objs[vals[p-1], k]) / rng
    return dist


def _tournament_select(pop_idx, rank, crowd, rng):
    i, j = rng.choice(pop_idx, 2, replace=False)
    if rank[i] < rank[j]:
        return i
    if rank[j] < rank[i]:
        return j
    return i if crowd[i] > crowd[j] else j


def _sbx_crossover(p1, p2, bounds, rng, eta=15.0):
    c1, c2 = p1.copy(), p2.copy()
    for k in range(len(p1)):
        if rng.random() > 0.9:
            continue
        lo, hi = bounds[k]
        x1, x2 = p1[k], p2[k]
        if abs(x1 - x2) < 1e-12:
            continue
        if x1 > x2:
            x1, x2 = x2, x1
        u = rng.random()
        beta = 1.0 + 2*(x1-lo)/(x2-x1+1e-12)
        alpha = 2.0 - beta**(-(eta+1))
        betaq = (u*alpha)**(1/(eta+1)) if u <= 1/alpha else (1/(2-u*alpha))**(1/(eta+1))
        v1 = 0.5*((x1+x2) - betaq*(x2-x1))
        beta = 1.0 + 2*(hi-x2)/(x2-x1+1e-12)
        alpha = 2.0 - beta**(-(eta+1))
        betaq = (u*alpha)**(1/(eta+1)) if u <= 1/alpha else (1/(2-u*alpha))**(1/(eta+1))
        v2 = 0.5*((x1+x2) + betaq*(x2-x1))
        c1[k] = float(np.clip(v1, lo, hi))
        c2[k] = float(np.clip(v2, lo, hi))
    return c1, c2


def _poly_mutate(p, bounds, rng, eta=20.0, p_mut=None):
    c = p.copy()
    n = len(p)
    if p_mut is None:
        p_mut = 1.0/n
    for k in range(n):
        if rng.random() > p_mut:
            continue
        lo, hi = bounds[k]
        x = c[k]
        if hi <= lo:
            continue
        delta1 = (x-lo)/(hi-lo); delta2 = (hi-x)/(hi-lo)
        u = rng.random()
        if u < 0.5:
            val = 2*u + (1-2*u)*(1-delta1)**(eta+1)
            deltaq = val**(1/(eta+1)) - 1
        else:
            val = 2*(1-u) + 2*(u-0.5)*(1-delta2)**(eta+1)
            deltaq = 1 - val**(1/(eta+1))
        c[k] = float(np.clip(x + deltaq*(hi-lo), lo, hi))
    return c


def _das_dennis(n_obj: int, n_partitions: int) -> np.ndarray:
    """Generate reference directions on the unit simplex (Das & Dennis,
    1998) — the systematic, evenly-spaced set of weight vectors NSGA-III
    uses to keep selection pressure spread across the WHOLE trade-off
    surface, including the balanced middle, not just the extremes."""
    def gen(n, p):
        if n == 1:
            return [[p]]
        out = []
        for i in range(p + 1):
            for tail in gen(n - 1, p - i):
                out.append([i] + tail)
        return out
    combos = np.array(gen(n_obj, n_partitions), dtype=float)
    return combos / n_partitions


def _ref_dirs_for(n_obj: int, target_count: int) -> np.ndarray:
    """Pick a partition count that gives roughly target_count reference
    directions for n_obj objectives."""
    from math import comb
    best_p, best_n = 1, n_obj
    for p in range(1, 20):
        n = comb(p + n_obj - 1, n_obj - 1)
        if abs(n - target_count) < abs(best_n - target_count):
            best_p, best_n = p, n
        if n > target_count * 2:
            break
    return _das_dennis(n_obj, best_p)


def _normalize_objectives(objs: np.ndarray, feasible: np.ndarray):
    """Ideal-point + extreme-point intercept normalization, standard
    NSGA-III preprocessing so objectives on very different scales (e.g.
    bump steer residual vs. swing-arm residual) don't distort which
    reference direction a candidate gets associated with."""
    n_obj = objs.shape[1]
    F = objs[feasible] if feasible.any() else objs
    ideal = F.min(axis=0)
    Ft = np.maximum(F - ideal, 0.0)

    extreme_points = np.zeros((n_obj, n_obj))
    for j in range(n_obj):
        w = np.full(n_obj, 1e-6)
        w[j] = 1.0
        asf = np.max(Ft / w, axis=1)
        extreme_points[j] = Ft[np.argmin(asf)]

    try:
        a = np.linalg.solve(extreme_points, np.ones(n_obj))
        intercepts = 1.0 / a
        if np.any(intercepts <= 1e-9) or not np.all(np.isfinite(intercepts)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        intercepts = Ft.max(axis=0)
        intercepts[intercepts < 1e-9] = 1e-9

    norm = np.maximum(objs - ideal, 0.0) / intercepts
    return norm


def _perp_distances(norm_objs: np.ndarray, ref_dirs: np.ndarray) -> np.ndarray:
    """Perpendicular distance from each normalized point to each reference
    direction's line through the origin. Returns (n_points, n_refdirs)."""
    ref_unit = ref_dirs / (np.linalg.norm(ref_dirs, axis=1, keepdims=True) + 1e-12)
    proj_len = norm_objs @ ref_unit.T                      # (n_pts, n_refs)
    proj = proj_len[:, :, None] * ref_unit[None, :, :]      # (n_pts, n_refs, n_obj)
    diff = norm_objs[:, None, :] - proj
    return np.linalg.norm(diff, axis=2)


def _nsga3_select(combined_objs, combined_feas, fronts, pop_size, ref_dirs, rng):
    """NSGA-III environmental selection: accept whole fronts until the next
    one would overflow, then fill the remaining slots by niche count
    (reference-direction association) instead of crowding distance — this
    is the part that keeps the final population spread evenly across the
    trade-off surface instead of collapsing toward extremes as the search
    runs longer."""
    accepted = []
    boundary = None
    for fr in fronts:
        if len(accepted) + len(fr) <= pop_size:
            accepted.extend(fr)
        else:
            boundary = fr
            break
    n_needed = pop_size - len(accepted)
    if boundary is None or n_needed <= 0:
        return accepted[:pop_size]

    pool_idx = accepted + boundary
    norm = _normalize_objectives(combined_objs[pool_idx], combined_feas[pool_idx])
    dist = _perp_distances(norm, ref_dirs)
    nearest_ref = np.argmin(dist, axis=1)

    local_pos = {idx: i for i, idx in enumerate(pool_idx)}
    accepted_set = set(accepted)
    niche_count = np.zeros(len(ref_dirs), dtype=int)
    for idx in accepted:
        niche_count[nearest_ref[local_pos[idx]]] += 1

    boundary_by_ref = {}
    for idx in boundary:
        r = nearest_ref[local_pos[idx]]
        boundary_by_ref.setdefault(r, []).append(idx)

    chosen = []
    while len(chosen) < n_needed:
        candidate_refs = [r for r, lst in boundary_by_ref.items() if lst]
        if not candidate_refs:
            break
        min_count = min(niche_count[r] for r in candidate_refs)
        best_refs = [r for r in candidate_refs if niche_count[r] == min_count]
        r = best_refs[rng.integers(len(best_refs))]
        lst = boundary_by_ref[r]
        if niche_count[r] == 0:
            pick = min(lst, key=lambda idx: dist[local_pos[idx], r])
        else:
            pick = lst[rng.integers(len(lst))]
        lst.remove(pick)
        if not lst:
            del boundary_by_ref[r]
        chosen.append(pick)
        niche_count[r] += 1

    return accepted + chosen


def run_nsga3_pymoo(
    evaluate: Callable,
    bounds: list,
    n_obj: int,
    pop_size: int = 92,
    generations: int = 100,
    seed: int = 42,
    progress_cb: Optional[Callable] = None,
    x0: Optional[np.ndarray] = None,
):
    """
    pymoo-backed NSGA-III. Returns the SAME tuple shape as the built-in
    run_nsga3 — (pop, objs, feas, payloads, front0, n_evals) — so the
    result-assembly code in run_optimizer doesn't care which engine ran.

    Uses pymoo's validated, faster NSGA-III implementation when the library
    is available. Constraints are handled the proper pymoo way (via the
    out["G"] inequality-constraint channel, where G<=0 means feasible)
    rather than the constraint-domination trick the built-in version uses —
    this is actually a cleaner formulation than mine.

    NOTE: this branch is written against pymoo's documented API but has NOT
    been executed/tested in the environment where it was written (pymoo
    wasn't installable there). The built-in run_nsga3 fallback IS fully
    tested. If this path misbehaves on first real use, that's the most
    likely culprit — compare against the built-in engine to isolate.
    """
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.core.callback import Callback

    n_var = len(bounds)
    xl = np.array([b[0] for b in bounds])
    xu = np.array([b[1] for b in bounds])

    # Cache: pymoo evaluates a whole population matrix at once, and we want
    # to keep each candidate's rich payload (hp/sweep/violations) keyed by
    # its exact parameter vector so we can recover it after the run.
    payload_cache = {}

    def _key(x):
        return tuple(np.round(x, 9))

    class SuspensionProblem(Problem):
        def __init__(self):
            # n_constr=1: a single aggregate feasibility constraint (number
            # of violated rules). G<=0 (zero violations) == feasible.
            super().__init__(n_var=n_var, n_obj=n_obj, n_constr=1,
                             xl=xl, xu=xu)

        def _evaluate(self, X, out, *args, **kwargs):
            F = np.zeros((len(X), n_obj))
            G = np.zeros((len(X), 1))
            for i, x in enumerate(X):
                obj, feasible, payload = evaluate(x)
                F[i, :] = obj
                # number of violations as the constraint value; 0 == feasible
                G[i, 0] = 0.0 if feasible else float(len(payload.get("violations", [1])))
                payload_cache[_key(x)] = (obj, feasible, payload)
            out["F"] = F
            out["G"] = G

    ref_dirs = get_reference_directions("das-dennis", n_obj,
                                        n_partitions=_pymoo_partitions(n_obj, pop_size))

    # Seed initial population around the baseline (same rationale as the
    # built-in engine: random search over ~20-30 dims rarely finds good
    # double-wishbone geometry, but the baseline is already close).
    if x0 is not None:
        n_pop = max(len(ref_dirs), pop_size)
        rng = np.random.default_rng(seed)
        span = (xu - xl)
        seeded = x0[None, :] + rng.normal(0, 0.06, size=(n_pop, n_var)) * span
        seeded = np.clip(seeded, xl, xu)
        seeded[0] = x0
        sampling = seeded
    else:
        sampling = FloatRandomSampling()

    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=max(len(ref_dirs), pop_size),
        sampling=sampling,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    n_eval_counter = {"n": 0}

    class ProgressCallback(Callback):
        def notify(self, algorithm):
            n_eval_counter["n"] = algorithm.evaluator.n_eval
            if progress_cb:
                # best aggregate (sum of objectives) among current feasible
                F = algorithm.pop.get("F")
                G = algorithm.pop.get("G")
                feas_mask = (G <= 0).all(axis=1) if G is not None else np.ones(len(F), bool)
                pool = F[feas_mask] if feas_mask.any() else F
                best = float(np.min(pool.sum(axis=1))) if len(pool) else float("inf")
                progress_cb(algorithm.n_gen, best)

    res = minimize(
        SuspensionProblem(),
        algorithm,
        ("n_gen", generations),
        seed=seed,
        callback=ProgressCallback(),
        verbose=False,
    )

    # Reassemble into the built-in engine's return format. We recover the
    # rich payloads from the cache. res.X / res.F hold the final non-
    # dominated set; we also pull the full final population for completeness.
    final_pop_X = res.pop.get("X")
    final_pop_F = res.pop.get("F")
    final_pop_G = res.pop.get("G")

    pop_list, objs_list, feas_list, payloads_list = [], [], [], []
    for i, x in enumerate(final_pop_X):
        cached = payload_cache.get(_key(x))
        if cached is None:
            obj, feasible, payload = evaluate(x)
        else:
            obj, feasible, payload = cached
        pop_list.append(x)
        objs_list.append(obj)
        feas_list.append(feasible)
        payloads_list.append(payload)

    pop = np.array(pop_list)
    objs = np.array(objs_list)
    feas = np.array(feas_list)

    # front0 = indices of the final non-dominated, feasible set
    front0 = [i for i in range(len(pop)) if feas[i]]
    if not front0:
        front0 = list(range(len(pop)))

    n_evals = n_eval_counter["n"] or len(payload_cache)
    return pop, objs, feas, payloads_list, front0, n_evals


def _pymoo_partitions(n_obj: int, target_pop: int) -> int:
    """Pick a das-dennis partition count giving roughly target_pop reference
    directions for n_obj objectives (mirrors the built-in _ref_dirs_for)."""
    from math import comb
    best_p, best_n = 1, n_obj
    for p in range(1, 20):
        n = comb(p + n_obj - 1, n_obj - 1)
        if abs(n - target_pop) < abs(best_n - target_pop):
            best_p, best_n = p, n
        if n > target_pop * 2:
            break
    return best_p


def run_nsga3(
    evaluate: Callable,
    bounds: list,
    n_obj: int,
    pop_size: int = 92,
    generations: int = 100,
    seed: int = 42,
    progress_cb: Optional[Callable] = None,
    x0: Optional[np.ndarray] = None,
):
    """
    NSGA-III: same overall genetic loop as NSGA-II (SBX crossover,
    polynomial mutation, baseline-seeded initial population, non-dominated
    sorting), but replaces crowding-distance truncation with reference-
    direction-based niching. This is what your project's original design
    called for — plain NSGA-II's crowding distance rewards spreading to
    the EDGES of the trade-off surface, which gets actively worse (more
    extreme, less balanced) the longer/bigger you run it. Reference
    directions guarantee even coverage of the WHOLE surface, including
    balanced middle-ground designs, regardless of population size.
    """
    rng = np.random.default_rng(seed)
    n_var = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    ref_dirs = _ref_dirs_for(n_obj, pop_size)
    pop_size = len(ref_dirs)  # NSGA-III convention: align pop size to ref dir count

    if x0 is not None:
        n_seeded = pop_size // 2
        span = (hi - lo)
        seeded = x0[None, :] + rng.normal(0, 0.06, size=(n_seeded, n_var)) * span
        seeded = np.clip(seeded, lo, hi)
        seeded[0] = x0  # always evaluate the exact baseline itself, not
                         # just noisy neighbors — elitism can only preserve
                         # what actually got evaluated
        n_random = pop_size - n_seeded
        random_part = rng.uniform(lo, hi, size=(n_random, n_var))
        pop = np.vstack([seeded, random_part])
    else:
        pop = rng.uniform(lo, hi, size=(pop_size, n_var))
    n_evals = 0

    def eval_pop(P):
        nonlocal n_evals
        objs, feas, payloads = [], [], []
        for x in P:
            o, f, pl = evaluate(x)
            objs.append(o); feas.append(f); payloads.append(pl)
            n_evals += 1
        return np.array(objs), np.array(feas), payloads

    objs, feas, payloads = eval_pop(pop)

    def _metric(i, objs_arr, feas_arr, payloads_list):
        if not feas_arr[i]:
            return float("inf")
        pl = payloads_list[i]
        return pl.get("legacy_score", float(objs_arr[i].sum())) if isinstance(pl, dict) else float(objs_arr[i].sum())

    elite = None  # (x, obj, feasible, payload, metric)
    def _update_elite(P, objs_arr, feas_arr, payloads_list):
        nonlocal elite
        for i in range(len(P)):
            if not feas_arr[i]:
                continue
            m = _metric(i, objs_arr, feas_arr, payloads_list)
            if elite is None or m < elite[4]:
                elite = (P[i].copy(), objs_arr[i].copy(), True, payloads_list[i], m)

    _update_elite(pop, objs, feas, payloads)

    for gen in range(generations):
        fronts = _fast_nondominated_sort(objs, feas)
        rank = {i: fi for fi, fr in enumerate(fronts) for i in fr}
        crowd = {}
        for fr in fronts:
            crowd.update(_crowding_distance(objs, fr))  # used only for parent selection pressure

        pop_idx = list(range(len(pop)))
        offspring = []
        while len(offspring) < pop_size:
            p1 = _tournament_select(pop_idx, rank, crowd, rng)
            p2 = _tournament_select(pop_idx, rank, crowd, rng)
            c1, c2 = _sbx_crossover(pop[p1], pop[p2], bounds, rng)
            c1 = _poly_mutate(c1, bounds, rng)
            c2 = _poly_mutate(c2, bounds, rng)
            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)
        offspring = np.array(offspring)

        o_objs, o_feas, o_payloads = eval_pop(offspring)
        _update_elite(offspring, o_objs, o_feas, o_payloads)

        combined_pop = np.vstack([pop, offspring])
        combined_objs = np.vstack([objs, o_objs])
        combined_feas = np.concatenate([feas, o_feas])
        combined_payloads = payloads + o_payloads

        fronts = _fast_nondominated_sort(combined_objs, combined_feas)
        new_idx = _nsga3_select(combined_objs, combined_feas, fronts, pop_size, ref_dirs, rng)

        pop = combined_pop[new_idx]
        objs = combined_objs[new_idx]
        feas = combined_feas[new_idx]
        payloads = [combined_payloads[i] for i in new_idx]

        # Elitist injection: NSGA-III's reference-direction niching optimizes
        # for even spread across the trade-off surface, not for preserving
        # any single scalar-best point — with enough objectives, the
        # literal best-by-score individual can legitimately get dropped in
        # favor of diversity. That's correct behavior for exploring the
        # full Pareto surface, but it means the search can't be trusted to
        # monotonically improve on its own. This guarantees it does: if the
        # tracked best-ever individual isn't already in the surviving
        # population, swap it in for whichever survivor currently scores
        # worst, every generation.
        if elite is not None:
            cur_metrics = [_metric(i, objs, feas, payloads) for i in range(len(pop))]
            if elite[4] < min(cur_metrics):
                worst_i = int(np.argmax(cur_metrics))
                pop[worst_i] = elite[0]
                objs[worst_i] = elite[1]
                feas[worst_i] = elite[2]
                payloads[worst_i] = elite[3]

        if progress_cb:
            best_feas = objs[feas] if feas.any() else objs
            best_score = float(np.min(best_feas.sum(axis=1))) if len(best_feas) else float("inf")
            progress_cb(gen+1, best_score)

    fronts = _fast_nondominated_sort(objs, feas)
    front0 = fronts[0] if fronts else list(range(len(pop)))
    return pop, objs, feas, payloads, front0, n_evals


def run_nsga2(
    evaluate: Callable,
    bounds: list,
    pop_size: int = 40,
    generations: int = 60,
    seed: int = 42,
    progress_cb: Optional[Callable] = None,
    x0: Optional[np.ndarray] = None,
):
    """
    Generic NSGA-II loop. Kept for reference/testing — run_optimizer now
    uses run_nsga3 instead (see its docstring for why). `evaluate` is
    called once per candidate vector x and must return
    (objective_vector, is_feasible, payload_dict).

    If `x0` is given (the baseline design's packed parameter vector), half
    the initial population is seeded as small perturbations around x0
    instead of pure-random samples across the whole bounds box. Random
    search over a ~20-30 dimensional hardpoint space essentially never
    finds a good double-wishbone geometry from scratch in a practical
    evaluation budget — but your starting geometry is already close to
    reasonable, so local exploration around it converges far faster than
    hoping a random point lands somewhere good.
    """
    rng = np.random.default_rng(seed)
    n_var = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    if x0 is not None:
        n_seeded = pop_size // 2
        span = (hi - lo)
        seeded = x0[None, :] + rng.normal(0, 0.06, size=(n_seeded, n_var)) * span
        seeded = np.clip(seeded, lo, hi)
        seeded[0] = x0
        n_random = pop_size - n_seeded
        random_part = rng.uniform(lo, hi, size=(n_random, n_var))
        pop = np.vstack([seeded, random_part])
    else:
        pop = rng.uniform(lo, hi, size=(pop_size, n_var))
    n_evals = 0

    def eval_pop(P):
        nonlocal n_evals
        objs, feas, payloads = [], [], []
        for x in P:
            o, f, pl = evaluate(x)
            objs.append(o); feas.append(f); payloads.append(pl)
            n_evals += 1
        return np.array(objs), np.array(feas), payloads

    objs, feas, payloads = eval_pop(pop)

    for gen in range(generations):
        fronts = _fast_nondominated_sort(objs, feas)
        rank = {}
        crowd = {}
        for fi, fr in enumerate(fronts):
            for i in fr:
                rank[i] = fi
            crowd.update(_crowding_distance(objs, fr))

        pop_idx = list(range(len(pop)))
        offspring = []
        while len(offspring) < pop_size:
            p1 = _tournament_select(pop_idx, rank, crowd, rng)
            p2 = _tournament_select(pop_idx, rank, crowd, rng)
            c1, c2 = _sbx_crossover(pop[p1], pop[p2], bounds, rng)
            c1 = _poly_mutate(c1, bounds, rng)
            c2 = _poly_mutate(c2, bounds, rng)
            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)
        offspring = np.array(offspring)

        o_objs, o_feas, o_payloads = eval_pop(offspring)

        combined_pop = np.vstack([pop, offspring])
        combined_objs = np.vstack([objs, o_objs])
        combined_feas = np.concatenate([feas, o_feas])
        combined_payloads = payloads + o_payloads

        fronts = _fast_nondominated_sort(combined_objs, combined_feas)
        new_idx = []
        for fr in fronts:
            if len(new_idx) + len(fr) <= pop_size:
                new_idx.extend(fr)
            else:
                cd = _crowding_distance(combined_objs, fr)
                fr_sorted = sorted(fr, key=lambda i: -cd[i])
                new_idx.extend(fr_sorted[:pop_size-len(new_idx)])
                break

        pop = combined_pop[new_idx]
        objs = combined_objs[new_idx]
        feas = combined_feas[new_idx]
        payloads = [combined_payloads[i] for i in new_idx]

        if progress_cb:
            best_feas = objs[feas] if feas.any() else objs
            best_score = float(np.min(best_feas.sum(axis=1))) if len(best_feas) else float("inf")
            progress_cb(gen+1, best_score)

    fronts = _fast_nondominated_sort(objs, feas)
    front0 = fronts[0] if fronts else list(range(len(pop)))
    return pop, objs, feas, payloads, front0, n_evals


# ── Run optimizer (public entry point — same signature as before) ───────────
def run_optimizer(
    base_hp    : Hardpoints,
    bounds     : dict,
    knowns     : Knowns,
    goals      : Goals,
    max_iter   : int = 300,
    popsize    : int = 10,
    progress_cb: Optional[Callable] = None,
) -> OptResult:
    """
    Runs NSGA-III multi-objective search (reference-direction based, see
    run_nsga3's docstring), then picks one recommended "best compromise"
    candidate from the Pareto front (using the legacy weighted score()
    over ALL goals, with a worst-term tie-break) so this still slots into
    app.py's existing single-result GUI unchanged. The rest of the front
    is attached as `.pareto_front` for future use (e.g. letting you click
    through alternative trade-off designs instead of only seeing one).

    max_iter is reused as `generations`, popsize as a population-size hint
    (NSGA-III then rounds the actual population to match its reference-
    direction count — usually close to what you asked for), to avoid
    changing the app.py call signature / GUI fields.
    """
    x0, sp_bounds, idx_map = _pack(base_hp, bounds,
                                   spindle_bounds=(knowns.spindle_length_min,
                                                   knowns.spindle_length_max))
    if not sp_bounds:
        r = run_sweep(base_hp, knowns.bump_in, knowns.droop_in)
        return OptResult(base_hp, r, score(r, goals), 0, True, [], score_breakdown(r, goals),
                         spindle_length=implied_spindle_length(base_hp))

    primary, secondary = _primary_and_secondary(goals)
    n_obj = len(primary) + (1 if secondary else 0)
    # Calibrated against this solver's actual per-eval cost (~0.05s/eval).
    # NOTE: after fixing the tie-rod/bump-steer bug and the front-view
    # roll-center instability (see solver.py), this is a genuinely harder
    # search than before — bump steer is now a real, tightly-toleranced
    # objective instead of a free win. These caps target ~3 minutes by
    # default; raise the GUI's "Max iter" / "Population" fields for a
    # longer, higher-quality search (e.g. before finalizing geometry for
    # fab) — with NSGA-III, a bigger budget converges toward a MORE
    # balanced result instead of drifting to extremes the way plain
    # NSGA-II did.
    generations = max(10, min(max_iter // 3, 250))
    pop_size = max(20, min(popsize * 4, 90))

    def evaluate(x):
        try:
            hp = _unpack(x, base_hp, idx_map)
            r = run_sweep(hp, knowns.bump_in, knowns.droop_in, steps=21)
            v = check_constraints(hp, knowns, r)
            obj = objective_vector(r, goals, primary, secondary)
            feasible = (len(v) == 0)
            return obj, feasible, {
                "hp": hp, "sweep": r, "violations": v,
                "legacy_score": score(r, goals),
            }
        except Exception:
            n_obj_local = len(primary) + (1 if secondary else 0)
            return np.full(n_obj_local, 1e6), False, {
                "hp": base_hp, "sweep": None, "violations": ["evaluation error"],
                "legacy_score": float("inf"),
            }

    def gen_progress(g, best):
        if progress_cb:
            progress_cb(g, best)

    # Engine selection: prefer pymoo's validated NSGA-III if the library is
    # installed; otherwise fall back to the built-in pure-numpy engine.
    # Both return (pop, objs, feas, payloads, front0, n_evals) so everything
    # below is identical regardless of which ran. The chosen engine name is
    # recorded on the result so the GUI can show which one was used.
    engine_used = "built-in"
    try:
        import pymoo  # noqa: F401
        use_pymoo = True
    except ImportError:
        use_pymoo = False

    if use_pymoo:
        try:
            pop, objs, feas, payloads, front0, n_evals = run_nsga3_pymoo(
                evaluate, sp_bounds, n_obj=n_obj, pop_size=pop_size,
                generations=generations, seed=42, progress_cb=gen_progress,
                x0=np.array(x0),
            )
            engine_used = "pymoo"
        except Exception as e:
            # If anything in the (untested-here) pymoo path fails at runtime,
            # don't crash the tool — fall back to the tested built-in engine.
            print(f"[optimizer] pymoo engine failed ({e}); "
                  f"falling back to built-in NSGA-III.")
            pop, objs, feas, payloads, front0, n_evals = run_nsga3(
                evaluate, sp_bounds, n_obj=n_obj, pop_size=pop_size,
                generations=generations, seed=42, progress_cb=gen_progress,
                x0=np.array(x0),
            )
            engine_used = "built-in (pymoo failed)"
    else:
        pop, objs, feas, payloads, front0, n_evals = run_nsga3(
            evaluate, sp_bounds, n_obj=n_obj, pop_size=pop_size,
            generations=generations, seed=42, progress_cb=gen_progress,
            x0=np.array(x0),
        )

    front_results = []
    for i in front0:
        pl = payloads[i]
        if pl["sweep"] is None:
            continue
        r = pl["sweep"]
        front_results.append(OptResult(
            hardpoints=pl["hp"], sweep=r,
            final_score=score(r, goals), n_evals=0,
            converged=True, violations=pl["violations"],
            score_breakdown=score_breakdown(r, goals),
            spindle_length=implied_spindle_length(pl["hp"]),
        ))

    if not front_results:
        best_i = int(np.argmin(objs.sum(axis=1)))
        pl = payloads[best_i]
        r = pl["sweep"] or run_sweep(base_hp, knowns.bump_in, knowns.droop_in)
        return OptResult(
            hardpoints=pl["hp"], sweep=r, final_score=score(r, goals),
            n_evals=n_evals, converged=False, violations=pl["violations"],
            score_breakdown=score_breakdown(r, goals), pareto_front=[],
            engine=engine_used, spindle_length=implied_spindle_length(pl["hp"]),
        )

    # Rank by (worst single normalized goal, then overall weighted score).
    # Sorting on weighted score alone lets one accidentally-perfect term
    # (e.g. bump steer landing on exactly 0.000 by coincidence) outweigh
    # five other goals being wrecked. Checking the worst term first favors
    # balanced designs over degenerate ones.
    def _worst_term(o: OptResult) -> float:
        terms, _ = _goal_terms(o.sweep, goals)
        return max(terms.values())

    front_results.sort(key=lambda o: (_worst_term(o), o.final_score))
    best = front_results[0]
    best.n_evals = n_evals
    best.converged = True
    best.pareto_front = front_results
    best.engine = engine_used
    for fr in front_results:
        fr.engine = engine_used

    return best
