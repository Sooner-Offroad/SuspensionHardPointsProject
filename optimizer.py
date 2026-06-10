"""
optimizer.py — Suspension Hardpoint Optimizer
Sooner Offroad Baja SAE — Teddy 2025-26

Two-layer system:
  Knowns  → hard constraints  (violations = infinite penalty, rejected outright)
  Goals   → soft targets      (weighted penalty score, minimize)

Uses scipy differential_evolution (genetic algorithm).
"""

import numpy as np
from scipy.optimize import differential_evolution
from dataclasses import dataclass, field
from typing import Optional, Callable
from solver import (
    Hardpoints, SweepResult, run_sweep,
    calc_camber, calc_caster, calc_kpi, calc_scrub,
    calc_mechanical_trail, calc_ackermann,
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
    # Chassis
    chassis_half_width    : float = 6.0      # half of 12in front width
    wheelbase             : float = 64.0
    ride_height           : float = 16.0

    # Inboard pivot height limits (Z from wheel center)
    uca_inboard_z_min     : float = 2.0
    uca_inboard_z_max     : float = 5.5
    lca_inboard_z_min     : float = -4.0
    lca_inboard_z_max     : float = -1.0

    # Inboard pivot Y limits (how far inboard chassis mounts can be)
    uca_inboard_y_min     : float = -10.0
    uca_inboard_y_max     : float = -4.0
    lca_inboard_y_min     : float = -10.0
    lca_inboard_y_max     : float = -4.0

    # Outboard point limits (knuckle geometry)
    ubj_z_min             : float = 1.5
    ubj_z_max             : float = 4.5
    lbj_z_min             : float = -4.0
    lbj_z_max             : float = -1.0
    bj_y_min              : float = -2.0     # ball joints stay near wheel centerline
    bj_y_max              : float = 0.5

    # Shock stroke (limits how far shock_lower can move from upper)
    shock_stroke_min      : float = 3.0      # inches, minimum shock length
    shock_stroke_max      : float = 14.0     # inches, maximum shock length

    # Travel requirements
    bump_in               : float = 5.0
    droop_in              : float = 5.0

    # Clearance
    min_jounce_clearance  : float = 0.5      # inches clearance at full bump

    # Hard rules (boolean)
    never_negative_caster : bool  = True
    never_negative_trail  : bool  = True


# ── Goals (soft targets) ──────────────────────────────────────────────────────
@dataclass
class Goals:
    """
    Kinematic targets from Teddy geometry doc.
    Each has a target value and a weight (0-100).
    Higher weight = optimizer prioritizes it more.
    """
    # Target value, weight, (tolerance for hard-ish boundary)
    bump_steer_target     : float = 0.020    # deg/in max
    bump_steer_weight     : float = 25.0

    roll_center_target    : float = 3.0      # inches above ground
    roll_center_weight    : float = 20.0

    camber_gain_target    : float = -0.08    # deg/in
    camber_gain_weight    : float = 15.0

    kpi_target            : float = 9.5      # degrees
    kpi_weight            : float = 12.0

    scrub_target          : float = 0.35     # inches
    scrub_weight          : float = 10.0

    motion_ratio_target   : float = 0.65
    motion_ratio_weight   : float = 10.0

    ackermann_target      : float = 95.0     # percent
    ackermann_weight      : float = 8.0

    swing_arm_target      : float = 75.0     # inches
    swing_arm_weight      : float = 5.0

    caster_target         : float = 5.5      # degrees
    caster_weight         : float = 5.0

    mech_trail_target     : float = 0.8      # inches
    mech_trail_weight     : float = 5.0

    static_camber_target  : float = -1.0     # degrees
    static_camber_weight  : float = 5.0


# ── Point bounds ──────────────────────────────────────────────────────────────
@dataclass
class PointBounds:
    x_min: float = -5.0; x_max: float = 5.0; x_locked: bool = False
    y_min: float = -10.0; y_max: float = 1.0; y_locked: bool = False
    z_min: float = -5.0; z_max: float = 8.0; z_locked: bool = False


def make_bounds_from_knowns(hp: Hardpoints, knowns: Knowns) -> dict:
    """Auto-generate point bounds from Knowns constraints."""
    R = 2.0  # default ± range for unconstrained axes
    b = {}

    # UCA inboard front
    v = hp.uca_inboard_front
    b["uca_inboard_front"] = PointBounds(
        x_min=v[0]-R, x_max=v[0]+R,
        y_min=knowns.uca_inboard_y_min, y_max=knowns.uca_inboard_y_max,
        z_min=knowns.uca_inboard_z_min, z_max=knowns.uca_inboard_z_max)

    # UCA inboard rear
    v = hp.uca_inboard_rear
    b["uca_inboard_rear"] = PointBounds(
        x_min=v[0]-R, x_max=v[0]+R,
        y_min=knowns.uca_inboard_y_min, y_max=knowns.uca_inboard_y_max,
        z_min=knowns.uca_inboard_z_min, z_max=knowns.uca_inboard_z_max)

    # UCA outboard (UBJ)
    b["uca_outboard"] = PointBounds(
        x_min=-2.0, x_max=2.0,
        y_min=knowns.bj_y_min, y_max=knowns.bj_y_max,
        z_min=knowns.ubj_z_min, z_max=knowns.ubj_z_max)

    # LCA inboard front
    v = hp.lca_inboard_front
    b["lca_inboard_front"] = PointBounds(
        x_min=v[0]-R, x_max=v[0]+R,
        y_min=knowns.lca_inboard_y_min, y_max=knowns.lca_inboard_y_max,
        z_min=knowns.lca_inboard_z_min, z_max=knowns.lca_inboard_z_max)

    # LCA inboard rear
    v = hp.lca_inboard_rear
    b["lca_inboard_rear"] = PointBounds(
        x_min=v[0]-R, x_max=v[0]+R,
        y_min=knowns.lca_inboard_y_min, y_max=knowns.lca_inboard_y_max,
        z_min=knowns.lca_inboard_z_min, z_max=knowns.lca_inboard_z_max)

    # LCA outboard (LBJ)
    b["lca_outboard"] = PointBounds(
        x_min=-2.0, x_max=2.0,
        y_min=knowns.bj_y_min, y_max=knowns.bj_y_max,
        z_min=knowns.lbj_z_min, z_max=knowns.lbj_z_max)

    # Tie rod inboard — constrained near rack position
    v = hp.tie_rod_inboard
    b["tie_rod_inboard"] = PointBounds(
        x_min=v[0]-2.0, x_max=v[0]+2.0,
        y_min=-8.0, y_max=-3.0,
        z_min=-3.5, z_max=0.0)

    # Tie rod outboard — follows knuckle, near LBJ height
    b["tie_rod_outboard"] = PointBounds(
        x_min=-3.0, x_max=0.5,
        y_min=knowns.bj_y_min, y_max=knowns.bj_y_max,
        z_min=-3.5, z_max=0.5)

    # Shock upper — chassis mount, high up
    v = hp.shock_upper
    b["shock_upper"] = PointBounds(
        x_min=v[0]-1.5, x_max=v[0]+1.5,
        y_min=-7.0, y_max=-3.0,
        z_min=4.0, z_max=9.0)

    # Shock lower — on LCA, near outboard pivot
    v = hp.shock_lower
    b["shock_lower"] = PointBounds(
        x_min=v[0]-1.5, x_max=v[0]+1.5,
        y_min=-4.0, y_max=-0.5,
        z_min=-3.5, z_max=0.0)

    return b


# ── Pack / unpack parameter vector ───────────────────────────────────────────
def _pack(hp: Hardpoints, bounds: dict):
    x0, sp_bounds, idx_map = [], [], []
    for name in POINT_NAMES:
        val = getattr(hp, name)
        b = bounds[name]
        for ai, (locked, lo, hi, v) in enumerate([
            (b.x_locked, b.x_min, b.x_max, val[0]),
            (b.y_locked, b.y_min, b.y_max, val[1]),
            (b.z_locked, b.z_min, b.z_max, val[2]),
        ]):
            if not locked:
                x0.append(np.clip(v, lo, hi))
                sp_bounds.append((lo, hi))
                idx_map.append((name, ai))
    return x0, sp_bounds, idx_map


def _unpack(x, base_hp: Hardpoints, idx_map) -> Hardpoints:
    hp = base_hp.copy()
    for i, (name, axis) in enumerate(idx_map):
        arr = getattr(hp, name).copy()
        arr[axis] = x[i]
        setattr(hp, name, arr)
    return hp


# ── Constraint checker ────────────────────────────────────────────────────────
def check_constraints(hp: Hardpoints, knowns: Knowns, result: SweepResult) -> list:
    """
    Returns list of violated constraint strings.
    Empty list = all constraints satisfied.
    """
    violations = []

    # Caster
    if knowns.never_negative_caster and result.s_caster < 0:
        violations.append(f"negative caster ({result.s_caster:.2f}°)")

    # Mechanical trail
    if knowns.never_negative_trail and result.s_mech_trail < 0:
        violations.append(f"negative mech trail ({result.s_mech_trail:.3f}in)")

    # Shock length bounds
    sl_min = np.min(result.shock_length)
    sl_max = np.max(result.shock_length)
    if sl_min < knowns.shock_stroke_min:
        violations.append(f"shock too compressed ({sl_min:.2f}in)")
    if sl_max > knowns.shock_stroke_max:
        violations.append(f"shock too extended ({sl_max:.2f}in)")

    # UBJ must be above LBJ
    if hp.uca_outboard[2] <= hp.lca_outboard[2]:
        violations.append("UBJ below LBJ")

    # UCA inboard must be above LCA inboard
    uca_z = (hp.uca_inboard_front[2] + hp.uca_inboard_rear[2]) / 2
    lca_z = (hp.lca_inboard_front[2] + hp.lca_inboard_rear[2]) / 2
    if uca_z <= lca_z:
        violations.append("UCA inboard below LCA inboard")

    return violations


# ── Score function ────────────────────────────────────────────────────────────
def score(result: SweepResult, goals: Goals) -> float:
    """
    Compute weighted penalty. Lower = better.
    Each term normalized so 1.0 = roughly one target-width of error.
    """
    total_w = sum([
        goals.bump_steer_weight, goals.roll_center_weight,
        goals.camber_gain_weight, goals.kpi_weight,
        goals.scrub_weight, goals.motion_ratio_weight,
        goals.ackermann_weight, goals.swing_arm_weight,
        goals.caster_weight, goals.mech_trail_weight,
        goals.static_camber_weight,
    ]) + 1e-9

    penalty = 0.0

    # Bump steer (max toe change per inch)
    if len(result.travel) > 1:
        dt = np.diff(result.travel)
        bs = np.max(np.abs(np.diff(result.toe) / (dt+1e-9)))
    else:
        bs = 0.0
    penalty += (goals.bump_steer_weight/total_w) * (bs / (goals.bump_steer_target+1e-9))

    # Roll center height
    rc_err = abs(result.s_rc_height - goals.roll_center_target)
    penalty += (goals.roll_center_weight/total_w) * (rc_err / 2.0)

    # Camber gain rate
    if len(result.travel) > 3:
        mid = len(result.travel)//2
        dt2 = np.diff(result.travel[mid-3:mid+3])
        dc  = np.diff(result.camber[mid-3:mid+3])
        cam_rate = float(np.mean(dc/(dt2+1e-9)))
    else:
        cam_rate = 0.0
    penalty += (goals.camber_gain_weight/total_w) * (abs(cam_rate-goals.camber_gain_target)/0.05)

    # KPI
    penalty += (goals.kpi_weight/total_w) * (abs(result.s_kpi-goals.kpi_target)/3.0)

    # Scrub
    penalty += (goals.scrub_weight/total_w) * (abs(result.s_scrub-goals.scrub_target)/0.3)

    # Motion ratio
    penalty += (goals.motion_ratio_weight/total_w) * (abs(result.s_motion_ratio-goals.motion_ratio_target)/0.15)

    # Ackermann
    penalty += (goals.ackermann_weight/total_w) * (abs(result.s_ackermann-goals.ackermann_target)/20.0)

    # Swing arm
    penalty += (goals.swing_arm_weight/total_w) * (abs(result.s_swing_arm-goals.swing_arm_target)/25.0)

    # Caster
    penalty += (goals.caster_weight/total_w) * (abs(result.s_caster-goals.caster_target)/2.0)

    # Mechanical trail
    penalty += (goals.mech_trail_weight/total_w) * (abs(result.s_mech_trail-goals.mech_trail_target)/0.4)

    # Static camber
    penalty += (goals.static_camber_weight/total_w) * (abs(result.s_camber-goals.static_camber_target)/1.0)

    return penalty


# ── Optimizer result ──────────────────────────────────────────────────────────
@dataclass
class OptResult:
    hardpoints   : Hardpoints
    sweep        : SweepResult
    final_score  : float
    n_evals      : int
    converged    : bool
    violations   : list
    score_breakdown: dict


def score_breakdown(result: SweepResult, goals: Goals) -> dict:
    """Return per-goal score components for display."""
    if len(result.travel) > 1:
        dt = np.diff(result.travel)
        bs = float(np.max(np.abs(np.diff(result.toe)/(dt+1e-9))))
    else:
        bs = 0.0

    if len(result.travel) > 3:
        mid = len(result.travel)//2
        dt2 = np.diff(result.travel[mid-3:mid+3])
        dc  = np.diff(result.camber[mid-3:mid+3])
        cam_rate = float(np.mean(dc/(dt2+1e-9)))
    else:
        cam_rate = 0.0

    return {
        "Bump steer":    (bs,                  goals.bump_steer_target,  "deg/in"),
        "Roll center":   (result.s_rc_height,  goals.roll_center_target, "in"),
        "Camber gain":   (cam_rate,             goals.camber_gain_target, "deg/in"),
        "KPI":           (result.s_kpi,         goals.kpi_target,         "deg"),
        "Scrub":         (result.s_scrub,       goals.scrub_target,       "in"),
        "Motion ratio":  (result.s_motion_ratio,goals.motion_ratio_target,"—"),
        "Ackermann":     (result.s_ackermann,   goals.ackermann_target,   "%"),
        "Swing arm":     (result.s_swing_arm,   goals.swing_arm_target,   "in"),
        "Caster":        (result.s_caster,      goals.caster_target,      "deg"),
        "Mech trail":    (result.s_mech_trail,  goals.mech_trail_target,  "in"),
        "Static camber": (result.s_camber,      goals.static_camber_target,"deg"),
    }


# ── Run optimizer ─────────────────────────────────────────────────────────────
def run_optimizer(
    base_hp    : Hardpoints,
    bounds     : dict,
    knowns     : Knowns,
    goals      : Goals,
    max_iter   : int = 300,
    popsize    : int = 10,
    progress_cb: Optional[Callable] = None,
) -> OptResult:

    _, sp_bounds, idx_map = _pack(base_hp, bounds)
    if not sp_bounds:
        r = run_sweep(base_hp, knowns.bump_in, knowns.droop_in)
        return OptResult(base_hp, r, score(r,goals), 0, True, [], score_breakdown(r,goals))

    n_evals    = [0]
    best_score = [float("inf")]
    gen_count  = [0]

    def objective(x):
        n_evals[0] += 1
        try:
            hp = _unpack(x, base_hp, idx_map)
            r  = run_sweep(hp, knowns.bump_in, knowns.droop_in, steps=21)
            v  = check_constraints(hp, knowns, r)
            if v:
                return 1e6 + len(v)*1000
            s = score(r, goals)
            if s < best_score[0]:
                best_score[0] = s
            return s
        except Exception:
            return 1e7

    def callback(xk, convergence):
        gen_count[0] += 1
        if progress_cb:
            progress_cb(gen_count[0], best_score[0])
        return False

    opt = differential_evolution(
        objective, bounds=sp_bounds,
        maxiter=max_iter, popsize=popsize,
        tol=1e-5, mutation=(0.5,1.0), recombination=0.7,
        seed=42, callback=callback, polish=True, workers=1,
    )

    best_hp = _unpack(opt.x, base_hp, idx_map)
    best_r  = run_sweep(best_hp, knowns.bump_in, knowns.droop_in)
    viols   = check_constraints(best_hp, knowns, best_r)

    return OptResult(
        hardpoints    = best_hp,
        sweep         = best_r,
        final_score   = float(opt.fun),
        n_evals       = n_evals[0],
        converged     = opt.success,
        violations    = viols,
        score_breakdown = score_breakdown(best_r, goals),
    )