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
HALF_TRACK_IN  = 26.0       # 52in track / 2
WHEELBASE_IN   = 64.0
CHASSIS_HALF   = 6.0        # 12in front width / 2
RIDE_HEIGHT_IN = 16.0


# ── Hardpoints ───────────────────────────────────────────────────────────────
@dataclass
class Hardpoints:
    """
    10 pickup points, all relative to wheel center origin.
    Inboard chassis points: negative Y, small X spread (front/rear legs).
    Outboard knuckle points: Y near 0, above/below wheel center.
    """
    uca_inboard_front : np.ndarray = field(default_factory=lambda: np.array([ 2.5, -7.0,  3.5]))
    uca_inboard_rear  : np.ndarray = field(default_factory=lambda: np.array([-2.5, -7.0,  3.5]))
    uca_outboard      : np.ndarray = field(default_factory=lambda: np.array([-0.44,-0.5,  2.4]))
    lca_inboard_front : np.ndarray = field(default_factory=lambda: np.array([ 3.2, -7.5, -2.5]))
    lca_inboard_rear  : np.ndarray = field(default_factory=lambda: np.array([-3.2, -7.5, -2.5]))
    lca_outboard      : np.ndarray = field(default_factory=lambda: np.array([ 0.0, -0.5, -2.5]))
    tie_rod_inboard   : np.ndarray = field(default_factory=lambda: np.array([-1.5, -5.5, -2.0]))
    tie_rod_outboard  : np.ndarray = field(default_factory=lambda: np.array([-1.5, -0.5, -2.0]))
    shock_upper       : np.ndarray = field(default_factory=lambda: np.array([ 0.0, -5.0,  7.0]))
    shock_lower       : np.ndarray = field(default_factory=lambda: np.array([ 0.0, -2.0, -2.0]))

    def copy(self):
        return Hardpoints(**{k: v.copy() for k, v in self.__dict__.items()})

    def to_dict(self):
        return {k: v.tolist() for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: np.array(v) for k, v in d.items()})

    def point_names(self):
        return list(self.__dict__.keys())

    def all_points(self):
        return {k: v for k, v in self.__dict__.items()}


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
    if abs(cross) < 1e-9:
        return (p1+p3)/2
    t = ((p3[0]-p1[0])*d2[1] - (p3[1]-p1[1])*d2[0]) / cross
    return p1 + t*d1


# ── Static geometry calculations ─────────────────────────────────────────────
def calc_camber(ubj, lbj):
    d = ubj - lbj
    return float(np.degrees(np.arctan2(d[1], d[2])))

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

def calc_ackermann(tie_out, tie_in, ubj, lbj):
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

    knuckle_dz = ((m.lca_outboard[2]-hp.lca_outboard[2]) +
                  (m.uca_outboard[2]-hp.uca_outboard[2])) / 2
    m.tie_rod_outboard = hp.tie_rod_outboard + np.array([0.,0.,knuckle_dz])
    m.shock_lower = hp.shock_lower + (m.lca_outboard - hp.lca_outboard)
    return m


def _mirror_right(hp: Hardpoints) -> Hardpoints:
    hp_r = hp.copy()
    for k in hp_r.__dict__:
        v = getattr(hp_r, k)
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

    for i, dz in enumerate(travel):
        ml = _move(hp,   dz)
        mr_hp = _move(hp_r, dz)
        frames.append(ml)

        camber[i] = calc_camber(ml.uca_outboard, ml.lca_outboard)
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
        s_ackermann   = calc_ackermann(hp.tie_rod_outboard, hp.tie_rod_inboard,
                                        hp.uca_outboard, hp.lca_outboard),
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
        lines.append(f"{name},{val[0]:.4f},{val[1]:.4f},{val[2]:.4f}")
    open(path,"w").write("\n".join(lines))

def export_kinematics_csv(r: SweepResult, path: str):
    hdr = "travel_in,camber_deg,toe_deg,caster_deg,rc_height_in,rc_lat_in,motion_ratio,track_chg_in"
    rows = [f"{r.travel[i]:.3f},{r.camber[i]:.4f},{r.toe[i]:.4f},{r.caster[i]:.4f},"
            f"{r.rc_height[i]:.4f},{r.rc_lateral[i]:.4f},{r.motion_ratio[i]:.4f},{r.track_change[i]:.4f}"
            for i in range(len(r.travel))]
    open(path,"w").write(hdr+"\n"+"\n".join(rows))