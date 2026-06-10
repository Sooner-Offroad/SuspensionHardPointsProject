"""
app.py — Suspension Design Tool
Sooner Offroad Baja SAE — Teddy 2025-26

Tab 1: Visualizer  — enter hardpoints, see 3D geometry animate + kinematic charts
Tab 2: Optimizer   — enter knowns + goals, genetic algorithm finds the hardpoints

Run:  python app.py
Deps: pip install numpy scipy matplotlib
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.animation as animation

from solver import (
    Hardpoints, SweepResult, run_sweep, POINT_LABELS,
    export_hardpoints_csv, export_kinematics_csv,
    spring_for_frequency, wheel_rate, ride_frequency,
    TIRE_RADIUS_IN, HALF_TRACK_IN,
)
from optimizer import (
    Knowns, Goals, PointBounds,
    make_bounds_from_knowns, run_optimizer, OptResult, POINT_NAMES,
)

# ── Theme ────────────────────────────────────────────────────────────────────
BG      = "#1a1a2e"
BG2     = "#16213e"
BG3     = "#0f3460"
ACCENT  = "#7c6af7"
TEXT    = "#e0e0e0"
DIM     = "#888888"
GREEN   = "#00d4aa"
YELLOW  = "#ffd700"
RED     = "#ff6b6b"
ORANGE  = "#ff9f43"
BORDER  = "#2d2d4e"

TARGETS = {
    "s_camber":       ((-0.5, 0.5),    "deg",   "0 deg"),
    "s_caster":       ((4.0, 8.0),     "deg",   "4-8"),
    "s_kpi":          ((7.0, 12.0),    "deg",   "7-12"),
    "s_scrub":        ((0.1, 0.6),     "in",    "0.1-0.6"),
    "s_mech_trail":   ((0.3, 1.5),     "in",    ">0"),
    "s_toe":          ((-0.2, 0.2),    "deg",   "0"),
    "s_rc_height":    ((1.0, 5.0),     "in",    "2-4"),
    "s_motion_ratio": ((0.55, 0.75),   "—",     "0.55-0.75"),
    "s_swing_arm":    ((50.0, 100.0),  "in",    "50-100"),
    "s_ackermann":    ((80.0, 115.0),  "%",     "80-110"),
}

STAT_LABELS = {
    "s_camber":       "Camber",
    "s_caster":       "Caster",
    "s_kpi":          "KPI",
    "s_scrub":        "Scrub",
    "s_mech_trail":   "Mech trail",
    "s_toe":          "Toe",
    "s_rc_height":    "RC height",
    "s_motion_ratio": "Motion ratio",
    "s_swing_arm":    "Swing arm",
    "s_ackermann":    "Ackermann",
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def lbl(parent, text, fg=None, font=None, **kw):
    return tk.Label(parent, text=text, bg=BG2,
                    fg=fg or TEXT, font=font or ("Consolas",9), **kw)

def entry_sv(parent, sv, width=8, bg=BG):
    return tk.Entry(parent, textvariable=sv, bg=bg, fg=TEXT,
                    insertbackground=TEXT, font=("Consolas",9),
                    width=width, bd=0, highlightthickness=1,
                    highlightbackground=BORDER, highlightcolor=ACCENT)

def section_header(parent, text):
    tk.Label(parent, text=text, bg=BG2, fg=ACCENT,
             font=("Consolas",10,"bold")).pack(fill="x", padx=8, pady=(8,2))


# ── Hardpoint entry panel ─────────────────────────────────────────────────────
class HardpointPanel(tk.Frame):
    def __init__(self, master, on_change_cb, **kw):
        super().__init__(master, bg=BG2, **kw)
        self.on_change_cb = on_change_cb
        self.vars = {}
        self._build()

    def _build(self):
        tk.Label(self, text="Hardpoints", bg=BG2, fg=ACCENT,
                 font=("Consolas",12,"bold")).pack(fill="x", padx=8, pady=(8,2))
        tk.Label(self, text="Origin = wheel center  |  +X fwd  +Y out  +Z up  (inches)",
                 bg=BG2, fg=DIM, font=("Consolas",8)).pack(fill="x", padx=8)

        hdr = tk.Frame(self, bg=BG2); hdr.pack(fill="x", padx=8, pady=(4,0))
        tk.Label(hdr, text="Point", bg=BG2, fg=DIM,
                 font=("Consolas",8), width=20, anchor="w").grid(row=0, column=0)
        for c,t in enumerate(["X","Y","Z"],1):
            tk.Label(hdr, text=t, bg=BG2, fg=DIM,
                     font=("Consolas",8), width=7).grid(row=0, column=c)

        canvas = tk.Canvas(self, bg=BG2, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=BG2)
        canvas.create_window((0,0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-(e.delta//120),"units"))

        hp = Hardpoints()
        for ri, name in enumerate(POINT_NAMES):
            val = getattr(hp, name)
            bg  = BG if ri%2==0 else BG2
            svars = []
            tk.Label(inner, text=POINT_LABELS[name], bg=bg, fg=TEXT,
                     font=("Consolas",9), width=20, anchor="w",
                     padx=4).grid(row=ri, column=0, sticky="w")
            for ci, v in enumerate(val):
                sv = tk.StringVar(value=f"{v:.3f}")
                sv.trace_add("write", lambda *a: self.on_change_cb())
                tk.Entry(inner, textvariable=sv, bg=bg, fg=TEXT,
                         insertbackground=TEXT, font=("Consolas",9),
                         width=7, bd=0, highlightthickness=1,
                         highlightbackground=BORDER, highlightcolor=ACCENT
                         ).grid(row=ri, column=ci+1, padx=2, pady=2)
                svars.append(sv)
            self.vars[name] = svars

    def get_hp(self):
        hp = Hardpoints()
        for name, svars in self.vars.items():
            try:
                setattr(hp, name, np.array([float(s.get()) for s in svars]))
            except ValueError:
                pass
        return hp

    def set_hp(self, hp: Hardpoints):
        for name, svars in self.vars.items():
            for sv, v in zip(svars, getattr(hp, name)):
                sv.set(f"{v:.3f}")


# ── Stats panel ───────────────────────────────────────────────────────────────
class StatsPanel(tk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, bg=BG2, **kw)
        section_header(self, "Static values")
        self._lbls = {}
        for key, label in STAT_LABELS.items():
            rng, unit, tgt = TARGETS.get(key, (None,"",""))
            f = tk.Frame(self, bg=BG2); f.pack(fill="x", padx=8, pady=1)
            tk.Label(f, text=label, bg=BG2, fg=DIM,
                     font=("Consolas",9), width=13, anchor="w").pack(side="left")
            vl = tk.Label(f, text="—", bg=BG2, fg=GREEN,
                          font=("Consolas",9,"bold"), width=8, anchor="e")
            vl.pack(side="left")
            tk.Label(f, text=unit, bg=BG2, fg=DIM,
                     font=("Consolas",8), width=4).pack(side="left")
            tk.Label(f, text=f"({tgt})", bg=BG2, fg=DIM,
                     font=("Consolas",8)).pack(side="left")
            self._lbls[key] = vl

        section_header(self, "Spring calc")
        self._spring = tk.Label(self, text="—", bg=BG2, fg=YELLOW,
                                font=("Consolas",9), justify="left")
        self._spring.pack(fill="x", padx=10)

    def update(self, r: SweepResult, freq_hz=2.5, sprung_lb=250.0):
        for key, lbl in self._lbls.items():
            val = getattr(r, key)
            rng = TARGETS.get(key, (None,"",""))[0]
            if rng and not (rng[0] <= val <= rng[1]):
                color = YELLOW if abs(val - np.mean(rng)) < (rng[1]-rng[0]) else RED
            else:
                color = GREEN
            fmt = f"{val:+.2f}" if abs(val)<100 else f"{val:.1f}"
            lbl.config(text=fmt, fg=color)

        mr = r.s_motion_ratio
        sp = spring_for_frequency(freq_hz, sprung_lb, mr)
        wr = wheel_rate(sp, mr)
        hz = ride_frequency(wr, sprung_lb)
        self._spring.config(text=f"Spring: {sp:.0f} lbf/in\n"
                                  f"Wheel rate: {wr:.0f} lbf/in\n"
                                  f"Ride freq: {hz:.2f} Hz")


# ── 3D Visualizer ─────────────────────────────────────────────────────────────
class Visualizer3D(tk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, bg=BG, **kw)
        self.fig3d = Figure(figsize=(5,4), facecolor="#0d0d1a")
        self.ax3d  = self.fig3d.add_subplot(111, projection="3d")
        self.ax3d.set_facecolor("#0d0d1a")
        self.canvas3d = FigureCanvasTkAgg(self.fig3d, master=self)
        self.canvas3d.get_tk_widget().pack(fill="both", expand=True)
        self._anim = None
        self._frames = []
        self._frame_idx = 0

        ctrl = tk.Frame(self, bg=BG2); ctrl.pack(fill="x", padx=4, pady=4)
        self._play_btn = tk.Button(ctrl, text="▶ Play", bg=ACCENT, fg=BG,
                                   font=("Consolas",9,"bold"), relief="flat",
                                   padx=8, pady=3, command=self._toggle_play,
                                   activebackground="#9d8fff")
        self._play_btn.pack(side="left", padx=4)
        tk.Label(ctrl, text="Travel:", bg=BG2, fg=DIM,
                 font=("Consolas",9)).pack(side="left", padx=(8,2))
        self._travel_lbl = tk.Label(ctrl, text="0.0 in", bg=BG2, fg=GREEN,
                                    font=("Consolas",9,"bold"))
        self._travel_lbl.pack(side="left")
        self._playing = False
        self._after_id = None

    def update(self, result: SweepResult):
        self._frames = result.frames
        self._travel = result.travel
        self._frame_idx = len(result.frames)//2
        self._draw_frame(self._frame_idx)

    def _draw_frame(self, idx):
        if not self._frames: return
        m = self._frames[idx]
        ax = self.ax3d
        ax.cla()
        ax.set_facecolor("#0d0d1a")
        ax.set_xlabel("X (fwd)", color=DIM, fontsize=7, labelpad=1)
        ax.set_ylabel("Y (out)", color=DIM, fontsize=7, labelpad=1)
        ax.set_zlabel("Z (up)",  color=DIM, fontsize=7, labelpad=1)
        ax.tick_params(colors=DIM, labelsize=6)
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(BORDER)
        ax.yaxis.pane.set_edgecolor(BORDER)
        ax.zaxis.pane.set_edgecolor(BORDER)
        ax.grid(True, color=BORDER, alpha=0.4, linewidth=0.5)

        def line(p1, p2, color, lw=1.5, alpha=1.0):
            ax.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]],
                    color=color, linewidth=lw, alpha=alpha)

        def dot(p, color, size=30, marker="o"):
            ax.scatter(*p, color=color, s=size, zorder=5, depthshade=False)

        # UCA
        line(m.uca_inboard_front, m.uca_outboard, "#7c6af7", 2)
        line(m.uca_inboard_rear,  m.uca_outboard, "#7c6af7", 2)
        line(m.uca_inboard_front, m.uca_inboard_rear, "#7c6af7", 1, 0.5)
        dot(m.uca_outboard, "#b0a8ff", 50, "^")

        # LCA
        line(m.lca_inboard_front, m.lca_outboard, "#00d4aa", 2)
        line(m.lca_inboard_rear,  m.lca_outboard, "#00d4aa", 2)
        line(m.lca_inboard_front, m.lca_inboard_rear, "#00d4aa", 1, 0.5)
        dot(m.lca_outboard, "#00ffcc", 50, "v")

        # Kingpin axis
        line(m.uca_outboard, m.lca_outboard, "#ffd700", 1.5)

        # Tie rod
        line(m.tie_rod_inboard, m.tie_rod_outboard, "#ff9f43", 2)
        dot(m.tie_rod_inboard,  "#ffb366", 30)
        dot(m.tie_rod_outboard, "#ffb366", 30)

        # Shock
        line(m.shock_upper, m.shock_lower, "#ff6b6b", 2)
        dot(m.shock_upper, "#ff9999", 40, "s")
        dot(m.shock_lower, "#ff9999", 30, "s")

        # Inboard chassis dots
        for p in [m.uca_inboard_front, m.uca_inboard_rear,
                  m.lca_inboard_front, m.lca_inboard_rear]:
            dot(p, "#aaaaaa", 20)

        # Wheel center origin
        dot(np.zeros(3), "#ffffff", 40, "*")

        # Wheel circle (approximate)
        theta = np.linspace(0, 2*np.pi, 40)
        wy = np.zeros_like(theta)
        wz = TIRE_RADIUS_IN * np.sin(theta)
        wx = TIRE_RADIUS_IN * np.cos(theta)
        ax.plot(wx, wy, wz, color="#444466", linewidth=0.8, alpha=0.6)

        # Mirror ground plane line
        ax.plot([-6,6],[0,0],[-TIRE_RADIUS_IN,-TIRE_RADIUS_IN],
                color=BORDER, linewidth=0.5, alpha=0.5)

        ax.set_xlim(-6, 6); ax.set_ylim(-12, 2); ax.set_zlim(-13, 10)
        title_text = f"Travel: {self._travel[idx]:+.2f} in"
        ax.set_title(title_text, color=TEXT, fontsize=8, pad=2)

        self.canvas3d.draw_idle()
        self._travel_lbl.config(text=f"{self._travel[idx]:+.2f} in")

    def _toggle_play(self):
        if self._playing:
            self._playing = False
            self._play_btn.config(text="▶ Play")
            if self._after_id:
                self.after_cancel(self._after_id)
        else:
            self._playing = True
            self._play_btn.config(text="⏹ Stop")
            self._animate()

    def _animate(self):
        if not self._playing or not self._frames: return
        self._frame_idx = (self._frame_idx + 1) % len(self._frames)
        self._draw_frame(self._frame_idx)
        self._after_id = self.after(80, self._animate)


# ── Kinematic charts panel ────────────────────────────────────────────────────
class ChartPanel(tk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, bg=BG, **kw)
        self.fig = Figure(figsize=(6,4), facecolor="#0d0d1a")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def update(self, r: SweepResult, goals):
        self.fig.clear()
        t = r.travel
        plots = [
            ("Camber (deg)",         r.camber,       "#7c6af7", None),
            ("Toe / Bump steer (deg)",r.toe,          "#00d4aa", None),
            ("Roll center ht (in)",  r.rc_height,    "#ffd700",
             np.full_like(t, goals.roll_center_target)),
            ("Motion ratio",         r.motion_ratio, "#ff6b6b",
             np.full_like(t, goals.motion_ratio_target)),
            ("Caster (deg)",         r.caster,       "#ff9f43", None),
            ("Track change (in)",    r.track_change, "#888888", None),
        ]
        for i, (title, y, color, tgt) in enumerate(plots):
            ax = self.fig.add_subplot(2, 3, i+1)
            ax.set_facecolor("#0d0d1a")
            ax.tick_params(colors=DIM, labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
            ax.set_title(title, color=DIM, fontsize=7, pad=2)
            ax.set_xlabel("Travel (in)", color=DIM, fontsize=6)
            ax.axhline(0, color=BORDER, linewidth=0.4)
            ax.axvline(0, color=BORDER, linewidth=0.4, linestyle="--")
            ax.plot(t, y, color=color, linewidth=1.5)
            if tgt is not None:
                ax.plot(t, tgt, color="#444466", linewidth=1, linestyle="--")
        self.fig.tight_layout(pad=1.2)
        self.canvas.draw_idle()


# ── Visualizer tab ────────────────────────────────────────────────────────────
class VisualizerTab(tk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, bg=BG, **kw)
        self._result = None
        self._hp = None
        self._build()

    def _build(self):
        # Left: hardpoint entry + stats
        left = tk.Frame(self, bg=BG2, width=360)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        self.hp_panel = HardpointPanel(left, on_change_cb=self._on_change)
        self.hp_panel.pack(fill="both", expand=True)

        # Right: 3D view on top, charts below
        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        # Top strip: settings + stats + compute button
        top_strip = tk.Frame(right, bg=BG2)
        top_strip.pack(fill="x")

        settings_frame = tk.Frame(top_strip, bg=BG2)
        settings_frame.pack(side="left", padx=8, pady=6)
        section_header(settings_frame, "Sweep")
        self._bump  = self._setting_row(settings_frame, "Bump (in)",   "5.0")
        self._droop = self._setting_row(settings_frame, "Droop (in)",  "5.0")
        self._steps = self._setting_row(settings_frame, "Steps",       "41")
        section_header(settings_frame, "Spring calc")
        self._freq  = self._setting_row(settings_frame, "Ride freq (Hz)", "2.5")
        self._mass  = self._setting_row(settings_frame, "Sprung mass (lb)","250")

        self.stats = StatsPanel(top_strip)
        self.stats.pack(side="left", fill="y", padx=8, pady=6)

        tk.Button(top_strip, text="▶ Compute", bg=ACCENT, fg=BG,
                  font=("Consolas",10,"bold"), relief="flat",
                  padx=10, pady=6, command=self._compute,
                  activebackground="#9d8fff"
                  ).pack(side="left", padx=10, anchor="n", pady=12)

        # 3D and charts side by side
        viz_row = tk.Frame(right, bg=BG)
        viz_row.pack(fill="both", expand=True)

        self.vis3d = Visualizer3D(viz_row)
        self.vis3d.pack(side="left", fill="both", expand=True)

        self.charts = ChartPanel(viz_row)
        self.charts.pack(side="left", fill="both", expand=True)

    def _setting_row(self, parent, label, default):
        f = tk.Frame(parent, bg=BG2); f.pack(fill="x", padx=8, pady=1)
        tk.Label(f, text=label, bg=BG2, fg=TEXT,
                 font=("Consolas",9), width=18, anchor="w").pack(side="left")
        sv = tk.StringVar(value=default)
        sv.trace_add("write", lambda *a: None)
        entry_sv(f, sv, width=6).pack(side="left")
        return sv

    def _on_change(self):
        if hasattr(self,"_after_id"): self.after_cancel(self._after_id)
        self._after_id = self.after(450, self._compute)

    def _f(self, sv, d):
        try: return float(sv.get())
        except: return d

    def _compute(self):
        hp = self.hp_panel.get_hp()
        try:
            result = run_sweep(hp,
                               bump_in  = self._f(self._bump,  5.0),
                               droop_in = self._f(self._droop, 5.0),
                               steps    = max(3,int(self._f(self._steps,41))))
            self._result = result
            self._hp = hp

            goals = Goals()  # default goals for chart reference lines
            self.stats.update(result, self._f(self._freq,2.5), self._f(self._mass,250))
            self.vis3d.update(result)
            self.charts.update(result, goals)
        except Exception as e:
            messagebox.showerror("Solver error", str(e))

    def set_hp(self, hp: Hardpoints):
        self.hp_panel.set_hp(hp)
        self._compute()

    def get_result(self): return self._result
    def get_hp(self):     return self._hp


# ── Optimizer tab ─────────────────────────────────────────────────────────────
class OptimizerTab(tk.Frame):
    def __init__(self, master, apply_cb, **kw):
        super().__init__(master, bg=BG2, **kw)
        self.apply_cb = apply_cb
        self._running = False
        self._last = None
        self._knowns_vars = {}
        self._goals_vars  = {}
        self._bounds_vars = {}
        self._build()

    def _build(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=4, pady=4)

        style = ttk.Style()
        style.configure("TNotebook.Tab", font=("Consolas",9),
                        background=BG2, foreground=DIM)
        style.map("TNotebook.Tab",
                  background=[("selected",BG)],
                  foreground=[("selected",ACCENT)])

        # Sub-tab 1: Knowns
        kt = tk.Frame(nb, bg=BG2); nb.add(kt, text=" Knowns (constraints) ")
        self._build_knowns(kt)

        # Sub-tab 2: Goals
        gt = tk.Frame(nb, bg=BG2); nb.add(gt, text=" Goals (targets) ")
        self._build_goals(gt)

        # Sub-tab 3: Search bounds
        bt = tk.Frame(nb, bg=BG2); nb.add(bt, text=" Search bounds ")
        self._build_bounds(bt)

        # Sub-tab 4: Results
        rt = tk.Frame(nb, bg=BG2); nb.add(rt, text=" Results ")
        self._build_results(rt)

        self._results_nb = nb

    # ── Knowns ──────────────────────────────────────────────────────────────
    def _build_knowns(self, parent):
        canvas = tk.Canvas(parent, bg=BG2, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True)
        inner = tk.Frame(canvas, bg=BG2)
        canvas.create_window((0,0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        k = Knowns()
        groups = [
            ("Chassis dimensions", [
                ("chassis_half_width",  "Half chassis width (in)",  k.chassis_half_width),
                ("wheelbase",           "Wheelbase (in)",           k.wheelbase),
                ("ride_height",         "Ride height (in)",         k.ride_height),
            ]),
            ("UCA inboard Z limits", [
                ("uca_inboard_z_min",  "UCA inboard Z min (in)",  k.uca_inboard_z_min),
                ("uca_inboard_z_max",  "UCA inboard Z max (in)",  k.uca_inboard_z_max),
            ]),
            ("LCA inboard Z limits", [
                ("lca_inboard_z_min",  "LCA inboard Z min (in)",  k.lca_inboard_z_min),
                ("lca_inboard_z_max",  "LCA inboard Z max (in)",  k.lca_inboard_z_max),
            ]),
            ("Inboard Y limits", [
                ("uca_inboard_y_min",  "UCA inboard Y min (in)",  k.uca_inboard_y_min),
                ("uca_inboard_y_max",  "UCA inboard Y max (in)",  k.uca_inboard_y_max),
                ("lca_inboard_y_min",  "LCA inboard Y min (in)",  k.lca_inboard_y_min),
                ("lca_inboard_y_max",  "LCA inboard Y max (in)",  k.lca_inboard_y_max),
            ]),
            ("Ball joint Z limits", [
                ("ubj_z_min",  "UBJ Z min (in)",  k.ubj_z_min),
                ("ubj_z_max",  "UBJ Z max (in)",  k.ubj_z_max),
                ("lbj_z_min",  "LBJ Z min (in)",  k.lbj_z_min),
                ("lbj_z_max",  "LBJ Z max (in)",  k.lbj_z_max),
            ]),
            ("Shock", [
                ("shock_stroke_min",  "Shock min length (in)",  k.shock_stroke_min),
                ("shock_stroke_max",  "Shock max length (in)",  k.shock_stroke_max),
            ]),
            ("Travel requirements", [
                ("bump_in",   "Required bump (in)",  k.bump_in),
                ("droop_in",  "Required droop (in)", k.droop_in),
            ]),
        ]

        row = 0
        for group_name, fields in groups:
            tk.Label(inner, text=group_name, bg=BG2, fg=ACCENT,
                     font=("Consolas",9,"bold")).grid(
                     row=row, column=0, columnspan=2,
                     sticky="w", padx=8, pady=(8,2))
            row += 1
            for key, label, default in fields:
                tk.Label(inner, text=label, bg=BG2, fg=TEXT,
                         font=("Consolas",9), width=30, anchor="w",
                         padx=12).grid(row=row, column=0, sticky="w")
                sv = tk.StringVar(value=str(default))
                entry_sv(inner, sv, width=8).grid(row=row, column=1, padx=4, pady=2)
                self._knowns_vars[key] = sv
                row += 1

        # Boolean constraints
        tk.Label(inner, text="Hard rules", bg=BG2, fg=ACCENT,
                 font=("Consolas",9,"bold")).grid(
                 row=row, column=0, columnspan=2, sticky="w", padx=8, pady=(8,2))
        row += 1
        for key, label, default in [
            ("never_negative_caster", "Never negative caster", True),
            ("never_negative_trail",  "Never negative mech trail", True),
        ]:
            bv = tk.BooleanVar(value=default)
            tk.Checkbutton(inner, text=label, variable=bv,
                           bg=BG2, fg=TEXT, selectcolor=BG,
                           activebackground=BG2, font=("Consolas",9)
                           ).grid(row=row, column=0, columnspan=2,
                                  sticky="w", padx=12, pady=2)
            self._knowns_vars[key] = bv
            row += 1

    # ── Goals ────────────────────────────────────────────────────────────────
    def _build_goals(self, parent):
        canvas = tk.Canvas(parent, bg=BG2, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True)
        inner = tk.Frame(canvas, bg=BG2)
        canvas.create_window((0,0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        tk.Label(inner, text="Goal", bg=BG2, fg=DIM,
                 font=("Consolas",8), width=22, anchor="w").grid(row=0, column=0, padx=8)
        tk.Label(inner, text="Target", bg=BG2, fg=DIM,
                 font=("Consolas",8), width=10).grid(row=0, column=1)
        tk.Label(inner, text="Weight (0-100)", bg=BG2, fg=DIM,
                 font=("Consolas",8), width=14).grid(row=0, column=2)
        tk.Label(inner, text="Teddy range", bg=BG2, fg=DIM,
                 font=("Consolas",8), width=16, anchor="w").grid(row=0, column=3)

        g = Goals()
        goal_fields = [
            ("bump_steer",    "Bump steer (deg/in)",      g.bump_steer_target,    g.bump_steer_weight,    "< 0.025"),
            ("roll_center",   "Roll center ht (in)",      g.roll_center_target,   g.roll_center_weight,   "2-4 in"),
            ("camber_gain",   "Camber gain (deg/in)",     g.camber_gain_target,   g.camber_gain_weight,   "-0.05 to -0.10"),
            ("kpi",           "KPI (deg)",                g.kpi_target,           g.kpi_weight,           "7-12 deg"),
            ("scrub",         "Scrub radius (in)",        g.scrub_target,         g.scrub_weight,         "0.1-0.6 in"),
            ("motion_ratio",  "Motion ratio",             g.motion_ratio_target,  g.motion_ratio_weight,  "0.55-0.75"),
            ("ackermann",     "Ackermann (%)",            g.ackermann_target,     g.ackermann_weight,     "80-110%"),
            ("swing_arm",     "Swing arm (in)",           g.swing_arm_target,     g.swing_arm_weight,     "50-100 in"),
            ("caster",        "Caster (deg)",             g.caster_target,        g.caster_weight,        "4-8 deg"),
            ("mech_trail",    "Mech trail (in)",          g.mech_trail_target,    g.mech_trail_weight,    "> 0.5"),
            ("static_camber", "Static camber (deg)",      g.static_camber_target, g.static_camber_weight, "-1 to -2 deg"),
        ]

        for ri, (key, label, tgt, wt, rng) in enumerate(goal_fields):
            bg = BG if ri%2==0 else BG2
            tk.Label(inner, text=label, bg=bg, fg=TEXT,
                     font=("Consolas",9), width=24, anchor="w",
                     padx=8).grid(row=ri+1, column=0, sticky="w", pady=2)
            sv_tgt = tk.StringVar(value=str(tgt))
            sv_wt  = tk.StringVar(value=str(wt))
            tk.Entry(inner, textvariable=sv_tgt, bg=bg, fg=TEXT,
                     insertbackground=TEXT, font=("Consolas",9), width=8,
                     bd=0, highlightthickness=1,
                     highlightbackground=BORDER, highlightcolor=ACCENT
                     ).grid(row=ri+1, column=1, padx=4, pady=2)
            tk.Entry(inner, textvariable=sv_wt, bg=bg, fg=TEXT,
                     insertbackground=TEXT, font=("Consolas",9), width=8,
                     bd=0, highlightthickness=1,
                     highlightbackground=BORDER, highlightcolor=ACCENT
                     ).grid(row=ri+1, column=2, padx=4, pady=2)
            tk.Label(inner, text=rng, bg=bg, fg=DIM,
                     font=("Consolas",8)).grid(row=ri+1, column=3,
                     sticky="w", padx=8)
            self._goals_vars[key] = (sv_tgt, sv_wt)

    # ── Bounds ───────────────────────────────────────────────────────────────
    def _build_bounds(self, parent):
        tk.Label(parent,
                 text="Set search range for each point. Lock = keep fixed at current value.",
                 bg=BG2, fg=DIM, font=("Consolas",8)
                 ).pack(fill="x", padx=8, pady=(8,2))

        hdr = tk.Frame(parent, bg=BG2); hdr.pack(fill="x", padx=8)
        for col,(t,w) in enumerate([("Point",20),("X min",6),("X max",6),("Xlk",4),
                                     ("Y min",6),("Y max",6),("Ylk",4),
                                     ("Z min",6),("Z max",6),("Zlk",4)]):
            tk.Label(hdr, text=t, bg=BG2, fg=DIM,
                     font=("Consolas",8), width=w, anchor="w"
                     ).grid(row=0, column=col, padx=1)

        canvas = tk.Canvas(parent, bg=BG2, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True)
        inner = tk.Frame(canvas, bg=BG2)
        canvas.create_window((0,0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        hp = Hardpoints()
        R = 2.0
        for ri, name in enumerate(POINT_NAMES):
            val = getattr(hp, name)
            bg = BG if ri%2==0 else BG2
            rv = {}
            tk.Label(inner, text=POINT_LABELS[name], bg=bg, fg=TEXT,
                     font=("Consolas",9), width=20, anchor="w",
                     padx=4).grid(row=ri, column=0, sticky="w")
            col = 1
            for ai, ax in enumerate(["x","y","z"]):
                v = val[ai]
                sv_lo = tk.StringVar(value=f"{v-R:.2f}")
                sv_hi = tk.StringVar(value=f"{v+R:.2f}")
                sv_lk = tk.BooleanVar(value=False)
                tk.Entry(inner, textvariable=sv_lo, bg=bg, fg=TEXT,
                         insertbackground=TEXT, font=("Consolas",8), width=6,
                         bd=0, highlightthickness=1,
                         highlightbackground=BORDER, highlightcolor=ACCENT
                         ).grid(row=ri, column=col,   padx=1, pady=2)
                tk.Entry(inner, textvariable=sv_hi, bg=bg, fg=TEXT,
                         insertbackground=TEXT, font=("Consolas",8), width=6,
                         bd=0, highlightthickness=1,
                         highlightbackground=BORDER, highlightcolor=ACCENT
                         ).grid(row=ri, column=col+1, padx=1, pady=2)
                tk.Checkbutton(inner, variable=sv_lk, bg=bg,
                               fg=DIM, selectcolor=BG,
                               activebackground=bg
                               ).grid(row=ri, column=col+2)
                rv[f"{ax}_min"] = sv_lo
                rv[f"{ax}_max"] = sv_hi
                rv[f"{ax}_lock"] = sv_lk
                col += 3
            self._bounds_vars[name] = rv

    # ── Results ──────────────────────────────────────────────────────────────
    def _build_results(self, parent):
        # Run controls
        ctrl = tk.Frame(parent, bg=BG2); ctrl.pack(fill="x", padx=10, pady=8)

        run_settings = tk.Frame(ctrl, bg=BG2); run_settings.pack(side="left")
        self._max_iter_sv = tk.StringVar(value="300")
        self._popsize_sv  = tk.StringVar(value="10")
        for lbl_txt, sv, w in [("Max iter","_max_iter_sv",5), ("Population","_popsize_sv",4)]:
            f = tk.Frame(run_settings, bg=BG2); f.pack(side="left", padx=4)
            tk.Label(f, text=lbl_txt, bg=BG2, fg=DIM,
                     font=("Consolas",9)).pack(side="left", padx=(0,4))
            entry_sv(f, getattr(self, sv), width=w).pack(side="left")

        self._run_btn = tk.Button(ctrl, text="▶  Run optimizer",
                                  bg=ACCENT, fg=BG,
                                  font=("Consolas",11,"bold"), relief="flat",
                                  padx=14, pady=7, command=self._run,
                                  activebackground="#9d8fff")
        self._run_btn.pack(side="left", padx=12)

        self._apply_btn = tk.Button(ctrl, text="✓ Apply to Visualizer",
                                    bg=BG, fg=GREEN,
                                    font=("Consolas",10,"bold"), relief="flat",
                                    padx=10, pady=7, command=self._apply,
                                    state="disabled",
                                    activebackground=BG2)
        self._apply_btn.pack(side="left", padx=4)

        self._prog = ttk.Progressbar(ctrl, mode="indeterminate", length=150)
        self._prog.pack(side="left", padx=10)

        self._status_lbl = tk.Label(ctrl, text="Ready", bg=BG2, fg=DIM,
                                    font=("Consolas",9))
        self._status_lbl.pack(side="left", padx=8)

        # Score display
        score_frame = tk.Frame(parent, bg=BG2); score_frame.pack(fill="x", padx=10)
        self._score_lbl = tk.Label(score_frame, text="—", bg=BG2, fg=GREEN,
                                   font=("Consolas",18,"bold"))
        self._score_lbl.pack(side="left", padx=8)
        tk.Label(score_frame, text="/ 100", bg=BG2, fg=DIM,
                 font=("Consolas",12)).pack(side="left")
        self._conv_lbl = tk.Label(score_frame, text="", bg=BG2, fg=DIM,
                                  font=("Consolas",9))
        self._conv_lbl.pack(side="left", padx=20)

        # Violations
        self._viol_lbl = tk.Label(parent, text="", bg=BG2, fg=RED,
                                  font=("Consolas",9), justify="left")
        self._viol_lbl.pack(fill="x", padx=12)

        # Per-goal breakdown table
        tk.Label(parent, text="Goal breakdown", bg=BG2, fg=ACCENT,
                 font=("Consolas",10,"bold")).pack(fill="x", padx=10, pady=(8,2))
        self._breakdown_frame = tk.Frame(parent, bg=BG2)
        self._breakdown_frame.pack(fill="x", padx=10)

        # Hardpoint results
        tk.Label(parent, text="Resulting hardpoints (inches)", bg=BG2, fg=ACCENT,
                 font=("Consolas",10,"bold")).pack(fill="x", padx=10, pady=(8,2))
        self._hp_text = tk.Text(parent, bg=BG, fg=GREEN,
                                font=("Consolas",9), height=12,
                                bd=0, highlightthickness=0)
        self._hp_text.pack(fill="x", padx=10, pady=(0,8))

    # ── Get inputs ───────────────────────────────────────────────────────────
    def _get_knowns(self) -> Knowns:
        k = Knowns()
        def f(key, default):
            v = self._knowns_vars.get(key)
            if v is None: return default
            if isinstance(v, tk.BooleanVar): return v.get()
            try: return float(v.get())
            except: return default
        for field_name in Knowns.__dataclass_fields__:
            default = getattr(k, field_name)
            setattr(k, field_name, f(field_name, default))
        return k

    def _get_goals(self) -> Goals:
        g = Goals()
        def f(sv, default):
            try: return float(sv.get())
            except: return default
        mapping = {
            "bump_steer":   ("bump_steer_target",    "bump_steer_weight"),
            "roll_center":  ("roll_center_target",   "roll_center_weight"),
            "camber_gain":  ("camber_gain_target",   "camber_gain_weight"),
            "kpi":          ("kpi_target",           "kpi_weight"),
            "scrub":        ("scrub_target",         "scrub_weight"),
            "motion_ratio": ("motion_ratio_target",  "motion_ratio_weight"),
            "ackermann":    ("ackermann_target",     "ackermann_weight"),
            "swing_arm":    ("swing_arm_target",     "swing_arm_weight"),
            "caster":       ("caster_target",        "caster_weight"),
            "mech_trail":   ("mech_trail_target",    "mech_trail_weight"),
            "static_camber":("static_camber_target", "static_camber_weight"),
        }
        for key, (tgt_field, wt_field) in mapping.items():
            sv_tgt, sv_wt = self._goals_vars[key]
            setattr(g, tgt_field, f(sv_tgt, getattr(g, tgt_field)))
            setattr(g, wt_field,  f(sv_wt,  getattr(g, wt_field)))
        return g

    def _get_bounds(self) -> dict:
        hp = Hardpoints()
        bounds = {}
        for name in POINT_NAMES:
            rv = self._bounds_vars.get(name, {})
            val = getattr(hp, name)
            def fv(sv, d):
                try: return float(sv.get())
                except: return d
            bounds[name] = PointBounds(
                x_min=fv(rv.get("x_min", tk.StringVar(value=str(val[0]-2))),val[0]-2),
                x_max=fv(rv.get("x_max", tk.StringVar(value=str(val[0]+2))),val[0]+2),
                x_locked=rv.get("x_lock", tk.BooleanVar()).get() if "x_lock" in rv else False,
                y_min=fv(rv.get("y_min", tk.StringVar(value=str(val[1]-2))),val[1]-2),
                y_max=fv(rv.get("y_max", tk.StringVar(value=str(val[1]+2))),val[1]+2),
                y_locked=rv.get("y_lock", tk.BooleanVar()).get() if "y_lock" in rv else False,
                z_min=fv(rv.get("z_min", tk.StringVar(value=str(val[2]-2))),val[2]-2),
                z_max=fv(rv.get("z_max", tk.StringVar(value=str(val[2]+2))),val[2]+2),
                z_locked=rv.get("z_lock", tk.BooleanVar()).get() if "z_lock" in rv else False,
            )
        return bounds

    # ── Run ──────────────────────────────────────────────────────────────────
    def _run(self):
        if self._running: return
        self._running = True
        self._run_btn.config(state="disabled")
        self._apply_btn.config(state="disabled")
        self._status_lbl.config(text="Running...", fg=YELLOW)
        self._prog.start(12)
        self._score_lbl.config(text="—")

        knowns = self._get_knowns()
        goals  = self._get_goals()
        bounds = self._get_bounds()
        hp     = Hardpoints()

        try: max_iter = int(self._max_iter_sv.get())
        except: max_iter = 300
        try: popsize = int(self._popsize_sv.get())
        except: popsize = 10

        def worker():
            def prog(gen, sc):
                self.after(0, lambda g=gen, s=sc:
                    self._status_lbl.config(
                        text=f"Gen {g}  best: {s:.4f}", fg=YELLOW))
            try:
                result = run_optimizer(hp, bounds, knowns, goals,
                                       max_iter=max_iter, popsize=popsize,
                                       progress_cb=prog)
                self.after(0, lambda r=result: self._done(r))
            except Exception as e:
                self.after(0, lambda err=e: self._error(err))

        threading.Thread(target=worker, daemon=True).start()

    def _done(self, r: OptResult):
        self._running = False
        self._prog.stop()
        self._run_btn.config(state="normal")
        self._apply_btn.config(state="normal")
        self._last = r

        raw_score = r.final_score
        display_score = max(0, 100 - raw_score * 20)
        color = GREEN if display_score > 75 else YELLOW if display_score > 50 else RED
        self._score_lbl.config(text=f"{display_score:.0f}", fg=color)
        conv_txt = "✓ converged" if r.converged else "~ not fully converged"
        self._conv_lbl.config(
            text=f"{conv_txt}  |  {r.n_evals} evaluations", fg=DIM)
        self._status_lbl.config(text="Done", fg=GREEN)

        # Violations
        if r.violations:
            self._viol_lbl.config(
                text="⚠ Constraint violations: " + ", ".join(r.violations), fg=RED)
        else:
            self._viol_lbl.config(text="✓ All constraints satisfied", fg=GREEN)

        # Breakdown table
        for w in self._breakdown_frame.winfo_children():
            w.destroy()
        tk.Label(self._breakdown_frame, text="Goal", bg=BG2, fg=DIM,
                 font=("Consolas",8), width=18, anchor="w").grid(row=0,column=0,padx=4)
        tk.Label(self._breakdown_frame, text="Got", bg=BG2, fg=DIM,
                 font=("Consolas",8), width=10).grid(row=0,column=1)
        tk.Label(self._breakdown_frame, text="Target", bg=BG2, fg=DIM,
                 font=("Consolas",8), width=10).grid(row=0,column=2)
        tk.Label(self._breakdown_frame, text="Unit", bg=BG2, fg=DIM,
                 font=("Consolas",8), width=6).grid(row=0,column=3)

        for ri, (goal_name, (got, tgt, unit)) in enumerate(r.score_breakdown.items()):
            close = abs(got-tgt) < abs(tgt)*0.2 + 0.1
            fg = GREEN if close else YELLOW
            bg = BG if ri%2==0 else BG2
            tk.Label(self._breakdown_frame, text=goal_name, bg=bg, fg=TEXT,
                     font=("Consolas",9), width=18, anchor="w",
                     padx=4).grid(row=ri+1, column=0, sticky="w", pady=1)
            tk.Label(self._breakdown_frame, text=f"{got:+.3f}", bg=bg, fg=fg,
                     font=("Consolas",9,"bold"), width=10).grid(row=ri+1, column=1)
            tk.Label(self._breakdown_frame, text=f"{tgt:+.3f}", bg=bg, fg=DIM,
                     font=("Consolas",9), width=10).grid(row=ri+1, column=2)
            tk.Label(self._breakdown_frame, text=unit, bg=bg, fg=DIM,
                     font=("Consolas",8), width=6).grid(row=ri+1, column=3)

        # Hardpoint results
        self._hp_text.config(state="normal")
        self._hp_text.delete("1.0","end")
        lines = [f"{'Point':<22} {'X':>8}  {'Y':>8}  {'Z':>8}"]
        lines.append("-"*50)
        for name, val in r.hardpoints.__dict__.items():
            lines.append(f"{POINT_LABELS[name]:<22} {val[0]:>8.3f}  {val[1]:>8.3f}  {val[2]:>8.3f}")
        self._hp_text.insert("1.0", "\n".join(lines))
        self._hp_text.config(state="disabled")

        # Switch to results tab
        self._results_nb.select(3)

    def _error(self, err):
        self._running = False
        self._prog.stop()
        self._run_btn.config(state="normal")
        self._status_lbl.config(text=f"Error: {err}", fg=RED)

    def _apply(self):
        if self._last:
            self.apply_cb(self._last.hardpoints)
            self._status_lbl.config(text="Applied to Visualizer ✓", fg=GREEN)


# ── Main application ──────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Suspension Design Tool — Sooner Offroad Baja SAE 2025-26 | Teddy")
        self.configure(bg=BG)
        self.geometry("1440x880")
        self.minsize(1100, 700)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame",     background=BG2)
        style.configure("TScrollbar", background=BG2, troughcolor=BG, arrowcolor=DIM)
        style.configure("TNotebook",  background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG2, foreground=DIM,
                        font=("Consolas",11), padding=(14,6))
        style.map("TNotebook.Tab",
                  background=[("selected",BG)],
                  foreground=[("selected",ACCENT)])

        self._build_menu()
        self._build_tabs()
        self.after(300, self.vis_tab._compute)

    def _build_menu(self):
        m = tk.Menu(self, bg=BG2, fg=TEXT,
                    activebackground=ACCENT, activeforeground=BG)
        fm = tk.Menu(m, tearoff=0, bg=BG2, fg=TEXT,
                     activebackground=ACCENT, activeforeground=BG)
        fm.add_command(label="Export hardpoints CSV", command=self._export_hp)
        fm.add_command(label="Export kinematics CSV", command=self._export_kin)
        fm.add_separator()
        fm.add_command(label="Reset hardpoints",      command=self._reset)
        fm.add_separator()
        fm.add_command(label="Quit", command=self.destroy)
        m.add_cascade(label="File", menu=fm)
        self.config(menu=m)

    def _build_tabs(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=0, pady=0)

        self.vis_tab = VisualizerTab(nb)
        nb.add(self.vis_tab, text="  Visualizer  ")

        self.opt_tab = OptimizerTab(nb, apply_cb=self._apply_opt)
        nb.add(self.opt_tab, text="  Optimizer  ")

    def _apply_opt(self, hp: Hardpoints):
        self.vis_tab.set_hp(hp)

    def _export_hp(self):
        hp = self.vis_tab.get_hp()
        if not hp:
            messagebox.showinfo("Nothing to export","Compute first."); return
        p = filedialog.asksaveasfilename(defaultextension=".csv",
            filetypes=[("CSV","*.csv")], initialfile="teddy_hardpoints.csv")
        if p:
            export_hardpoints_csv(hp, p)
            messagebox.showinfo("Saved", p)

    def _export_kin(self):
        r = self.vis_tab.get_result()
        if not r:
            messagebox.showinfo("Nothing to export","Compute first."); return
        p = filedialog.asksaveasfilename(defaultextension=".csv",
            filetypes=[("CSV","*.csv")], initialfile="teddy_kinematics.csv")
        if p:
            export_kinematics_csv(r, p)
            messagebox.showinfo("Saved", p)

    def _reset(self):
        self.vis_tab.set_hp(Hardpoints())


if __name__ == "__main__":
    App().mainloop()