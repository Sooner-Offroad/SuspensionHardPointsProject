"""
Suspension Hardpoint Designer — Double Wishbone (Front)
Baja SAE / Offroad

Tabs:
  1. Hardpoints  — manual entry + live kinematic plots
  2. Optimizer   — set ranges, weights, run solver, apply result

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

from solver import (
    Hardpoints, KinematicTargets, KinematicResult,
    run_sweep, export_hardpoints_csv, export_kinematics_csv,
)
from optimizer import (
    PointBounds, OptimizerWeights, make_default_bounds,
    run_optimizer, OptimizeResult, POINT_NAMES,
)

# ── colors ──────────────────────────────────────────────────────────────────
BG      = "#1e1e2e"
BG2     = "#2a2a3e"
ACCENT  = "#7c6af7"
TEXT    = "#cdd6f4"
DIMTEXT = "#6c7086"
GREEN   = "#a6e3a1"
YELLOW  = "#f9e2af"
RED     = "#f38ba8"
BORDER  = "#313244"

POINT_LABELS = {
    "uca_inboard_front": "UCA Inboard Front",
    "uca_inboard_rear":  "UCA Inboard Rear",
    "uca_outboard":      "UCA Outboard (UBJ)",
    "lca_inboard_front": "LCA Inboard Front",
    "lca_inboard_rear":  "LCA Inboard Rear",
    "lca_outboard":      "LCA Outboard (LBJ)",
    "tie_rod_inboard":   "Tie Rod Inboard",
    "tie_rod_outboard":  "Tie Rod Outboard",
    "shock_upper":       "Shock Upper Mount",
    "shock_lower":       "Shock Lower Mount",
    "wheel_center":      "Wheel Center",
}


# ── helpers ──────────────────────────────────────────────────────────────────
def entry(parent, sv, width=8, bg=None):
    bg = bg or BG
    e = tk.Entry(parent, textvariable=sv, bg=bg, fg=TEXT,
                 insertbackground=TEXT, font=("Consolas", 9),
                 width=width, bd=0, highlightthickness=1,
                 highlightbackground=BORDER, highlightcolor=ACCENT)
    return e


def label(parent, text, fg=None, font=None, **kw):
    return tk.Label(parent, text=text, bg=BG2,
                    fg=fg or TEXT, font=font or ("Consolas", 9), **kw)


# ── Hardpoint entry panel ────────────────────────────────────────────────────
class HardpointPanel(tk.Frame):
    def __init__(self, master, on_change_cb, **kw):
        super().__init__(master, bg=BG2, **kw)
        self.on_change_cb = on_change_cb
        self.vars = {}
        self._build()

    def _build(self):
        label(self, "Hardpoints (mm)", fg=ACCENT,
              font=("Consolas", 12, "bold")).pack(fill="x", pady=(6,2), padx=8)

        hdr = tk.Frame(self, bg=BG2)
        hdr.pack(fill="x", padx=8)
        label(hdr, "Point", width=20, anchor="w").grid(row=0, column=0)
        for c, t in enumerate(["X","Y","Z"], 1):
            label(hdr, t, width=8).grid(row=0, column=c)

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
                        lambda e: canvas.yview_scroll(-(e.delta//120), "units"))

        hp = Hardpoints()
        for row_i, name in enumerate(POINT_NAMES):
            val = getattr(hp, name)
            bg = BG if row_i % 2 == 0 else BG2
            svars = []
            tk.Label(inner, text=POINT_LABELS[name], bg=bg, fg=TEXT,
                     font=("Consolas", 9), width=20, anchor="w",
                     padx=4).grid(row=row_i, column=0, sticky="w")
            for ci, v in enumerate(val):
                sv = tk.StringVar(value=f"{v:.1f}")
                sv.trace_add("write", lambda *a, n=name: self._on_edit(n))
                e = tk.Entry(inner, textvariable=sv, bg=bg, fg=TEXT,
                             insertbackground=TEXT, font=("Consolas",9),
                             width=8, bd=0, highlightthickness=1,
                             highlightbackground=BORDER, highlightcolor=ACCENT)
                e.grid(row=row_i, column=ci+1, padx=2, pady=2)
                svars.append(sv)
            self.vars[name] = svars

    def _on_edit(self, key):
        self.on_change_cb()

    def get_hardpoints(self) -> Hardpoints:
        hp = Hardpoints()
        for name, svars in self.vars.items():
            try:
                setattr(hp, name, np.array([float(s.get()) for s in svars]))
            except ValueError:
                pass
        return hp

    def set_hardpoints(self, hp: Hardpoints):
        for name, svars in self.vars.items():
            for sv, v in zip(svars, getattr(hp, name)):
                sv.set(f"{v:.1f}")


# ── Settings (sweep + targets) ───────────────────────────────────────────────
class SettingsPanel(tk.Frame):
    def __init__(self, master, on_change_cb, **kw):
        super().__init__(master, bg=BG2, **kw)
        self.on_change_cb = on_change_cb
        label(self, "Sweep Settings", fg=ACCENT,
              font=("Consolas",11,"bold")).pack(fill="x", padx=8, pady=(6,2))
        self.bump  = self._row("Bump travel (mm)",    "50")
        self.droop = self._row("Droop travel (mm)",   "50")
        self.steps = self._row("Sweep steps",         "41")
        self.track = self._row("Half-track (mm)",     "300")
        label(self, "Targets", fg=ACCENT,
              font=("Consolas",11,"bold")).pack(fill="x", padx=8, pady=(8,2))
        self.t_camber = self._row("Camber gain (deg/mm)", "-0.05")
        self.t_rc     = self._row("Roll center ht (mm)",  "60")
        self.t_bsteer = self._row("Bump steer (deg/mm)",  "0.002")
        self.t_mr     = self._row("Motion ratio",         "0.65")

    def _row(self, lbl, default):
        f = tk.Frame(self, bg=BG2); f.pack(fill="x", padx=8, pady=2)
        tk.Label(f, text=lbl, bg=BG2, fg=TEXT,
                 font=("Consolas",9), width=24, anchor="w").pack(side="left")
        sv = tk.StringVar(value=default)
        sv.trace_add("write", lambda *a: self.on_change_cb())
        tk.Entry(f, textvariable=sv, bg=BG, fg=TEXT,
                 insertbackground=TEXT, font=("Consolas",9),
                 width=8, bd=0, highlightthickness=1,
                 highlightbackground=BORDER, highlightcolor=ACCENT).pack(side="left")
        return sv

    def _f(self, sv, default):
        try: return float(sv.get())
        except: return default

    def get_bump(self)  -> float: return self._f(self.bump,  50.0)
    def get_droop(self) -> float: return self._f(self.droop, 50.0)
    def get_steps(self) -> int:   return max(3, int(self._f(self.steps, 41)))
    def get_track(self) -> float: return self._f(self.track, 300.0)
    def get_targets(self) -> KinematicTargets:
        return KinematicTargets(
            camber_gain_per_mm = self._f(self.t_camber, -0.05),
            roll_center_height = self._f(self.t_rc,     60.0),
            bump_steer_per_mm  = self._f(self.t_bsteer, 0.002),
            motion_ratio       = self._f(self.t_mr,     0.65),
        )


# ── Stats panel ───────────────────────────────────────────────────────────────
class StatsPanel(tk.Frame):
    STATS = [
        ("static_camber",       "Static camber",   "deg"),
        ("static_toe",          "Static toe",      "deg"),
        ("static_caster",       "Static caster",   "deg"),
        ("static_rc_height",    "RC height",       "mm"),
        ("static_motion_ratio", "Motion ratio",    "—"),
    ]
    def __init__(self, master, **kw):
        super().__init__(master, bg=BG2, **kw)
        label(self, "Static values", fg=ACCENT,
              font=("Consolas",11,"bold")).pack(fill="x", padx=8, pady=(6,2))
        self._labels = {}
        for key, lbl, unit in self.STATS:
            f = tk.Frame(self, bg=BG2); f.pack(fill="x", padx=8, pady=1)
            tk.Label(f, text=lbl, bg=BG2, fg=DIMTEXT,
                     font=("Consolas",9), width=16, anchor="w").pack(side="left")
            vl = tk.Label(f, text="—", bg=BG2, fg=GREEN,
                          font=("Consolas",9,"bold"), width=9, anchor="e")
            vl.pack(side="left")
            tk.Label(f, text=unit, bg=BG2, fg=DIMTEXT,
                     font=("Consolas",9), width=4).pack(side="left")
            self._labels[key] = vl
        label(self, "Score", fg=ACCENT,
              font=("Consolas",11,"bold")).pack(fill="x", padx=8, pady=(8,2))
        self._score = tk.Label(self, text="—", bg=BG2, fg=GREEN,
                               font=("Consolas",16,"bold"))
        self._score.pack(padx=8)
        self._detail = tk.Label(self, text="", bg=BG2, fg=DIMTEXT,
                                font=("Consolas",8), justify="left")
        self._detail.pack(padx=8, pady=2)

    def update(self, result: KinematicResult, targets: KinematicTargets):
        for key, _, _ in self.STATS:
            self._labels[key].config(text=f"{getattr(result, key):+.3f}")
        score, details = self._score_text(result, targets)
        color = GREEN if score > 75 else YELLOW if score > 50 else RED
        self._score.config(text=f"{score:.0f} / 100", fg=color)
        self._detail.config(text=details)

    def _score_text(self, r, t):
        from optimizer import score_result, OptimizerWeights
        w = OptimizerWeights(25, 25, 25, 25)
        raw = score_result(r, t, w)
        score = max(0, 100 - raw * 30)
        rc_err = abs(r.static_rc_height - t.roll_center_height)
        mr_err = abs(r.static_motion_ratio - t.motion_ratio)
        details = f"RC err: {rc_err:.1f}mm  |  MR err: {mr_err:.3f}"
        return score, details


# ── Plot panel ────────────────────────────────────────────────────────────────
class PlotPanel(tk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, bg=BG, **kw)
        self.fig = Figure(figsize=(8,5), facecolor=BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def update(self, result: KinematicResult, targets: KinematicTargets):
        self.fig.clear()
        plots = [
            ("Camber (deg)", result.camber_deg, "#7c6af7", None),
            ("Toe / Bump steer (deg)", result.toe_deg, "#a6e3a1", None),
            ("Roll center height (mm)", result.roll_center_z, "#f9e2af",
             np.full_like(result.travel_mm, targets.roll_center_height)),
            ("Motion ratio", result.motion_ratio, "#f38ba8",
             np.full_like(result.travel_mm, targets.motion_ratio)),
        ]
        t = result.travel_mm
        for i, (title, y, color, tgt) in enumerate(plots):
            ax = self.fig.add_subplot(2, 2, i+1)
            ax.set_facecolor(BG2)
            ax.tick_params(colors=DIMTEXT, labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
            ax.set_title(title, color=DIMTEXT, fontsize=8, pad=3)
            ax.set_xlabel("Travel (mm)", color=DIMTEXT, fontsize=7)
            ax.axhline(0, color=BORDER, linewidth=0.5)
            ax.axvline(0, color=BORDER, linewidth=0.5, linestyle="--")
            ax.plot(t, y, color=color, linewidth=1.5, label="actual")
            if tgt is not None:
                ax.plot(t, tgt, color=DIMTEXT, linewidth=1,
                        linestyle="--", label="target")
                ax.legend(fontsize=6, facecolor=BG2, labelcolor=DIMTEXT)
        self.fig.tight_layout(pad=1.5)
        self.canvas.draw_idle()


# ── Optimizer tab ─────────────────────────────────────────────────────────────
class OptimizerTab(tk.Frame):
    def __init__(self, master, get_hp_cb, get_settings_cb,
                 apply_result_cb, **kw):
        super().__init__(master, bg=BG2, **kw)
        self.get_hp = get_hp_cb
        self.get_settings = get_settings_cb
        self.apply_result = apply_result_cb
        self._running = False
        self._bounds_vars = {}   # name -> {x_min, x_max, x_lock, y_min, ...}
        self._weight_vars = {}
        self._build()

    def _build(self):
        # ── top: weights + run ──────────────────────────────────────────
        top = tk.Frame(self, bg=BG2)
        top.pack(fill="x", padx=10, pady=8)

        label(top, "Objective weights (0–100)", fg=ACCENT,
              font=("Consolas",11,"bold")).grid(row=0, column=0,
              columnspan=8, sticky="w", pady=(0,4))

        weight_defs = [
            ("Camber gain", "camber_gain", "30"),
            ("Roll center", "roll_center", "30"),
            ("Bump steer",  "bump_steer",  "20"),
            ("Motion ratio","motion_ratio","20"),
        ]
        for col, (lbl, key, default) in enumerate(weight_defs):
            tk.Label(top, text=lbl, bg=BG2, fg=DIMTEXT,
                     font=("Consolas",9)).grid(row=1, column=col*2, sticky="w", padx=(8,2))
            sv = tk.StringVar(value=default)
            tk.Entry(top, textvariable=sv, bg=BG, fg=TEXT,
                     insertbackground=TEXT, font=("Consolas",9),
                     width=5, bd=0, highlightthickness=1,
                     highlightbackground=BORDER, highlightcolor=ACCENT
                     ).grid(row=1, column=col*2+1, padx=(0,12))
            self._weight_vars[key] = sv

        # Optimizer settings row
        opt_frame = tk.Frame(self, bg=BG2)
        opt_frame.pack(fill="x", padx=10, pady=(0,4))
        for lbl, key, default, w in [
            ("Max iterations", "max_iter", "200", 5),
            ("Population size","popsize",  "8",   4),
        ]:
            tk.Label(opt_frame, text=lbl, bg=BG2, fg=DIMTEXT,
                     font=("Consolas",9)).pack(side="left", padx=(8,2))
            sv = tk.StringVar(value=default)
            tk.Entry(opt_frame, textvariable=sv, bg=BG, fg=TEXT,
                     insertbackground=TEXT, font=("Consolas",9),
                     width=w, bd=0, highlightthickness=1,
                     highlightbackground=BORDER, highlightcolor=ACCENT
                     ).pack(side="left", padx=(0,12))
            self._weight_vars[key] = sv

        # Run button + progress
        btn_frame = tk.Frame(self, bg=BG2)
        btn_frame.pack(fill="x", padx=10, pady=4)

        self._run_btn = tk.Button(btn_frame, text="▶  Run Optimizer",
                                  bg=ACCENT, fg=BG,
                                  font=("Consolas",10,"bold"), relief="flat",
                                  padx=14, pady=6, command=self._run,
                                  activebackground="#9d8fff", activeforeground=BG)
        self._run_btn.pack(side="left", padx=(0,12))

        self._apply_btn = tk.Button(btn_frame, text="✓  Apply Result",
                                    bg=BG, fg=GREEN,
                                    font=("Consolas",10,"bold"), relief="flat",
                                    padx=14, pady=6, command=self._apply,
                                    state="disabled",
                                    activebackground=BG2, activeforeground=GREEN)
        self._apply_btn.pack(side="left", padx=(0,12))

        self._status = tk.Label(btn_frame, text="Ready", bg=BG2, fg=DIMTEXT,
                                font=("Consolas",9))
        self._status.pack(side="left", padx=8)

        self._prog = ttk.Progressbar(btn_frame, mode="indeterminate", length=160)
        self._prog.pack(side="left", padx=8)

        # Result summary
        self._result_lbl = tk.Label(self, text="", bg=BG2, fg=GREEN,
                                    font=("Consolas",9), justify="left")
        self._result_lbl.pack(fill="x", padx=18, pady=2)

        # ── bounds table (scrollable) ───────────────────────────────────
        label(self, "Hardpoint bounds (mm) — uncheck lock to allow movement",
              fg=ACCENT, font=("Consolas",10,"bold")).pack(
              fill="x", padx=10, pady=(8,2))

        # Column headers
        hdr = tk.Frame(self, bg=BG2); hdr.pack(fill="x", padx=10)
        for col, (text, width) in enumerate([
            ("Point", 20), ("X min",6),("X max",6),("X lock",5),
            ("Y min",6),("Y max",6),("Y lock",5),
            ("Z min",6),("Z max",6),("Z lock",5),
        ]):
            tk.Label(hdr, text=text, bg=BG2, fg=DIMTEXT,
                     font=("Consolas",8), width=width,
                     anchor="w").grid(row=0, column=col, padx=2)

        canvas = tk.Canvas(self, bg=BG2, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=BG2)
        canvas.create_window((0,0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        hp = Hardpoints()
        for row_i, name in enumerate(POINT_NAMES):
            val = getattr(hp, name)
            RANGE = 25.0
            row_vars = {}
            bg = BG if row_i % 2 == 0 else BG2

            tk.Label(inner, text=POINT_LABELS[name], bg=bg, fg=TEXT,
                     font=("Consolas",9), width=20, anchor="w",
                     padx=4).grid(row=row_i, column=0, sticky="w")

            col = 1
            for axis_i, axis in enumerate(["x","y","z"]):
                v = val[axis_i]
                sv_min = tk.StringVar(value=f"{v-RANGE:.1f}")
                sv_max = tk.StringVar(value=f"{v+RANGE:.1f}")
                sv_lock = tk.BooleanVar(value=False)

                tk.Entry(inner, textvariable=sv_min, bg=bg, fg=TEXT,
                         insertbackground=TEXT, font=("Consolas",8),
                         width=6, bd=0, highlightthickness=1,
                         highlightbackground=BORDER, highlightcolor=ACCENT
                         ).grid(row=row_i, column=col, padx=2, pady=2)
                tk.Entry(inner, textvariable=sv_max, bg=bg, fg=TEXT,
                         insertbackground=TEXT, font=("Consolas",8),
                         width=6, bd=0, highlightthickness=1,
                         highlightbackground=BORDER, highlightcolor=ACCENT
                         ).grid(row=row_i, column=col+1, padx=2, pady=2)
                tk.Checkbutton(inner, variable=sv_lock, bg=bg,
                               fg=DIMTEXT, selectcolor=BG,
                               activebackground=bg).grid(row=row_i, column=col+2)

                row_vars[f"{axis}_min"]  = sv_min
                row_vars[f"{axis}_max"]  = sv_max
                row_vars[f"{axis}_lock"] = sv_lock
                col += 3

            self._bounds_vars[name] = row_vars

        self._last_result = None

    def _get_bounds(self):
        hp = self.get_hp()
        bounds = {}
        for name in POINT_NAMES:
            rv = self._bounds_vars[name]
            def f(sv, default):
                try: return float(sv.get())
                except: return default
            val = getattr(hp, name)
            bounds[name] = PointBounds(
                x_min=f(rv["x_min"], val[0]-25), x_max=f(rv["x_max"], val[0]+25),
                x_locked=rv["x_lock"].get(),
                y_min=f(rv["y_min"], val[1]-25), y_max=f(rv["y_max"], val[1]+25),
                y_locked=rv["y_lock"].get(),
                z_min=f(rv["z_min"], val[2]-25), z_max=f(rv["z_max"], val[2]+25),
                z_locked=rv["z_lock"].get(),
            )
        return bounds

    def _get_weights(self):
        def f(key, default):
            try: return float(self._weight_vars[key].get())
            except: return default
        return OptimizerWeights(
            camber_gain  = f("camber_gain",  30),
            roll_center  = f("roll_center",  30),
            bump_steer   = f("bump_steer",   20),
            motion_ratio = f("motion_ratio", 20),
        )

    def _run(self):
        if self._running:
            return
        self._running = True
        self._run_btn.config(state="disabled")
        self._apply_btn.config(state="disabled")
        self._status.config(text="Running...", fg=YELLOW)
        self._prog.start(10)
        self._last_result = None

        hp       = self.get_hp()
        bounds   = self._get_bounds()
        weights  = self._get_weights()
        settings = self.get_settings()
        targets  = settings.get_targets()
        bump     = settings.get_bump()
        droop    = settings.get_droop()
        track    = settings.get_track()

        try:
            max_iter = int(self._weight_vars["max_iter"].get())
        except: max_iter = 200
        try:
            popsize  = int(self._weight_vars["popsize"].get())
        except: popsize = 8

        def worker():
            def progress(gen, score):
                self.after(0, lambda g=gen, s=score:
                    self._status.config(
                        text=f"Gen {g}  best score: {s:.4f}", fg=YELLOW))
            try:
                result = run_optimizer(
                    hp, bounds, targets, weights,
                    bump_mm=bump, droop_mm=droop, steps=21,
                    track_half=track, max_iter=max_iter, popsize=popsize,
                    progress_cb=progress,
                )
                self.after(0, lambda r=result: self._done(r))
            except Exception as e:
                self.after(0, lambda err=e: self._error(err))

        threading.Thread(target=worker, daemon=True).start()

    def _done(self, result: OptimizeResult):
        self._running = False
        self._prog.stop()
        self._run_btn.config(state="normal")
        self._apply_btn.config(state="normal")
        self._last_result = result

        k = result.best_kinematic
        conv = "✓ converged" if result.converged else "~ not fully converged"
        self._status.config(
            text=f"Done — score: {result.best_score:.4f}  {conv}", fg=GREEN)
        self._result_lbl.config(
            text=(f"Camber: {k.static_camber:+.2f}°  |  "
                  f"RC: {k.static_rc_height:.1f}mm  |  "
                  f"MR: {k.static_motion_ratio:.3f}  |  "
                  f"Evals: {result.n_evaluations}"))

    def _error(self, err):
        self._running = False
        self._prog.stop()
        self._run_btn.config(state="normal")
        self._status.config(text=f"Error: {err}", fg=RED)

    def _apply(self):
        if self._last_result:
            self.apply_result(self._last_result.best_hardpoints)
            self._status.config(text="Result applied to Hardpoints tab ✓", fg=GREEN)


# ── Main app ──────────────────────────────────────────────────────────────────
class SuspensionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Suspension Hardpoint Designer — Double Wishbone Front | Baja SAE")
        self.configure(bg=BG)
        self.geometry("1340x820")
        self.minsize(1000, 650)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame",     background=BG2)
        style.configure("TScrollbar", background=BG2,
                        troughcolor=BG, arrowcolor=DIMTEXT)
        style.configure("TNotebook",  background=BG, borderwidth=0)
        style.configure("TNotebook.Tab",
                        background=BG2, foreground=DIMTEXT,
                        font=("Consolas",10), padding=(12,5))
        style.map("TNotebook.Tab",
                  background=[("selected", BG)],
                  foreground=[("selected", ACCENT)])

        self._last_result = None
        self._last_hp     = None
        self._build_menu()
        self._build_layout()
        self.after(250, self._compute)

    def _build_menu(self):
        m = tk.Menu(self, bg=BG2, fg=TEXT,
                    activebackground=ACCENT, activeforeground=BG)
        fm = tk.Menu(m, tearoff=0, bg=BG2, fg=TEXT,
                     activebackground=ACCENT, activeforeground=BG)
        fm.add_command(label="Export hardpoints CSV",  command=self._export_hp)
        fm.add_command(label="Export kinematics CSV",  command=self._export_kin)
        fm.add_separator()
        fm.add_command(label="Reset to defaults",      command=self._reset)
        fm.add_separator()
        fm.add_command(label="Quit",                   command=self.destroy)
        m.add_cascade(label="File", menu=fm)
        self.config(menu=m)

    def _build_layout(self):
        # Left: hardpoint entry
        left = tk.Frame(self, bg=BG2, width=350)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        self.hp_panel = HardpointPanel(left, on_change_cb=self._on_change)
        self.hp_panel.pack(fill="both", expand=True)

        # Right: notebook
        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True, padx=6, pady=6)

        # Tab 1: Kinematics
        kin_tab = tk.Frame(nb, bg=BG)
        nb.add(kin_tab, text="  Kinematics  ")

        top = tk.Frame(kin_tab, bg=BG)
        top.pack(fill="x")
        self.settings = SettingsPanel(top, on_change_cb=self._on_change)
        self.settings.pack(side="left", fill="y", padx=(4,2), pady=6)
        self.stats = StatsPanel(top)
        self.stats.pack(side="left", fill="y", padx=(2,4), pady=6)
        tk.Button(top, text="▶  Compute", bg=ACCENT, fg=BG,
                  font=("Consolas",10,"bold"), relief="flat",
                  padx=12, pady=6, command=self._compute,
                  activebackground="#9d8fff", activeforeground=BG
                  ).pack(side="left", padx=10, anchor="n", pady=10)
        self.plots = PlotPanel(kin_tab)
        self.plots.pack(fill="both", expand=True, padx=4, pady=(0,6))

        # Tab 2: Optimizer
        self.opt_tab = OptimizerTab(
            nb,
            get_hp_cb       = self.hp_panel.get_hardpoints,
            get_settings_cb = lambda: self.settings,
            apply_result_cb = self._apply_optimizer_result,
        )
        nb.add(self.opt_tab, text="  Optimizer  ")

    def _on_change(self):
        if hasattr(self, "_after_id"):
            self.after_cancel(self._after_id)
        self._after_id = self.after(400, self._compute)

    def _compute(self):
        hp      = self.hp_panel.get_hardpoints()
        targets = self.settings.get_targets()
        try:
            result = run_sweep(
                hp,
                bump_mm    = self.settings.get_bump(),
                droop_mm   = self.settings.get_droop(),
                steps      = self.settings.get_steps(),
                track_half = self.settings.get_track(),
            )
            self._last_result = result
            self._last_hp     = hp
            self.stats.update(result, targets)
            self.plots.update(result, targets)
        except Exception as e:
            messagebox.showerror("Solver error", str(e))

    def _apply_optimizer_result(self, hp: Hardpoints):
        self.hp_panel.set_hardpoints(hp)
        self._compute()

    def _export_hp(self):
        if not self._last_hp:
            messagebox.showinfo("Nothing to export", "Run a computation first.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv",
            filetypes=[("CSV","*.csv")], initialfile="hardpoints.csv")
        if p:
            export_hardpoints_csv(self._last_hp, p)
            messagebox.showinfo("Saved", p)

    def _export_kin(self):
        if not self._last_result:
            messagebox.showinfo("Nothing to export", "Run a computation first.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv",
            filetypes=[("CSV","*.csv")], initialfile="kinematics.csv")
        if p:
            export_kinematics_csv(self._last_result, p)
            messagebox.showinfo("Saved", p)

    def _reset(self):
        self.hp_panel.set_hardpoints(Hardpoints())
        self._compute()


if __name__ == "__main__":
    SuspensionApp().mainloop()
