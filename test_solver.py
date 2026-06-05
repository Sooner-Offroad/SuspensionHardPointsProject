"""
Suspension Hardpoint Designer — GUI
Double Wishbone Front Suspension
Baja SAE / Offroad use

Run:
    python app.py

Requirements:
    pip install numpy scipy matplotlib
    (tkinter is built into Python — no install needed)

Optional upgrade to PyQt5 for a more polished look:
    pip install pyqt5
    (This app uses tkinter but the solver is UI-independent)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from solver import (
    Hardpoints, KinematicTargets, KinematicResult,
    run_sweep, export_hardpoints_csv, export_kinematics_csv
)


# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------
BG       = "#1e1e2e"
BG2      = "#2a2a3e"
ACCENT   = "#7c6af7"
TEXT     = "#cdd6f4"
TEXT_DIM = "#6c7086"
GREEN    = "#a6e3a1"
YELLOW   = "#f9e2af"
RED      = "#f38ba8"
BORDER   = "#313244"


# ---------------------------------------------------------------------------
# Hardpoint entry panel
# ---------------------------------------------------------------------------

class HardpointPanel(ttk.Frame):
    """Left panel: all editable hardpoint coordinates."""

    POINTS = [
        ("uca_inboard_front", "UCA Inboard Front"),
        ("uca_inboard_rear",  "UCA Inboard Rear"),
        ("uca_outboard",      "UCA Outboard (UBJ)"),
        ("lca_inboard_front", "LCA Inboard Front"),
        ("lca_inboard_rear",  "LCA Inboard Rear"),
        ("lca_outboard",      "LCA Outboard (LBJ)"),
        ("tie_rod_inboard",   "Tie Rod Inboard"),
        ("tie_rod_outboard",  "Tie Rod Outboard"),
        ("shock_upper",       "Shock Upper Mount"),
        ("shock_lower",       "Shock Lower Mount"),
        ("wheel_center",      "Wheel Center (static)"),
    ]

    def __init__(self, master, on_change_cb, **kwargs):
        super().__init__(master, **kwargs)
        self.on_change_cb = on_change_cb
        self.vars = {}   # {point_name: [tk.StringVar x, y, z]}
        self._build()

    def _build(self):
        # Header
        hdr = tk.Label(self, text="Hardpoints (mm)", bg=BG2, fg=ACCENT,
                       font=("Consolas", 12, "bold"), pady=8)
        hdr.pack(fill="x")

        axis_row = tk.Frame(self, bg=BG2)
        axis_row.pack(fill="x", padx=8)
        tk.Label(axis_row, text="Point", bg=BG2, fg=TEXT_DIM,
                 font=("Consolas", 9), width=20, anchor="w").grid(row=0, column=0)
        for col, lbl in enumerate(["X", "Y", "Z"], 1):
            tk.Label(axis_row, text=lbl, bg=BG2, fg=TEXT_DIM,
                     font=("Consolas", 9), width=8).grid(row=0, column=col)

        # Scrollable frame
        canvas = tk.Canvas(self, bg=BG2, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=BG2)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        hp = Hardpoints()

        for row_i, (key, label) in enumerate(self.POINTS):
            vals = getattr(hp, key)
            svars = []
            bg = BG if row_i % 2 == 0 else BG2

            tk.Label(inner, text=label, bg=bg, fg=TEXT,
                     font=("Consolas", 9), width=20, anchor="w",
                     padx=4).grid(row=row_i, column=0, sticky="w")

            for col_i, val in enumerate(vals):
                sv = tk.StringVar(value=f"{val:.1f}")
                sv.trace_add("write", lambda *a, k=key: self._on_edit(k))
                e = tk.Entry(inner, textvariable=sv, bg=bg, fg=TEXT,
                             insertbackground=TEXT, font=("Consolas", 9),
                             width=8, bd=0, highlightthickness=1,
                             highlightbackground=BORDER,
                             highlightcolor=ACCENT)
                e.grid(row=row_i, column=col_i + 1, padx=2, pady=2)
                svars.append(sv)

            self.vars[key] = svars

        # Mousewheel scroll
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))

    def _on_edit(self, key):
        self.on_change_cb()

    def get_hardpoints(self) -> Hardpoints:
        hp = Hardpoints()
        for key, svars in self.vars.items():
            try:
                vals = [float(sv.get()) for sv in svars]
                setattr(hp, key, np.array(vals))
            except ValueError:
                pass
        return hp

    def set_hardpoints(self, hp: Hardpoints):
        for key, svars in self.vars.items():
            vals = getattr(hp, key)
            for sv, v in zip(svars, vals):
                sv.set(f"{v:.1f}")


# ---------------------------------------------------------------------------
# Travel & targets panel
# ---------------------------------------------------------------------------

class SettingsPanel(tk.Frame):

    def __init__(self, master, on_change_cb, **kwargs):
        super().__init__(master, bg=BG2, **kwargs)
        self.on_change_cb = on_change_cb

        tk.Label(self, text="Sweep Settings", bg=BG2, fg=ACCENT,
                 font=("Consolas", 11, "bold"), pady=6).pack(fill="x", padx=8)

        self.bump   = self._row("Bump travel (mm)",   "50")
        self.droop  = self._row("Droop travel (mm)",  "50")
        self.steps  = self._row("Sweep steps",        "41")
        self.track  = self._row("Half-track (mm)",    "330")

        tk.Label(self, text="Targets", bg=BG2, fg=ACCENT,
                 font=("Consolas", 11, "bold"), pady=6).pack(fill="x", padx=8)

        self.t_camber = self._row("Camber gain (deg/mm)",  "-0.05")
        self.t_rc     = self._row("Roll center height (mm)", "50")
        self.t_bsteer = self._row("Max bump steer (deg/mm)", "0.002")
        self.t_mr     = self._row("Motion ratio target",    "0.65")

    def _row(self, label, default):
        f = tk.Frame(self, bg=BG2)
        f.pack(fill="x", padx=8, pady=2)
        tk.Label(f, text=label, bg=BG2, fg=TEXT,
                 font=("Consolas", 9), width=26, anchor="w").pack(side="left")
        sv = tk.StringVar(value=default)
        sv.trace_add("write", lambda *a: self.on_change_cb())
        tk.Entry(f, textvariable=sv, bg=BG, fg=TEXT,
                 insertbackground=TEXT, font=("Consolas", 9),
                 width=8, bd=0, highlightthickness=1,
                 highlightbackground=BORDER, highlightcolor=ACCENT).pack(side="left")
        return sv

    def get_bump(self)  -> float: return self._f(self.bump,  50.0)
    def get_droop(self) -> float: return self._f(self.droop, 50.0)
    def get_steps(self) -> int:   return max(3, int(self._f(self.steps, 41)))
    def get_track(self) -> float: return self._f(self.track, 330.0)

    def get_targets(self) -> KinematicTargets:
        return KinematicTargets(
            camber_gain_per_mm = self._f(self.t_camber, -0.05),
            roll_center_height = self._f(self.t_rc,     50.0),
            bump_steer_per_mm  = self._f(self.t_bsteer, 0.002),
            motion_ratio       = self._f(self.t_mr,     0.65),
        )

    def _f(self, sv, default):
        try:    return float(sv.get())
        except: return default


# ---------------------------------------------------------------------------
# Results / stats panel
# ---------------------------------------------------------------------------

class StatsPanel(tk.Frame):

    STATS = [
        ("static_camber",       "Static camber",        "deg"),
        ("static_toe",          "Static toe",           "deg"),
        ("static_caster",       "Static caster",        "deg"),
        ("static_rc_height",    "Roll center height",   "mm"),
        ("static_motion_ratio", "Motion ratio (static)","—"),
    ]

    def __init__(self, master, **kwargs):
        super().__init__(master, bg=BG2, **kwargs)
        tk.Label(self, text="Static Values", bg=BG2, fg=ACCENT,
                 font=("Consolas", 11, "bold"), pady=6).pack(fill="x", padx=8)
        self._labels = {}
        for key, label, unit in self.STATS:
            f = tk.Frame(self, bg=BG2)
            f.pack(fill="x", padx=8, pady=1)
            tk.Label(f, text=label, bg=BG2, fg=TEXT_DIM,
                     font=("Consolas", 9), width=24, anchor="w").pack(side="left")
            lbl = tk.Label(f, text="—", bg=BG2, fg=GREEN,
                           font=("Consolas", 9, "bold"), width=10, anchor="e")
            lbl.pack(side="left")
            tk.Label(f, text=unit, bg=BG2, fg=TEXT_DIM,
                     font=("Consolas", 9), width=4).pack(side="left")
            self._labels[key] = lbl

        # Score section
        tk.Label(self, text="Target Score", bg=BG2, fg=ACCENT,
                 font=("Consolas", 11, "bold"), pady=6).pack(fill="x", padx=8)
        self._score_lbl = tk.Label(self, text="—", bg=BG2, fg=GREEN,
                                   font=("Consolas", 14, "bold"))
        self._score_lbl.pack(padx=8, pady=4)
        self._score_detail = tk.Label(self, text="", bg=BG2, fg=TEXT_DIM,
                                      font=("Consolas", 8), justify="left",
                                      wraplength=220)
        self._score_detail.pack(padx=8)

    def update(self, result: KinematicResult, targets: KinematicTargets):
        for key, _, _ in self.STATS:
            val = getattr(result, key)
            self._labels[key].config(text=f"{val:+.3f}")

        # Score against targets (simple weighted penalty)
        score, details = self._score(result, targets)
        color = GREEN if score > 75 else YELLOW if score > 50 else RED
        self._score_lbl.config(text=f"{score:.0f} / 100", fg=color)
        self._score_detail.config(text=details)

    def _score(self, r: KinematicResult, t: KinematicTargets) -> tuple[float, str]:
        lines = []
        total = 0.0

        # RC height within 20mm of target
        rc_err = abs(r.static_rc_height - t.roll_center_height)
        rc_pts = max(0.0, 25.0 * (1 - rc_err / 30.0))
        total += rc_pts
        lines.append(f"RC height err: {rc_err:.1f}mm  → {rc_pts:.0f}/25")

        # Camber gain rate
        if len(r.travel_mm) > 1:
            dt = np.diff(r.travel_mm)
            dc = np.diff(r.camber_deg)
            rates = dc / (dt + 1e-9)
            mid = len(rates) // 2
            cam_rate = float(np.mean(rates[mid-2:mid+2]))
        else:
            cam_rate = 0.0
        cam_err = abs(cam_rate - t.camber_gain_per_mm)
        cam_pts = max(0.0, 25.0 * (1 - cam_err / 0.1))
        total += cam_pts
        lines.append(f"Camber rate: {cam_rate:+.4f}  → {cam_pts:.0f}/25")

        # Motion ratio
        mr_err = abs(r.static_motion_ratio - t.motion_ratio)
        mr_pts = max(0.0, 25.0 * (1 - mr_err / 0.3))
        total += mr_pts
        lines.append(f"Motion ratio err: {mr_err:.3f}  → {mr_pts:.0f}/25")

        # Bump steer (max toe change per mm)
        if len(r.travel_mm) > 1:
            dt2 = np.diff(r.travel_mm)
            dtoe = np.diff(r.toe_deg)
            max_bs = float(np.max(np.abs(dtoe / (dt2 + 1e-9))))
        else:
            max_bs = 0.0
        bs_pts = max(0.0, 25.0 * (1 - max_bs / (t.bump_steer_per_mm * 5 + 1e-9)))
        total += bs_pts
        lines.append(f"Bump steer: {max_bs:.5f}  → {bs_pts:.0f}/25")

        return total, "\n".join(lines)


# ---------------------------------------------------------------------------
# Plot panel
# ---------------------------------------------------------------------------

class PlotPanel(tk.Frame):

    def __init__(self, master, **kwargs):
        super().__init__(master, bg=BG, **kwargs)
        self.fig = Figure(figsize=(8, 6), facecolor=BG)
        self.axes = []
        self._setup_axes()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _setup_axes(self):
        self.fig.clear()
        plots = [
            ("Camber (deg)", ACCENT),
            ("Toe / Bump Steer (deg)", GREEN),
            ("Roll Center Height (mm)", YELLOW),
            ("Motion Ratio", RED),
        ]
        self.axes = []
        for i, (title, _) in enumerate(plots):
            ax = self.fig.add_subplot(2, 2, i + 1)
            ax.set_facecolor(BG2)
            ax.tick_params(colors=TEXT_DIM, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
            ax.set_title(title, color=TEXT_DIM, fontsize=8, pad=4)
            ax.set_xlabel("Travel (mm)", color=TEXT_DIM, fontsize=7)
            ax.axhline(0, color=BORDER, linewidth=0.5)
            ax.axvline(0, color=BORDER, linewidth=0.5, linestyle="--")
            self.axes.append((ax, plots[i][1]))
        self.fig.tight_layout(pad=1.5)

    def update(self, result: KinematicResult, targets: KinematicTargets):
        t = result.travel_mm
        data = [
            (result.camber_deg,      targets.camber_gain_per_mm * t + result.camber_deg[len(t)//2]),
            (result.toe_deg,         None),
            (result.roll_center_z,   np.full_like(t, targets.roll_center_height)),
            (result.motion_ratio,    np.full_like(t, targets.motion_ratio)),
        ]

        for i, ((ax, color), (y, target)) in enumerate(zip(self.axes, data)):
            ax.clear()
            ax.set_facecolor(BG2)
            ax.tick_params(colors=TEXT_DIM, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
            titles = ["Camber (deg)", "Toe / Bump Steer (deg)",
                      "Roll Center Height (mm)", "Motion Ratio"]
            ax.set_title(titles[i], color=TEXT_DIM, fontsize=8, pad=4)
            ax.set_xlabel("Travel (mm)", color=TEXT_DIM, fontsize=7)
            ax.axhline(0, color=BORDER, linewidth=0.5)
            ax.axvline(0, color=BORDER, linewidth=0.5, linestyle="--")

            ax.plot(t, y, color=color, linewidth=1.5, label="actual")
            if target is not None:
                ax.plot(t, target, color=TEXT_DIM, linewidth=1,
                        linestyle="--", label="target")
                ax.legend(fontsize=6, facecolor=BG2, labelcolor=TEXT_DIM)

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class SuspensionApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Suspension Hardpoint Designer — Double Wishbone (Front)")
        self.configure(bg=BG)
        self.geometry("1280x780")
        self.minsize(900, 600)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=BG2)
        style.configure("TScrollbar", background=BG2, troughcolor=BG,
                        arrowcolor=TEXT_DIM)

        self._build_menu()
        self._build_layout()
        self._last_result: KinematicResult | None = None
        self._last_hp: Hardpoints | None = None

        # Initial compute
        self.after(200, self._compute)

    # ------------------------------------------------------------------
    def _build_menu(self):
        menubar = tk.Menu(self, bg=BG2, fg=TEXT, activebackground=ACCENT,
                          activeforeground=BG)
        file_menu = tk.Menu(menubar, tearoff=0, bg=BG2, fg=TEXT,
                            activebackground=ACCENT, activeforeground=BG)
        file_menu.add_command(label="Export hardpoints CSV",
                              command=self._export_hp)
        file_menu.add_command(label="Export kinematics CSV",
                              command=self._export_kin)
        file_menu.add_separator()
        file_menu.add_command(label="Reset to defaults",
                              command=self._reset)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

    # ------------------------------------------------------------------
    def _build_layout(self):
        # ---- Left sidebar: hardpoints ----
        left = tk.Frame(self, bg=BG2, width=340)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        self.hp_panel = HardpointPanel(left, on_change_cb=self._on_change)
        self.hp_panel.pack(fill="both", expand=True)

        # ---- Right side ----
        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        # Top strip: settings + stats
        top = tk.Frame(right, bg=BG)
        top.pack(fill="x")

        self.settings = SettingsPanel(top, on_change_cb=self._on_change)
        self.settings.pack(side="left", fill="y", padx=(8, 4), pady=8)

        self.stats = StatsPanel(top)
        self.stats.pack(side="left", fill="y", padx=(4, 8), pady=8)

        # Compute button
        btn = tk.Button(top, text="▶  Compute", bg=ACCENT, fg=BG,
                        font=("Consolas", 10, "bold"), relief="flat",
                        padx=12, pady=6, command=self._compute,
                        activebackground="#9d8fff", activeforeground=BG)
        btn.pack(side="left", padx=12, pady=8, anchor="n")

        # Plots
        self.plots = PlotPanel(right)
        self.plots.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    # ------------------------------------------------------------------
    def _on_change(self):
        # Debounce: schedule recompute 400ms after last keystroke
        if hasattr(self, "_after_id"):
            self.after_cancel(self._after_id)
        self._after_id = self.after(400, self._compute)

    def _compute(self):
        hp = self.hp_panel.get_hardpoints()
        bump  = self.settings.get_bump()
        droop = self.settings.get_droop()
        steps = self.settings.get_steps()
        track = self.settings.get_track()
        targets = self.settings.get_targets()

        try:
            result = run_sweep(hp, bump_mm=bump, droop_mm=droop,
                               steps=steps, track_half=track)
            self._last_result = result
            self._last_hp     = hp
            self.stats.update(result, targets)
            self.plots.update(result, targets)
        except Exception as e:
            messagebox.showerror("Solver error", str(e))

    # ------------------------------------------------------------------
    def _export_hp(self):
        if self._last_hp is None:
            messagebox.showinfo("Nothing to export", "Run a computation first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="hardpoints.csv"
        )
        if path:
            export_hardpoints_csv(self._last_hp, path)
            messagebox.showinfo("Exported", f"Hardpoints saved to:\n{path}")

    def _export_kin(self):
        if self._last_result is None:
            messagebox.showinfo("Nothing to export", "Run a computation first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="kinematics.csv"
        )
        if path:
            export_kinematics_csv(self._last_result, path)
            messagebox.showinfo("Exported", f"Kinematics saved to:\n{path}")

    def _reset(self):
        self.hp_panel.set_hardpoints(Hardpoints())
        self._compute()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = SuspensionApp()
    app.mainloop()