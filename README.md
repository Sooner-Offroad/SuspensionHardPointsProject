# Suspension Hardpoint Designer
### Double Wishbone Front Suspension — Baja SAE

---

## Setup (one time)

```bash
pip install numpy scipy matplotlib
```

tkinter is built into Python — no install needed.

---

## Run

```bash
python app.py
```

---

## File structure

```
suspension_tool/
├── app.py          # GUI application (run this)
├── solver.py       # Kinematic solver — pure math, no UI
├── test_solver.py  # Headless sanity check
└── README.md
```

---

## Coordinate system (SAE standard)

```
  X = forward (positive toward front of car)
  Y = left    (positive toward driver left)
  Z = up      (positive upward)
```

All units: **mm** and **degrees**.

The tool models the **front-left corner**. The right side is automatically
mirrored (Y negated) for roll center calculation.

---

## Default hardpoints (Baja SAE scale)

These are realistic starting values based on a ~54" front track, ~215mm
ride height. Tune them to match your actual chassis pickup points.

| Point              | X     | Y     | Z     | Notes                        |
|--------------------|-------|-------|-------|------------------------------|
| UCA inboard front  | +60   | 155   | 295   | Chassis, front leg           |
| UCA inboard rear   | -60   | 155   | 295   | Chassis, rear leg            |
| UCA outboard (UBJ) | 0     | 297   | 280   | Upper ball joint on knuckle  |
| LCA inboard front  | +80   | 120   | 175   | Chassis, front leg           |
| LCA inboard rear   | -80   | 120   | 175   | Chassis, rear leg            |
| LCA outboard (LBJ) | 0     | 300   | 150   | Lower ball joint on knuckle  |
| Tie rod inboard    | -30   | 160   | 195   | Rack end                     |
| Tie rod outboard   | -30   | 295   | 195   | Steering arm on knuckle      |
| Shock upper        | 0     | 195   | 345   | Chassis mount                |
| Shock lower        | 0     | 270   | 195   | LCA-side mount               |
| Wheel center       | 0     | 300   | 215   | Hub center at static RH      |

---

## Outputs

### Live plots (4 panels)
- **Camber** vs. travel — negative gain in bump is good for Baja
- **Toe / Bump steer** vs. travel — want this near flat
- **Roll center height** vs. travel — target ~40–80mm for offroad
- **Motion ratio** vs. travel — target ~0.6–0.75 for Baja

### Static values panel
Reports key values at ride height: camber, toe, caster, RC height, motion ratio.

### Target score (0–100)
Weighted penalty against your targets. 4 sub-scores of 25 each:
- RC height error vs. target
- Camber gain rate vs. target
- Motion ratio error vs. target
- Max bump steer vs. tolerance

### CSV exports (File menu)
- **hardpoints.csv** — all XYZ coords, ready to import into CAD
- **kinematics.csv** — full travel sweep table

---

## Tuning tips for Baja

| Parameter         | Typical range     | Why it matters                          |
|-------------------|-------------------|-----------------------------------------|
| Camber (static)   | -1° to -2°        | Compensates for body roll on-course     |
| RC height         | 40–100mm          | Higher = less roll, more jacking        |
| Bump steer        | < 0.003 deg/mm    | Reduces steering fight over rough terrain|
| Motion ratio      | 0.55–0.75         | Controls wheel rate vs. spring rate     |
| Camber gain       | -0.03 to -0.07 deg/mm | Keeps tire contact patch in bump   |

---

## Upgrading to PyQt5 (optional, better UI)

```bash
pip install pyqt5
```

The solver (`solver.py`) is completely independent of the UI — swap the
frontend without touching any math.

---

## Next steps / roadmap
- [ ] Trailing arm rear suspension module
- [ ] 3D wireframe preview (matplotlib 3D)
- [ ] Anti-squat / anti-dive calculation
- [ ] Optimization mode (auto-iterate toward targets)
- [ ] Bump steer correction solver
- [ ] Ackermann geometry for steering
