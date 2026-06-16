import numpy as np
import yaml
import copy

from pathlib import Path
from pymoo.core.problem import ElementwiseProblem
from kinematics.io.geometry_loader import load_geometry
from kinematics.io.sweep_loader import parse_sweep_file
from kinematics.main import solve_sweep
from kinematics.core.geometry import Point3
from kinematics.metrics import compute_metrics_for_state

class SuspensionProblem(ElementwiseProblem):
    def __init__(self, geometry_path: str, sweep_path: str):


        #load the suspension and sweep (input) data
        self.suspension = load_geometry(geometry_path)
        self.sweep_config = parse_sweep_file(sweep_path)
        self.optimized_points = sorted(list(self.suspension.REQUIRED_POINTS), key=lambda p: p.value)
        # number of variables being optimized, in x,y,z for each coordinate based hardpoint (since tyre data is fixed that doesnt count)
        n_var = len(self.optimized_points) * 3
        # number of objectives (for example camber change, caster change, etc.)
        n_obj = 1
        # number of constraints, these are objectives the solver MUST achieve exactly
        n_con = 0
        flat_initial_coords = []
        for point_id in self.optimized_points:
            point3_obj = self.suspension.hardpoints[point_id]
            
            # Unpack X, Y, Z using indices [0], [1], [2] instead of .x, .y, .z
            flat_initial_coords.extend([point3_obj[0], point3_obj[1], point3_obj[2]])
            
        # Convert to a NumPy array (this will automatically be of length n_var!)
        initial_hardpoints = np.array(flat_initial_coords)
        
        # 5. Define bounding boxes (permitting +/- 20mm of movement from baseline design)
        xl = initial_hardpoints - 5.0
        xu = initial_hardpoints + 5.0
        print("Problem Succesfully Initialized!")
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_con, xl=xl, xu=xu)
    
    def _evaluate(self, x, out, *args, **kwargs):

        try:

            eval_suspension = copy.deepcopy(self.suspension)
            #update hardpoints
            for i, point_id in enumerate(self.optimized_points):
                start_idx = i * 3
                pt_x = x[start_idx]
                pt_y = x[start_idx + 1]
                pt_z = x[start_idx + 2]

                eval_suspension.hardpoints[point_id] = Point3([pt_x, pt_y, pt_z])
            #evaluate here
            solution_states, solver_stats = solve_sweep(eval_suspension, self.sweep_config)

            # --- TEMPORARY DEBUG CODE ---
            '''print("\n--- Checking Solved Positions for the First State ---")
            first_state = solution_states[0]
            
            for pid in self.suspension.OUTPUT_POINTS:
                # Grab the actual solved coordinate using the PID label
                pos = first_state.positions.get(pid)
                if pos is not None:
                    # 'pid.name' gives the human-readable string
                    print(f"{pid.name}: X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f}")'''

            output_points = eval_suspension.OUTPUT_POINTS
            config = eval_suspension.config
            all_sweep_metrics = []
            max_abs_scrub = 0.0
            for st in solution_states:
                metrics = {}
                if config is not None:
                    # This returns a dictionary of metric names to floats
                    metrics = compute_metrics_for_state(st, eval_suspension, config)
                    current_scrub = metrics.get("scrub_radius_mm", 0.0)
                    if abs(current_scrub) > max_abs_scrub:
                        max_abs_scrub = abs(current_scrub)
                all_sweep_metrics.append(metrics)

            print(f"Max Abs Scrub: {max_abs_scrub}")
            out["F"] = [max_abs_scrub]
            

        except Exception as e:
            print(f"Evaluation crashed: {type(e).__name__} - {e}")
            # punish bad results here
            out["F"] = [1e6] * self.n_obj


        
      
