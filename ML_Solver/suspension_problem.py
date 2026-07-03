import numpy as np
import yaml
import copy
import time

from pathlib import Path
from pymoo.core.problem import ElementwiseProblem
from kinematics.io.geometry_loader import load_geometry
from kinematics.io.sweep_loader import parse_sweep_file
from kinematics.main import solve_sweep
from kinematics.core.geometry import Point3
from kinematics.metrics import compute_metrics_for_state

class SuspensionProblem(ElementwiseProblem):
    def __init__(self, geometry_path: str, sweep_path: str, **kwargs):


        #load the suspension and sweep (input) data
        self.suspension = load_geometry(geometry_path)
        self.sweep_config = parse_sweep_file(sweep_path)
        self.optimized_points = sorted(list(self.suspension.REQUIRED_POINTS), key=lambda p: p.value)
        self.count = 0
        # number of variables being optimized, in x,y,z for each coordinate based hardpoint (since tyre data is fixed that doesnt count)
        n_var = len(self.optimized_points) * 3
        # number of objectives (for example camber change, caster change, etc.)
        n_obj = 7
        # number of constraints, these are objectives the solver MUST achieve exactly
        n_con = 0

        # keep count of crashes
        self.crashes = 0

        flat_initial_coords = []
        for point_id in self.optimized_points:
            point3_obj = self.suspension.hardpoints[point_id]
            
            # Unpack X, Y, Z using indices [0], [1], [2] instead of .x, .y, .z
            flat_initial_coords.extend([point3_obj[0], point3_obj[1], point3_obj[2]])
            
        # Convert to a NumPy array (this will automatically be of length n_var!)
        initial_hardpoints = np.array(flat_initial_coords)
        
        # 5. Define bounding boxes (permitting +/- 20mm of movement from baseline design)
        xl = initial_hardpoints - 10.0
        xu = initial_hardpoints + 10.0
        print("Problem Succesfully Initialized!")
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_con, xl=xl, xu=xu, **kwargs)
    
    def _evaluate(self, x, out, *args, **kwargs):

        try:
            #t_start_copy = time.perf_counter()
            
            eval_suspension = copy.deepcopy(self.suspension)
            
            #t_copy = time.perf_counter() - t_start_copy

            #update hardpoints
            for i, point_id in enumerate(self.optimized_points):
                start_idx = i * 3
                pt_x = x[start_idx]
                pt_y = x[start_idx + 1]
                pt_z = x[start_idx + 2]

                eval_suspension.hardpoints[point_id] = Point3([pt_x, pt_y, pt_z])
            
            t_start_solve = time.perf_counter()

            solution_states, solver_stats = solve_sweep(eval_suspension, self.sweep_config)
            
            t_solve = time.perf_counter() - t_start_solve


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

            static_scrub = 0.0
            static_camber = 0.0
            static_toe = 0.0
            static_kpi = 0.0
            static_mech_trail = 0.0

            static_state_metrics = compute_metrics_for_state(eval_suspension.initial_state(), eval_suspension, config)
            
            static_scrub = static_state_metrics.get("scrub_radius_mm", 0.0)
            static_camber = static_state_metrics.get("camber_deg", 0.0)
            static_toe = static_state_metrics.get("roadwheel_angle_deg", 0.0)
            static_kpi = static_state_metrics.get("kpi_deg", 0.0)
            static_mech_trail = static_state_metrics.get("mechanical_trail_mm", 0.0)


            all_sweep_metrics = []
            max_abs_camber_rate = 0.0 # change in camber per inch of travel
            #t_start_metrics = time.perf_counter()
            prev_wheel_center_z = None
            prev_camber = None

            for i,st in enumerate(solution_states):
                metrics = {}
                if config is not None:
                    # This returns a dictionary of metric names to floats
                    metrics = compute_metrics_for_state(st, eval_suspension, config)

                    current_toe = metrics.get("roadwheel_angle_deg", 0.0)
                    current_camber = metrics.get("camber_deg", 0.0)
                    current_wheel_center = st.get(16) # Get position of a specific point. returns the point3 object
                    current_wheel_center_z = current_wheel_center[2]

                   # This saves the wheel center position at the start of the sweep
                    if i > 0:
                        instantaneous_travel_inches = (current_wheel_center_z - prev_wheel_center_z) / 25.4
                        delta_camber = current_camber - prev_camber
                        if abs(instantaneous_travel_inches) > 1e-4:
                            instantaneous_camber_rate = delta_camber / instantaneous_travel_inches
                            max_abs_camber_rate = max(max_abs_camber_rate, abs(instantaneous_camber_rate))                    


                    prev_wheel_center_z = current_wheel_center_z
                    prev_camber = current_camber



                all_sweep_metrics.append(metrics)
            
            #t_metrics = time.perf_counter() - t_start_metrics

            # Performance debug info (make sure to uncomment all time code for this to work)
            '''num_states = len(solution_states) if solution_states else 1
            print(f"\n--- Timing Breakdown ---")
            print(f"Deepcopy Time: {t_copy:.4f}s")
            print(f"Total Solve Time: {t_solve:.4f}s ({t_solve/num_states:.4f}s per state)")
            print(f"Metrics Loop Time: {t_metrics:.4f}s")'''

            '''self.count = self.count + 1
            print("Iteration Number:", self.count)'''

            f1 = static_scrub
            f2 = abs(static_camber)
            f3 = abs(static_toe)
            f4 = abs(static_kpi)
            f5 = abs(static_mech_trail)
            f6 = max_abs_camber_rate
            f7 = max_abs_toe_rate

            out["F"] = [f1, f2, f3, f4, f5, f6, f7]       

        except Exception as e:
            #print(f"Evaluation crashed: {type(e).__name__} - {e}")
            self.crashes = self.crashes + 1
            print("Solver Fail Count:", self.crashes)
            # punish bad results here
            out["F"] = [1e6] * self.n_obj


        
      
