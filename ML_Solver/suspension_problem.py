import numpy as np
import yaml
from pathlib import Path
from pymoo.core.problem import ElementwiseProblem
from kinematics.io.geometry_loader import load_geometry
from kinematics.io.sweep_loader import parse_sweep_file
from kinematics.main import solve_sweep
from kinematics.core.geometry import Point3

class SuspensionProblem(ElementwiseProblem):
    def __init__(self, geometry_path: str, sweep_path: str):


        #load the suspension and sweep (input) data
        self.suspension = load_geometry(geometry_path)
        self.sweep_config = parse_sweep_file(sweep_path)
        self.optimized_points = sorted(list(self.suspension.REQUIRED_POINTS), key=lambda p: p.value)
        # number of variables being optimized, in x,y,z for each coordinate based hardpoint (since tyre data is fixed that doesnt count)
        n_var = len(self.optimized_points) * 3
        # number of objectives (for example camber change, caster change, etc.)
        n_obj = 12
        # number of constraints, these are objectives the solver MUST achieve exactly
        n_con = 0
        flat_initial_coords = []
        for point_id in self.optimized_points:
            point3_obj = self.suspension.hardpoints[point_id]
            # Unpack X, Y, Z from the Point3 object and append them
            flat_initial_coords.extend([point3_obj.x, point3_obj.y, point3_obj.z])
            
        # Convert to a NumPy array (this will automatically be of length n_var!)
        initial_hardpoints = np.array(flat_initial_coords)
        
        # 5. Define bounding boxes (permitting +/- 20mm of movement from baseline design)
        xl = initial_hardpoints - 20.0
        xu = initial_hardpoints + 20.0
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_con, xl=xl, xu=xu)
    
    def _evaluate(self, x, out, *args, **kwargs):

        try:
            #update hardpoints
            for i, point_id in enumerate(self.optimized_points):
                start_idx = i * 3
                pt_x = x[start_idx]
                pt_y = x[start_idx + 1]
                pt_z = x[start_idx + 2]

                self.suspension.hardpoints[point_id] = Point3(pt_x, pt_y, pt_z)
            #evaluate here
            solution_states, solver_stats = solve_sweep(self.suspension, self.sweep_config)

            

        except Exception as e:
            # punish bad results here
            out["F"] = [1e6] * self.n_obj


        
      
