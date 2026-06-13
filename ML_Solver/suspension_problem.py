import numpy as np
from pymoo.core.problem import ElementwiseProblem

class SuspensionProblem(ElementwiseProblem):
    def __init__(self):
        # number of variables being optimized, in x,y,z for each coordinate based hardpoint (since tyre data is fixed that doesnt count)
        n_var = 30
        # number of objectives (for example camber change, caster change, etc.)
        n_obj = 12
        # number of constraints, these are objectives the solver MUST achieve exactly
        n_con = 0

        # Initial hardpoints go here (x y and z from origin in mm)
        initial_hardpoints = np.array([1.0,2.0,3.0]) #placeholders for now
        # Define bounding boxes in the x, y and z direction
        xl = initial_hardpoints - 20
        xu = initial_hardpoints + 20
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_con, xl=xl, xu=xu)
    
    def _evaluate(self, x, out, *args, **kwargs):

        try:
            #evaluate here
        except Exception:
            # punish bad results here


        
      
