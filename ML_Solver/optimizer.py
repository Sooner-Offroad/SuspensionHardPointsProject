import pymoo 
import numpy as np

from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination import get_termination
from pymoo.parallelization.starmap import StarmapParallelization
from suspension_problem import SuspensionProblem
from pymoo.mcdm.pseudo_weights import PseudoWeights


def main():

    # Labels for points and objectives in order. Changing this will only change the labels, you need to go into suspension.py to actually change the objectives/points
    HARDPOINTS = [
    "\nLOWER_WISHBONE_INBOARD_FRONT",
    "\nLOWER_WISHBONE_INBOARD_REAR",
    "\nLOWER_WISHBONE_OUTBOARD",
    "\nUPPER_WISHBONE_INBOARD_FRONT",
    "\nUPPER_WISHBONE_INBOARD_REAR",
    "\nUPPER_WISHBONE_OUTBOARD",
    "\nTRACKROD_INBOARD",
    "\nTRACKROD_OUTBOARD",
    "\nAXLE_INBOARD",
    "\nAXLE_OUTBOARD",
    ]

    DESIGN_LABELS = [f"{point}_{axis}" for point in HARDPOINTS for axis in ["X", "Y", "Z"]] 
    OBJECTIVE_LABELS = ["Static Scrub", "Camber", "Toe", "Kingpin Inclination", "Mechanical Trail", "Camber Rate", "Toe Rate"]

    # Values for objectives in order. Changing this WILL affect the values of the objectives, in the order given in suspension_problem. Units mm and degrees.
    OBJECTIVE_VALUES = [0, 0, 0, 10, 38.1, 0, 0]
    # Weights in same order
    weights = np.array([0.22, 0.22, 0.22, 0.04, 0.04, 0.22, 0.04])
    # Length of side of cube in which optimizer will search for each point.
    cube_side_length_mm = 30 #mm

    # Set the location to wherever your geometry yaml and sweep files are. VERY IMPORTANT
    geometry_yaml = r"C:\Users\adwai\Desktop\Skool\Local Git Repos\SuspensionHardPointsProject\ML_Solver\data\geometry.yaml" 
    sweep_yaml = r"C:\Users\adwai\Desktop\Skool\Local Git Repos\SuspensionHardPointsProject\ML_Solver\data\sweep.yaml"

    n_procs = 20 # number of logical processors, check how many you have available before running, using cntl + shift + esc and checking cpu logical processors amount.
    #run about 3-4 processors lower than what is available. You can run max but your cpu will be fully occupied. The lower the number of cores, the longer it will take.
    pool = Pool(n_procs)
    runner = StarmapParallelization(pool.starmap)

    problem = SuspensionProblem(
        geometry_path=geometry_yaml, 
        sweep_path=sweep_yaml,
        cube_side_length_mm=cube_side_length_mm,
        objective_values=OBJECTIVE_VALUES,
        elementwise_runner=runner
    )

    # create the reference directions to be used for the optimization (NSGA and its variants only)
    ref_dirs = get_reference_directions("energy", 7, 100)
    
    # create the algorithm object
    algorithm = UNSGA3(ref_dirs=ref_dirs)
    
    # increase the number here if you want it to run more generations
    termination = get_termination("n_gen", 2)

    # Loop code starts here
    print("Starting optimization loop...")
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1,          # Setting a seed makes your runs reproducible
        verbose=True     # This prints generation progress in your terminal
    )

    pool.close()
    pool.join()
    points_pareto = res.X
    objectives_pareto = res.F

    best_idx = PseudoWeights(weights).do(objectives_pareto)

    best_suspension_geometry = points_pareto[best_idx]
    best_suspension_metrics = objectives_pareto[best_idx]


    # Print to file
    with open("results.txt", "a") as f:
        f.write(f"=== Entire Pareto Front ({len(points_pareto)} Configurations) ===\n")
        

        for i, design in enumerate(points_pareto):
            # 2. Pair each label with its corresponding value and format to 2 decimal places
            design_pairs = [f"{label}: {val:.2f}" for label, val in zip(DESIGN_LABELS, design)]
            pareto_points_string = ", ".join(design_pairs)
            
            objective_pairs = [f"{label}: {val:.2f}" for label, val in zip(OBJECTIVE_LABELS, objectives_pareto[i])]
            objectives_string = ", ".join(objective_pairs)
            
            # 3. Write the clearly labeled line to your file
            f.write(f"Design [{pareto_points_string}] -> Objective Deltas (HOW FAR OFF EACH OBJECTIVE IS FROM OPTIMAL) [{objectives_string}]\n")

        best_design_string = ", ".join([f"{label}: {val:.2f}" for label, val in zip(DESIGN_LABELS, best_suspension_geometry)])
        best_objectives_string = ", ".join([f"{label}: {val:.2f}" for label, val in zip(OBJECTIVE_LABELS, best_suspension_metrics)])

        f.write("\n" + "="*50 + "\n\n")
        f.write("=== Optimization Run Details ===\n")
        f.write("Best Suspension Points (Coordinates):\n")
        f.write(best_design_string)
        f.write("\n\n")
        f.write("Best Suspension Objective Values (Distance to target, not direct values):\n")
        f.write(best_objectives_string)
        f.write("\n" + "="*30 + "\n\n")

    
    # Print to terminal. Only best points.
    print("\n--- Optimization Complete ---")
    print(f"Execution time: {res.exec_time:.2f} seconds")
    print(f"Best Suspension Points (Coordinates):\n{best_design_string}")
    print(f"Best Suspension Objective Values (Deltas):\n{best_objectives_string}")
    print("Units: millimeters and degrees.")


if __name__ == "__main__":
    main()


