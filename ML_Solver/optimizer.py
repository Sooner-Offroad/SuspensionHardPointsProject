import pymoo 
import numpy as np

from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination import get_termination
from pymoo.parallelization.starmap import StarmapParallelization
from suspension_problem import SuspensionProblem
from pymoo.mcdm.pseudo_weights import PseudoWeights


def main():

    # Description: Hardpoint Optimizer that utilizes open kinematics and pymoo (NSGA algorithm) to optimize suspension hardpoint locations.
    #              Writes solutions to a text file, that will be saved in the current directory. Use cd followed by the folder location to 
    #              change where the results file is saved.
    # CREDITS: 
    #          1. Nick McCleery for the kinematics package, https://github.com/nickmccleery/open-kinematics , under Apache License Version 2.0
    #          2. J. Blank and K. Deb, pymoo: Multi-Objective Optimization in Python, in IEEE Access, vol. 8, pp. 89497-89509, 2020, doi: 10.1109/ACCESS.2020.2990567, https://github.com/anyoptimization/pymoo

    # ===========================
    # ======== VARIABLES ========
    # ===========================

    # Set the location to wherever your geometry yaml and sweep files are. VERY IMPORTANT. CODE WILL BREAK IF YOU DO NOT UPDATE THIS. MAKE SURE TO HAVE AN r before the string.
    geometry_yaml = r"C:\Users\adwai\Desktop\Skool\Local Git Repos\SuspensionHardPointsProject\ML_Solver\data\geometry.yaml" 
    sweep_yaml = r"C:\Users\adwai\Desktop\Skool\Local Git Repos\SuspensionHardPointsProject\ML_Solver\data\sweep.yaml"

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

    # Labels for objectives in order. Editing this will only change the labels, similar to above.
    OBJECTIVE_LABELS = ["Static Scrub", "Static Camber", "Static Toe", "Static Kingpin Inclination", "Static Mechanical Trail", "Max Camber Rate", "Max Toe Rate"]

    # Values for objectives in order. Changing this WILL affect the values of the objectives, in the order given in suspension_problem. Units mm and degrees, except camber and toe rate which are degrees/inch.
    OBJECTIVE_VALUES = [0, 0, 0, 10, 38.1, 0.5, 0]

    # Weights in same order
    weights = np.array([0.22, 0.22, 0.22, 0.04, 0.04, 0.22, 0.04])

    # Length of side of cube in which optimizer will search for each point.
    cube_side_length_mm = 30 #mm

    # Number of generations
    gen_amount = 100

    n_procs = 20 # number of logical processors, check how many you have available before running, using cntl + shift + esc and checking cpu logical processors amount.
    #run about 3-4 processors lower than what is available. You can run max but your cpu will be fully occupied. The lower the number of cores, the longer it will take.

    # create the reference directions to be used for the optimization (NSGA and its variants only) First argument is the algorithm, second is the number of directions (objectives), third is the number of samples.
    ref_dirs = get_reference_directions("energy", 7, 100)  

    # The algorithm the optimizer will use. Note that changing this may require the main code to be updated if the new algorithm has any new parameters.
    algorithm = UNSGA3(ref_dirs=ref_dirs)

    # How the results are written to the file. "w" will clear the file each run and output the new points only, "a" will append the new points to the end of the file and not clear old results.
    writemode = "a"

    # ===========================
    # ======== MAIN CODE ========
    # ===========================

    # Set up CPU parallelization
    pool = Pool(n_procs)
    runner = StarmapParallelization(pool.starmap)

    # Create the termination condition
    termination = get_termination("n_gen", gen_amount)

    # Create the problem to be run in the optimizer
    problem = SuspensionProblem(
        geometry_path=geometry_yaml, 
        sweep_path=sweep_yaml,
        cube_side_length_mm=cube_side_length_mm,
        objective_values=OBJECTIVE_VALUES,
        elementwise_runner=runner
    )
    

    print("Starting optimization loop...")

    #  Run the optimizer
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1,          # Setting a seed makes your runs reproducible
        verbose=True     # This prints generation progress in your terminal
    )

    # Close the pool
    pool.close()
    pool.join()

    # ===========================
    # ====== POST-PROCESS =======
    # ===========================

    # Get the Pareto front (set of best points) and associated metrics
    points_pareto = res.X
    objectives_pareto = res.F

    # Get the id of the best points according to weights
    best_idx = PseudoWeights(weights).do(objectives_pareto)

    # Get the best point according to weights
    best_suspension_geometry = points_pareto[best_idx]
    best_suspension_metrics = objectives_pareto[best_idx]


    # Print results to file. Includes entire pareto front, and the best overall point (bottom of file) 
    with open("results.txt", writemode) as f:
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
    print("Units: millimeters and degrees. Angle rates are in degrees per inch.")


if __name__ == "__main__":
    main()


