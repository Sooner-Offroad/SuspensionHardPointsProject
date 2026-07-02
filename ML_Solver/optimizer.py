import pymoo 
import numpy as np

from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination import get_termination
from pymoo.parallelization.starmap import StarmapParallelization
from suspension_problem import SuspensionProblem
from pymoo.mcdm.pseudo_weights import PseudoWeights


def main():
    geometry_yaml = r"C:\Users\adwai\Desktop\Skool\Local Git Repos\SuspensionHardPointsProject\ML_Solver\data\geometry.yaml" 
    sweep_yaml = r"C:\Users\adwai\Desktop\Skool\Local Git Repos\SuspensionHardPointsProject\ML_Solver\data\sweep.yaml"

    n_procs = 20
    pool = Pool(n_procs)
    runner = StarmapParallelization(pool.starmap)

    problem = SuspensionProblem(
        geometry_path=geometry_yaml, 
        sweep_path=sweep_yaml,
        elementwise_runner=runner
    )

    # create the reference directions to be used for the optimization (NSGA only)
    ref_dirs = get_reference_directions("energy", 7, 100)
    
    # create the algorithm object
    algorithm = UNSGA3(ref_dirs=ref_dirs)
    
    
    termination = get_termination("n_gen", 250)

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

    weights = np.array([0.22, 0.22, 0.22, 0.22, 0.04, 0.04, 0.04])
    best_idx = PseudoWeights(weights).do(objectives_pareto)

    best_suspension_geometry = points_pareto[best_idx]
    best_suspension_metrics = objectives_pareto[best_idx]

    print("\n--- Optimization Complete ---")
    print(f"Execution time: {res.exec_time:.2f} seconds")
    print(f"Best Suspension Points (Coordinates):\n{best_suspension_geometry}")
    print(f"Best Suspension Objective Values (Deltas):\n{best_suspension_metrics}")

    with open("results.txt", "a") as f:
        f.write("=== Optimization Run Details ===\n")
        f.write("Best Suspension Points (Coordinates):\n")
        f.write(np.array2string(best_suspension_geometry, precision=3, separator=', '))
        f.write("\n\n")
        f.write("Best Suspension Objective Values (Deltas):\n")
        f.write(np.array2string(best_suspension_metrics, precision=4, separator=', '))
        f.write("\n" + "="*30 + "\n\n")


if __name__ == "__main__":
    main()


