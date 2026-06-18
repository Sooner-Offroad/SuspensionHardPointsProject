import pymoo 
import numpy as np

from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination import get_termination
from pymoo.parallelization.starmap import StarmapParallelization
from suspension_problem import SuspensionProblem


def main():
    geometry_yaml = r"C:\Users\adwai\Desktop\Skool\Local Git Repos\SuspensionHardPointsProject\ML_Solver\data\geometry.yaml"
    sweep_yaml = r"C:\Users\adwai\Desktop\Skool\Local Git Repos\SuspensionHardPointsProject\ML_Solver\data\sweep.yaml"

    # initialize the thread pool and create the runner
    #n_threads = 10
    #pool = ThreadPool(n_threads)
    n_procs = 10
    pool = Pool(n_procs)
    runner = StarmapParallelization(pool.starmap)

    problem = SuspensionProblem(
        geometry_path=geometry_yaml, 
        sweep_path=sweep_yaml,
        elementwise_runner=runner
    )

    # create the reference directions to be used for the optimization (NSGA only)
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
    
    # create the algorithm object
    algorithm = GA(pop_size=92,
                    ref_dirs=ref_dirs)
    
    
    termination = get_termination("n_gen", 100)

    print("Starting optimization loop...")
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1,          # Setting a seed makes your runs reproducible
        verbose=True     # This prints generation progress in your terminal
    )
    pool.close()
    # 6. Inspect the results
    print("\n--- Optimization Complete ---")
    print(f"Execution time: {res.exec_time:.2f} seconds")

    if problem.n_obj == 1:
        print(f"Best Objective Value Found (Scrub Radius): {res.F[0]:.4f} mm")
        print("Optimized Hardpoint Coordinates (Flattened Array):")
        print(res.X)
    else:
        print(f"Found {len(res.F)} optimal design trade-offs on the Pareto front.")
        # res.X will be a 2D array of shapes (num_designs, num_variables)
        # res.F will be a 2D array of shapes (num_designs, 12_objectives)

if __name__ == "__main__":
    main()


