import mlrose_hiive
import numpy as np
from search_algorithms import run_ro
from plots import timings_v_problem_size

np.random.seed(13) #set random seed for regeneration

# Define alternative N-Queens fitness function for maximization problem
# Copied from: https://mlrose.readthedocs.io/en/stable/source/tutorial1.html
def queens_max(state):

    # Initialize counter
        fitness_cnt = 0

        # For all pairs of queens
        for i in range(len(state) - 1):
            for j in range(i + 1, len(state)):

                # Check for horizontal, diagonal-up and diagonal-down attacks
                if (state[j] != state[i]) \
                    and (state[j] != state[i] + (j - i)) \
                    and (state[j] != state[i] - (j - i)):

                        # If no attacks, then increment counter
                    fitness_cnt += 1

        return fitness_cnt

# Initialize custom fitness function object


def run_queens():
    timings = {
        "RHC": [],
        "SA": [],
        "GA": [],
        "MIMIC": []
    }
    sizes = list(range(25, 200, 25))



    for r in sizes:
        print(f"length: {r}")

        # knapsack problem
        queens_problem = mlrose_hiive.CustomFitness(queens_max)
        problem = mlrose_hiive.DiscreteOpt(length = r, fitness_fn = queens_problem, maximize = True, max_val = 2)
        problem.set_mimic_fast_mode(True)
        
        curves_list = run_ro(problem, "n-Queens", r)
        for curve_info in curves_list:
            timings[curve_info["label"]].append(sum(curve_info["time_log"]))
    timings_v_problem_size("n-Queens", sizes, timings)


