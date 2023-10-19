import mlrose_hiive
import numpy as np
from search_algorithms import run_ro
from plots import timings_v_problem_size

np.random.seed(13) #set random seed for regeneration

# Declare range of random weights and values to test
weights = np.random.rand(300)
values = np.random.rand(300)
max_weight_pct = 1.0  # Maximum weight as a percentage of the total weight capacity

def run_knapsack():
    timings = {
        "RHC": [],
        "SA": [],
        "GA": [],
        "MIMIC": []
    }
    sizes = list(range(25, 200, 25))



    for r in sizes:
        weight=weights[:r]
        value=values[:r]
        print(f"length: {r}")

        # print(f"Weight: {weight}, Value: {value}")

        # knapsack problem
        ks_problem = mlrose_hiive.Knapsack(weight, value, max_weight_pct)
        problem = mlrose_hiive.DiscreteOpt(length = r, fitness_fn = ks_problem, maximize = True, max_val = 2)
        problem.set_mimic_fast_mode(True)

        curves_list = run_ro(problem, "Knapsack", r)
        for curve_info in curves_list:
            timings[curve_info["label"]].append(sum(curve_info["time_log"]))
    timings_v_problem_size("Knapsack", sizes, timings)

