import mlrose_hiive
import numpy as np
from search_algorithms import run_ro

np.random.seed(13) #set random seed for regeneration

# Declare range of random weights and values to test
weights = np.random.rand(300)
values = np.random.rand(300)
max_weight_pct = 1.0  # Maximum weight as a percentage of the total weight capacity

def run_knapsack():
    for r in range(25,150,25):
        weight=weights[:r]
        value=values[:r]
        print(f"length: {r}")

        # print(f"Weight: {weight}, Value: {value}")

        # knapsack problem
        ks_problem = mlrose_hiive.Knapsack(weight, value, max_weight_pct)
        problem = mlrose_hiive.DiscreteOpt(length = r, fitness_fn = ks_problem, maximize = True, max_val = 2)
        problem.set_mimic_fast_mode(True)

        run_ro(problem)


