import mlrose_hiive
import time 

# In this file, there are four local random search algorithms which are: 
# 1. Randomized Hell Climbing (RHC)
# 2. Simulated Annealing (SA)
# 3. Genetic Algorithm (GA)
# 4. MIMIC

# Hyperparameter tuning:
global_max_iters = 100
global_max_attempts = 20

# Define a function to run an optimization algorithm and measure time
def run_optimization(problem, algorithm, algorithm_name):
    start_time = time.time()
    best_state, best_fitness, _ = algorithm(problem)
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution Time for {algorithm_name}: {execution_time:.6f} seconds")
    # print(f"Best State: {best_state}")
    # print(f"Best Fitness: {best_fitness}")

# 1. Randomized Hill Climbing (RHC)
def rhc(problem):
    return mlrose_hiive.random_hill_climb(problem, max_iters=global_max_iters, max_attempts=global_max_attempts, restarts=5, curve=True)

# 2. Simulated Annealing (SA)
def sa(problem):
    return mlrose_hiive.simulated_annealing(problem, max_iters=global_max_iters, max_attempts=global_max_attempts, curve=True)

# 3. Genetic Algorithm (GA)
def ga(problem):
    return mlrose_hiive.genetic_alg(problem, max_iters=global_max_iters, pop_size=10, max_attempts=global_max_attempts, curve=True)

# 4. MIMIC
def mimic(problem):
    return mlrose_hiive.mimic(problem, max_iters=global_max_iters, pop_size=10, max_attempts=global_max_attempts, curve=True)

# Run each optimization algorithm
def run_ro(problem):
    run_optimization(problem, rhc, "RHC")
    run_optimization(problem, sa, "SA")
    run_optimization(problem, ga, "GA")
    run_optimization(problem, mimic, "MIMIC")
