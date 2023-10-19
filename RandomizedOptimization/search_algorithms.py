import mlrose_hiive
import time 
from plots import fit_v_iteration, fevals_v_iteration
# In this file, there are four local random search algorithms which are: 
# 1. Randomized Hell Climbing (RHC)
# 2. Simulated Annealing (SA)
# 3. Genetic Algorithm (GA)
# 4. MIMIC

# Hyperparameter tuning:
# global_max_iters = 25
global_max_iters = float('inf')
global_max_attempts = 100

# Define a function to run an optimization algorithm and measure time
def run_optimization(problem, algorithm, algorithm_name):
    time_log = []
    start_time = time.time()
    best_state, best_fitness, best_curve = algorithm(problem)
    for i in range(len(best_curve)):
        time_log.append(time.time() - start_time)
    execution_time = time_log[-1] if time_log else 0
    print(f"Execution Time for {algorithm_name}: {execution_time:.6f} seconds")
    final_fevals = len(best_curve)
    # print(f"Final Fitness for {algorithm_name}: {best_fitness}")
    # print(f"Total Function Evaluations for {algorithm_name}: {final_fevals}")

    return best_fitness, best_curve, time_log

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
def run_ro(problem, fitness_problem, problem_size):
    rhc_best_fitness, rhc_best_curve, rhc_time_log = run_optimization(problem, rhc, "RHC")
    sa_best_fitness, sa_best_curve, sa_time_log = run_optimization(problem, sa, "SA")
    ga_best_fitness, ga_best_curve, ga_time_log = run_optimization(problem, ga, "GA")
    mimic_best_fitness, mimic_best_curve, mimic_time_log = run_optimization(problem, mimic, "MIMIC")

    curves_list = [
        { "curve": rhc_best_curve, "label": "RHC", "time_log": rhc_time_log },
        { "curve": sa_best_curve, "label": "SA", "time_log": sa_time_log },
        { "curve": ga_best_curve, "label": "GA", "time_log": ga_time_log },
        { "curve": mimic_best_curve, "label": "MIMIC", "time_log": mimic_time_log }
    ]
    fit_v_iteration(fitness_problem, problem_size, curves_list)
    # fevals_vs_time(fitness_problem, problem_size, rhc_best_curve, sa_best_curve, ga_best_curve, mimic_best_curve, rhc_time_log, sa_time_log, ga_time_log, mimic_time_log)
    fevals_v_iteration(fitness_problem, problem_size, curves_list)
    return curves_list

    