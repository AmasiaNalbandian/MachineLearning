import mlrose_hiive
import time
def rhc(problem):
    # Use an optimization algorithm to find the best solution for this specific combination
    start_t = time.time()
    rhc_best_state, rhc_best_fitness, _ = mlrose_hiive.random_hill_climb(problem, restarts=5, curve = True)
    end_t = time.time() - start_t
    print(f"Execution Time for RHC: {end_t:.2f} seconds")
    print(f"Best State: {rhc_best_state}")
    print(f"Best Fitness: {rhc_best_fitness}")

# 2. Simulated Annealing (SA)
def sa(problem):
    max_attempts=100
    start_t = time.time()
    sa_best_state, sa_best_fitness, _ = mlrose_hiive.simulated_annealing(problem, max_attempts = max_attempts, curve = True)
    end_t = time.time() - start_t
    print(f"Execution Time for RHC: {end_t:.2f} seconds")
    print(f"Best State: {sa_best_state}")
    print(f"Best Fitness: {sa_best_fitness}")

# 3. Genetic Algorithm (GA)
def ga(problem):
    max_attempts=100
    max_iters=10
    start_t = time.time()
    ga_best_state, ga_best_fitness, _ = mlrose_hiive.genetic_alg(problem, pop_size = 10, max_attempts = 10, curve = True)
    end_t = time.time() - start_t
    print(f"Execution Time for RHC: {end_t:.2f} seconds")
    print(f"Best State: {ga_best_state}")
    print(f"Best Fitness: {ga_best_fitness}")

# 4. MIMIC
def mimic(problem):
    max_attempts=100
    max_iters=10
    start_t = time.time()
    mimic_best_state, mimic_best_fitness, _ = mlrose_hiive.mimic(problem, pop_size = 10, max_attempts = 20, curve = True)
    end_t = time.time() - start_t
    print(f"Execution Time for RHC: {end_t:.2f} seconds")
    print(f"Best State: {mimic_best_state}")
    print(f"Best Fitness: {mimic_best_fitness}")