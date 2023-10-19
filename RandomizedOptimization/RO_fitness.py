# Main file used to run algorithms with problems to generate graphs

import mlrose_hiive
import time 
from plots import fit_v_iteration, fevals_v_iteration
import numpy as np
import networkx as nx
from concurrent.futures import ProcessPoolExecutor

np.random.seed(42)

global_max_iters = float('inf')
global_max_attempts = 100

# 1. Randomized Hill Climb(SA)
def rhc(problem, restarts=5, max_iters=global_max_iters):
    return mlrose_hiive.random_hill_climb(problem, max_iters=max_iters, max_attempts=global_max_attempts, restarts=restarts, curve=True)

# 2. Simulated Annealing (SA)
def sa(problem, schedule=mlrose_hiive.GeomDecay(), max_attempts=global_max_attempts):
    return mlrose_hiive.simulated_annealing(problem, max_iters=global_max_iters, max_attempts=max_attempts, schedule=schedule,curve=True)

# 3. Genetic Algorithm (GA)
def ga(problem, mutation_prob=0.1, pop_size=200):
    return mlrose_hiive.genetic_alg(problem, max_iters=global_max_iters, max_attempts=global_max_attempts, mutation_prob=mutation_prob, pop_size=pop_size, curve=True)

# 4. MIMIC
def mimic(problem, pop_size=200, keep_pct=0.2):
    return mlrose_hiive.mimic(problem, max_iters=global_max_iters, max_attempts=global_max_attempts, pop_size=pop_size, keep_pct=keep_pct, curve=True)

def queens_max(state):
        fitness_cnt = 0
        for i in range(len(state) - 1):
            for j in range(i + 1, len(state)):
                if (state[j] != state[i]) \
                    and (state[j] != state[i] + (j - i)) \
                    and (state[j] != state[i] - (j - i)):
                    fitness_cnt += 1
        return fitness_cnt

def run_optimization(algorithm_results, algorithm_name, fitness_problem_name, hparams):
    best_state, best_fitness, best_curve = algorithm_results
    time_log = [0]  # Initialize time log with 0 as the first entry
    start_time = time.time()
    for i in range(len(best_curve)):
        time_log.append(time.time() - start_time)
    execution_time = time_log[-1] if time_log else 0
    print(f"Execution Time for {fitness_problem_name} {algorithm_name} {hparams}: {execution_time:.6f} seconds")
    # print(f"Final Fitness for {fitness_problem_name} {algorithm_name}: {best_fitness}")
    # print(f"Total Function Evaluations for {fitness_problem_name} {algorithm_name}: {len(best_curve)}")
    return best_fitness, best_curve, time_log

def run_rhc(problem, fitness_problem, minimize=False, restarts=[2,5,10,50], max_iters=[50,100, global_max_iters]):
    problem_size=150

    curves_list = []
    neg_curves_list=[]

    # change the restarts and max iters
    for r in restarts:
        param = (f"restarts: {r}")
        rhc_results = rhc(problem=problem, restarts=r)
        _, rhc_best_curve, rhc_time_log = run_optimization(rhc_results, "RHC", fitness_problem, param)
        curves_list.append({"curve": rhc_best_curve, "label": f'RHC restarts={r} max_iters=C', "time_log": rhc_time_log})
        if minimize:
            neg_curves_list.append({"curve": -rhc_best_curve, "label": f'RHC restarts={r} max_iters=C', "time_log": rhc_time_log})

    if minimize:
        fit_v_iteration(fitness_problem, problem_size, neg_curves_list, f'RHC size={problem_size} restarts tuning')
    else:
        fit_v_iteration(fitness_problem, problem_size, curves_list, f'RHC size={problem_size} restarts tuning')
    fevals_v_iteration(fitness_problem, problem_size, curves_list, f'RHC size={problem_size} restarts tuning')
    
    #reset
    curves_list = []
    neg_curves_list=[]
    
    for iteration in max_iters:
        param = (f"max_iters: {iteration}")
        rhc_results = rhc(problem=problem, max_iters=iteration)
        _, rhc_best_curve, rhc_time_log = run_optimization(rhc_results, "RHC", fitness_problem, param)
        curves_list.append({"curve": rhc_best_curve, "label": f'RHC restarts=C max_iters={iteration}', "time_log": rhc_time_log})
        if minimize:
            neg_curves_list.append({"curve": -rhc_best_curve, "label": f'RHC restarts=C max_iters={iteration}', "time_log": rhc_time_log})

    if minimize:
        fit_v_iteration(fitness_problem, problem_size, neg_curves_list, f'RHC size={problem_size} max_iters tuning')
    else:
        fit_v_iteration(fitness_problem, problem_size, curves_list, f'RHC size={problem_size} max_iters tuning')

    fevals_v_iteration(fitness_problem, problem_size, curves_list, f'RHC size={problem_size} max_iters tuning')
    # fevals_v_fitness_v_time(fitness_problem, problem_size, curves_list, f'size={problem_size} restarts tuning')

def run_sa(
        problem, 
        fitness_problem, 
        minimize=False, 
        schedule=[ 
            mlrose_hiive.ExpDecay(),
            mlrose_hiive.GeomDecay(),
            mlrose_hiive.ArithDecay(),
        ], 
        max_attempts=[50, 100, 150],
        problem_size=150
        ):
    curves_list = []
    neg_curves_list=[]
    
    # change the restarts and max iters
    for r in schedule:
        param = (f"schedule: {r.__class__.__name__}")
        sa_results = sa(problem=problem, schedule=r)
        _, sa_best_curve, sa_time_log = run_optimization(sa_results, "SA", fitness_problem, param)
        curves_list.append({"curve": sa_best_curve, "label": f'SA schedule={r.__class__.__name__}', "time_log": sa_time_log})
        if minimize:
            neg_curves_list.append({"curve": -sa_best_curve, "label": f'SA schedule={r.__class__.__name__}', "time_log": sa_time_log})

    if minimize:
        fit_v_iteration(fitness_problem, problem_size, neg_curves_list, f'SA size={problem_size} schedule tuning')
    else:
        fit_v_iteration(fitness_problem, problem_size, curves_list, f'SA size={problem_size} schedule tuning')
    fevals_v_iteration(fitness_problem, problem_size, curves_list, f'SA size={problem_size} schedule tuning')
    
    #reset
    curves_list = []
    neg_curves_list=[]
    
    for iteration in max_attempts:
        param = (f"max_attempts: {iteration}")
        sa_results = sa(problem=problem, max_attempts=iteration)
        _, sa_best_curve, sa_time_log = run_optimization(sa_results, "SA", fitness_problem, param)
        curves_list.append({"curve": sa_best_curve, "label": f'SA max_attempts={iteration}', "time_log": sa_time_log})
        if minimize:
            neg_curves_list.append({"curve": -sa_best_curve, "label": f'SA max_attempts={iteration}', "time_log": sa_time_log})

    if minimize:
        fit_v_iteration(fitness_problem, problem_size, neg_curves_list, f'SA size={problem_size} max_attempts tuning')
    else:
        fit_v_iteration(fitness_problem, problem_size, curves_list, f'SA size={problem_size} max_attempts tuning')

    fevals_v_iteration(fitness_problem, problem_size, curves_list, f'SA size={problem_size} max_attempts tuning')
    # fevals_v_fitness_v_time(fitness_problem, problem_size, curves_list, f'size={problem_size} restarts tuning')

def run_ga(
        problem, 
        fitness_problem, 
        minimize=False, 
        mutation_prob = [0.1,0.3,0.7], 
        pop_size= [50,100,200],
        problem_size=150
        ):
    curves_list = []
    neg_curves_list=[]
    
    # change the restarts and max iters
    for r in mutation_prob:
        param = (f"mutation_prob: {r}")
        ga_results = ga(problem=problem, mutation_prob=r)
        _, ga_best_curve, ga_time_log = run_optimization(ga_results, "GA", fitness_problem, param)
        curves_list.append({"curve": ga_best_curve, "label": f'GA mutation_prob={r}', "time_log": ga_time_log})
        if minimize:
            neg_curves_list.append({"curve": -ga_best_curve, "label": f'GA mutation_prob={r}', "time_log": ga_time_log})

    if minimize:
        fit_v_iteration(fitness_problem, problem_size, neg_curves_list, f'GA size={problem_size} mutation_prob tuning')
    else:
        fit_v_iteration(fitness_problem, problem_size, curves_list, f'GA size={problem_size} mutation_prob tuning')
    fevals_v_iteration(fitness_problem, problem_size, curves_list, f'GA size={problem_size} mutation_prob tuning')
    
    #reset
    curves_list = []
    neg_curves_list=[]
    
    for iteration in pop_size:
        param = (f"pop_size: {iteration}")
        ga_results = ga(problem=problem, pop_size=iteration)
        _, ga_best_curve, ga_time_log = run_optimization(ga_results, "GA", fitness_problem, param)
        curves_list.append({"curve": ga_best_curve, "label": f'GA pop_size={iteration}', "time_log": ga_time_log})
        if minimize:
            neg_curves_list.append({"curve": -ga_best_curve, "label": f'SA pop_size={iteration}', "time_log": ga_time_log})

    if minimize:
        fit_v_iteration(fitness_problem, problem_size, neg_curves_list, f'GA size={problem_size} pop_size tuning')
    else:
        fit_v_iteration(fitness_problem, problem_size, curves_list, f'GA size={problem_size} pop_size tuning')

    fevals_v_iteration(fitness_problem, problem_size, curves_list, f'GA size={problem_size} pop_size tuning')
    # fevals_v_fitness_v_time(fitness_problem, problem_size, curves_list, f'size={problem_size} restarts tuning')

def run_mimic(
        problem, 
        fitness_problem, 
        minimize=False, 
        keep_pct = [0.2,0.4,0.8], 
        pop_size= [50,100,200],
        problem_size=150
        ):
    curves_list = []
    neg_curves_list=[]
    
    # change the restarts and max iters
    for r in keep_pct:
        param = (f"keep_pct: {r}")
        mimic_results = mimic(problem=problem, keep_pct=r)
        _, mimic_best_curve, mimic_time_log = run_optimization(mimic_results, "mimic", fitness_problem, param)
        curves_list.append({"curve": mimic_best_curve, "label": f'mimic keep_pct={r}', "time_log": mimic_time_log})
        if minimize:
            neg_curves_list.append({"curve": -mimic_best_curve, "label": f'mimic keep_pct={r}', "time_log": mimic_time_log})

    if minimize:
        fit_v_iteration(fitness_problem, problem_size, neg_curves_list, f'MIMIC size={problem_size} keep_pct tuning')
    else:
        fit_v_iteration(fitness_problem, problem_size, curves_list, f'MIMIC size={problem_size} keep_pct tuning')
    fevals_v_iteration(fitness_problem, problem_size, curves_list, f'MIMIC size={problem_size} keep_pct tuning')
    
    #reset
    curves_list = []
    neg_curves_list=[]
    
    for iteration in pop_size:
        param = (f"pop_size: {iteration}")
        mimic_results = mimic(problem=problem, pop_size=iteration)
        _, mimic_best_curve, mimic_time_log = run_optimization(mimic_results, "mimic", fitness_problem, param)
        curves_list.append({"curve": mimic_best_curve, "label": f'mimic pop_size={iteration}', "time_log": mimic_time_log})
        if minimize:
            neg_curves_list.append({"curve": -mimic_best_curve, "label": f'mimic pop_size={iteration}', "time_log": mimic_time_log})

    if minimize:
        fit_v_iteration(fitness_problem, problem_size, neg_curves_list, f'MIMIC size={problem_size} pop_size tuning')
    else:
        fit_v_iteration(fitness_problem, problem_size, curves_list, f'MIMIC size={problem_size} pop_size tuning')

    fevals_v_iteration(fitness_problem, problem_size, curves_list, f'MIMIC size={problem_size} pop_size tuning')
    # fevals_v_fitness_v_time(fitness_problem, problem_size, curves_list, f'size={problem_size} restarts tuning')



def get_queens_problem():
    problem_size=150

    queens_problem = mlrose_hiive.CustomFitness(queens_max)
    problem = mlrose_hiive.DiscreteOpt(length = problem_size, fitness_fn = queens_problem, maximize = True, max_val = 2)
    return problem

def graph_coloring_fitness(state):
    # Initialize the number of conflicts to 0
    conflicts = 0

    # Iterate through each pair of adjacent nodes in the graph
    for node in G.nodes:
        for neighbor in G.neighbors(node):
            if state[node] == state[neighbor]:
                conflicts += 1

    return conflicts

G = nx.random_regular_graph(d=3, n=10)

def get_kcolor_problem():    
    problem = mlrose_hiive.DiscreteOpt(
        length=len(G.nodes),
        fitness_fn=mlrose_hiive.CustomFitness(graph_coloring_fitness),
        maximize=False,
        max_val=3  # The number of colors to use (e.g., 3 colors for a basic test)
    )
    return problem

# Declare range of random weights and values to test
weights = np.random.rand(300)
values = np.random.rand(300)
max_weight_pct = 1.0  # Maximum weight as a percentage of the total weight capacity
    
# knapsack problem
def get_knapsack_problem():
    problem_size=150
    fitness_problem="Knapsack"

    weight=weights[:problem_size]
    value=values[:problem_size]

    ks_problem = mlrose_hiive.Knapsack(weight, value, max_weight_pct)
    problem = mlrose_hiive.DiscreteOpt(length = problem_size, fitness_fn = ks_problem, maximize = True, max_val = 2)
    return problem

queens_fitness_problem="n-Queens"
kcolor_fitness_problem="k-Color"
knapsack_fitness_problem="Knapsack"

# start =time.time()
# run_rhc(get_queens_problem(), queens_fitness_problem, minimize=False)
# run_rhc(get_kcolor_problem(), kcolor_fitness_problem, minimize=True)
# run_rhc(get_knapsack_problem(), knapsack_fitness_problem, minimize=False)

# run_sa(get_queens_problem(), queens_fitness_problem, minimize=False)
# run_sa(get_kcolor_problem(), kcolor_fitness_problem, minimize=True)
# run_sa(get_knapsack_problem(), knapsack_fitness_problem, minimize=False)

# run_ga(get_queens_problem(), queens_fitness_problem, minimize=False)
# run_ga(get_kcolor_problem(), kcolor_fitness_problem, minimize=True)
# run_ga(get_knapsack_problem(), knapsack_fitness_problem, minimize=False)

# run_mimic(get_queens_problem(), queens_fitness_problem, minimize=False)
# run_mimic(get_kcolor_problem(), kcolor_fitness_problem, minimize=True)
# run_mimic(get_knapsack_problem(), knapsack_fitness_problem, minimize=False)
import os



if __name__ == '__main__':
    start =time.time()

    tasks = [
        # (run_rhc, (get_queens_problem(), queens_fitness_problem, False), {}),
        # (run_rhc, (get_kcolor_problem(), kcolor_fitness_problem, True), {}),
        # (run_rhc, (get_knapsack_problem(), knapsack_fitness_problem, False), {}),
        
        # (run_sa, (get_queens_problem(), queens_fitness_problem, False), {}),
        # (run_sa, (get_kcolor_problem(), kcolor_fitness_problem, True), {}),
        # (run_sa, (get_knapsack_problem(), knapsack_fitness_problem, False), {}),
        
        # (run_ga, (get_queens_problem(), queens_fitness_problem, False), {}),
        # (run_ga, (get_kcolor_problem(), kcolor_fitness_problem, True), {}),
        # (run_ga, (get_knapsack_problem(), knapsack_fitness_problem, False), {}),
        
        (run_mimic, (get_queens_problem(), queens_fitness_problem, False), {}),
        (run_mimic, (get_kcolor_problem(), kcolor_fitness_problem, True), {}),
        (run_mimic, (get_knapsack_problem(), knapsack_fitness_problem, False), {})
    ]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(fn, *args, **kwargs) for fn, args, kwargs in tasks]
        results = [f.result() for f in futures]
    end = time.time() - start
    print(f"took {end}s to run.")
