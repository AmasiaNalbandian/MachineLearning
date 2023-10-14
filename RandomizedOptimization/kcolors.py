import mlrose_hiive
import numpy as np
from search_algorithms import run_ro
import networkx as nx

np.random.seed(13) #set random seed for regeneration

# Create a random graph for testing
G = nx.random_regular_graph(d=3, n=10)

# Define a fitness function for the k-coloring problem
# Copied from: ChatGPT
def graph_coloring_fitness(state):
    # Initialize the number of conflicts to 0
    conflicts = 0

    # Iterate through each pair of adjacent nodes in the graph
    for node in G.nodes:
        for neighbor in G.neighbors(node):
            if state[node] == state[neighbor]:
                conflicts += 1

    return conflicts

# Initialize custom fitness function object


def run_kcolors():
    for r in range(25,150,25):
        print(f"length: {r}")
        problem = mlrose_hiive.DiscreteOpt(
            length=len(G.nodes),
            fitness_fn=mlrose_hiive.CustomFitness(graph_coloring_fitness),
            maximize=False,
            max_val=3  # The number of colors to use (e.g., 3 colors for a basic test)
        )
        problem.set_mimic_fast_mode(True)
        
        run_ro(problem)




