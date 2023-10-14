import argparse
from nqueens import run_queens
from knapsack import run_knapsack
from kcolors import run_kcolors

# Create a parser for command-line arguments
parser = argparse.ArgumentParser(description='Solve optimization problems.')
parser.add_argument('problem', type=str, choices=['queens', 'knapsack', 'kcolor', 'all'], help='Name of the problem to solve (or "all" for all problems)')

# Parse the command-line arguments
args = parser.parse_args()
# Execute the corresponding code based on the argument provided
if args.problem == 'queens' or args.problem == 'all':
    print("Running queens problem-----------------------------------")
    run_queens()

if args.problem == 'knapsack' or args.problem == 'all':
    print("Running knapsack problem-----------------------------------")
    run_knapsack()

if args.problem == 'kcolor' or args.problem == 'all':
    print("Running k-color problem-----------------------------------")
    run_kcolors()
