import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  

def plot_optimal_policy_with_holes(policy, desc, map_size, learner_name):
    # Ensure policy is a numpy array and reshape
    action_symbols = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    policy_symbols = np.array([action_symbols[action] for action in policy]).reshape(map_size, map_size)
    
    # Initialize the color grid
    color_grid = np.zeros((map_size, map_size))
    
    # Set up the figure
    plt.figure(figsize=(8, 8))
    plt.show()
    
    # Iterate over the description and policy to set color grid and annotations
    for row in range(map_size):
        for col in range(map_size):
            cell = desc[row][col].decode('utf-8') if isinstance(desc[row][col], bytes) else desc[row][col]
            if cell == 'S':  # Start
                color_grid[row, col] = 1
            elif cell == 'G':  # Goal
                color_grid[row, col] = 2
            elif cell == 'H':  # Hole
                color_grid[row, col] = -1
            else:
                color_grid[row, col] = 0.01  # Frozen cells

    # Plot the heatmap
    sns.heatmap(color_grid, annot=policy_symbols, fmt='s', cmap='coolwarm', cbar=False, linewidths=1, linecolor='black', square=True)
    plt.title(f"Optimal Policy for {learner_name}")
    plt.show()
    for i in range(map_size):
        for j in range(map_size):
            if color_grid[i][j] == -1:
                plt.annotate("H", xy=(j+0.5, i+0.5),
                    ha='center', va='center', color='black')
            elif color_grid[i][j] == 1:
                plt.annotate("S", xy=(j+0.5, i+0.5),
                    ha='center', va='center', color='black')
            elif color_grid[i][j] == 2:
                plt.annotate("G", xy=(j+0.5, i+0.5),
                    ha='center', va='center', color='black')
            elif color_grid[i][j] == 0.01:
                plt.annotate(str(policy_symbols[i][j]), xy=(j+0.5, i+0.5),
                        ha='center', va='center', color='black')
    plt.show()

def print_value_iteration_results(learner, map_size):
    # Print the value function
    print("Value Function:")
    print(np.array(learner.V).reshape(map_size, map_size))

    # Print the policy
    print("Policy (directions):")
    policy_directions = np.array(learner.policy).reshape(map_size, map_size)
    print(policy_directions)

    # Print the number of iterations
    print("Iterations:", learner.iter)

    # Print the convergence delta
    print("Convergence Delta:")
    deltas = [s['Error'] for s in learner.run_stats]
    print(deltas)

    # Plot the change in Value Function Delta if desired
    plt.figure(figsize=(10, 5))
    plt.plot(deltas)
    plt.title('Convergence of Value Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Delta')
    plt.grid(True)
    plt.show()
    
    # Optionally, you can also print the reward matrix if it's not too large
    if map_size <= 10:
        print("Reward Matrix:")
        print(np.array(learner.R).reshape(map_size, map_size))


def vi_alg_stats(learner, map_size):
    print_value_iteration_results(learner, map_size)
