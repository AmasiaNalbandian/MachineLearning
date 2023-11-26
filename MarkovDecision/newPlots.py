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
    plt.figure(figsize=(10, 10))

    for row in range(map_size):
        for col in range(map_size):
            cell = desc[row][col].decode('utf-8') if isinstance(desc[row][col], bytes) else desc[row][col]
            if cell == 'H':  # Hole
                policy_symbols[row][col] = ""  # No arrow for holes
                color_grid[row, col] = -1
            elif cell == 'S':  # Start
                color_grid[row, col] = 1
            elif cell == 'G':  # Goal
                color_grid[row, col] = 2
            else:
                color_grid[row, col] = 0.01  # Frozen cells
    # Plot the heatmap
    sns.heatmap(color_grid, annot=policy_symbols, fmt='s', cmap='coolwarm', cbar=False, linewidths=1, linecolor='black', square=True)
    plt.title(f"Optimal Policy for {learner_name}")
    for i in range(map_size):
        for j in range(map_size):
            if color_grid[i][j] == -1:
                plt.annotate("H", xy=(j+0.5, i+0.5),
                    ha='center', va='center', color='black')
            # elif color_grid[i][j] == 1:
            #     plt.annotate("S", xy=(j+0.5, i+0.5),
            #         ha='center', va='center', color='black')
            elif color_grid[i][j] == 2:
                plt.annotate("G", xy=(j+0.5, i+0.5),
                    ha='center', va='center', color='black')
            elif color_grid[i][j] == 0.01:
                plt.annotate(str(policy_symbols[i][j]), xy=(j+0.5, i+0.5),
                        ha='center', va='center', color='black')
    plt.show()

def print_value_iteration_results(learner, map_size, learner_name, debug = False):
    # Print the value function
    print("Value Function:") if debug else None
    print(np.array(learner.V).reshape(map_size, map_size)) if debug else None

    # Print the policy
    print("Policy (directions):") if debug else None
    policy_directions = np.array(learner.policy).reshape(map_size, map_size)
    print(policy_directions) if debug else None

    # Print the number of iterations
    print("Iterations:", learner.iter) if debug else None

    # Print the convergence delta
    print("Convergence Delta:") if debug else None
    deltas = [s['Error'] for s in learner.run_stats]
    print(deltas) if debug else None

    # Plot the change in Value Function Delta if desired
    plt.figure(figsize=(10, 6))
    plt.plot(deltas)
    plt.title(f'Convergence of {learner_name} Value Function')
    plt.xlabel('Iteration')
    plt.ylabel('Delta')
    plt.grid(True)
    plt.show()
    

import numpy as np
import matplotlib.pyplot as plt

def print_delta_algorithm_results(learners, map_size, learner_names, debug=False):
    plt.figure(figsize=(10, 6))

    for idx, learner in enumerate(learners):
        learner_name = learner_names[idx]

        # Print the value function
        if debug:
            print(f"Value Function for {learner_name}:")
            try:
                print(np.array(learner.V).reshape(map_size, map_size))
            except AttributeError:
                # Handle differently for learners like Q-learning
                print(f"Value function not directly available for {learner_name}.")

        # Print the policy
        if debug:
            print(f"Policy (directions) for {learner_name}:")
            try:
                policy_directions = np.array(learner.policy).reshape(map_size, map_size)
                print(policy_directions)
            except AttributeError:
                # Handle differently for learners like Q-learning
                print(f"Policy not directly available for {learner_name}.")

        # Print the number of iterations
        if debug:
            print(f"Iterations for {learner_name}:", getattr(learner, 'iter', 'N/A'))

        # Plot the convergence delta
        if hasattr(learner, 'run_stats'):
            deltas = [s['Error'] for s in learner.run_stats]
            if debug:
                print(f"Convergence Delta for {learner_name}:")
                print(deltas)

            plt.plot(deltas, label=learner_name)

    plt.title('Convergence Comparison of Learning Algorithms')
    plt.xlabel('Iteration')
    plt.ylabel('Delta')
    plt.grid(True)
    plt.legend()
    plt.show()


    