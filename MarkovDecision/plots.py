import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
np.random.seed(26)


# def plot_policy(policy, map_size):
#     """Plot the policy as arrows indicating the direction to move in each state."""
#     policy_arrows = ['←', '↓', '→', '↑']  # Assume these are the actions in order
#     policy_grid = np.chararray((map_size, map_size), unicode=True)
    
#     for s in range(map_size * map_size):
#         policy_grid[s // map_size, s % map_size] = policy_arrows[policy[s]]
    
#     fig, ax = plt.subplots()
#     sns.heatmap(np.zeros((map_size, map_size)), annot=policy_grid, fmt='', cbar=False, ax=ax)
#     ax.set_title('Optimal Policy')
#     plt.show()


def plot_policy(policy, map_size, terminal_states=None):
    """Plot the policy as arrows indicating the direction to move in each state."""
    policy_arrows = ['←', '↓', '→', '↑']  # Assume these are the actions in order
    policy_grid = np.chararray((map_size, map_size), unicode=True, itemsize=3)
    
    for s in range(map_size * map_size):
        # Check if it's a terminal state and should not have an arrow
        if terminal_states and s in terminal_states:
            policy_grid[s // map_size, s % map_size] = ''
        else:
            policy_grid[s // map_size, s % map_size] = policy_arrows[int(policy[s])]
    
    fig, ax = plt.subplots()
    sns.heatmap(np.zeros((map_size, map_size)), annot=policy_grid, fmt='', cbar=False, ax=ax)
    ax.set_title('Optimal Policy')
    plt.show()

# Example usage



def plot_value_function(V, map_size):
    """Plot the value function for each state."""
    value_grid = np.reshape(V, (map_size, map_size))
    
    fig, ax = plt.subplots()
    sns.heatmap(value_grid, annot=True, fmt='.2f', cmap='viridis', ax=ax)
    ax.set_title('Value Function')
    plt.show()

def plot_convergence(run_stats, title='Convergence Plot'):
    """Plot the convergence of the value function over iterations."""
    iterations = [stat['Iteration'] for stat in run_stats]
    errors = [stat['Error'] for stat in run_stats]

    plt.figure()
    plt.plot(iterations, errors)
    plt.xlabel('Iterations')
    plt.ylabel('Value Difference')
    plt.title(title)
    plt.show()


def find_terminal_states(env):
    terminal_states = []
    map_desc = env.desc.tolist()
    map_size = len(map_desc)
    
    for row in range(map_size):
        for col in range(map_size):
            if map_desc[row][col] in b'HG':  # We check for bytes literals
                # Convert the 2D position to a state index
                state = row * map_size + col
                terminal_states.append(state)
                
    return terminal_states


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_value_iteration_results(V, policy, env, map_size):
    """Plot the value function and policy obtained from Value Iteration."""
    # Convert V to a NumPy array if it's not already
    V = np.array(V) if not isinstance(V, np.ndarray) else V

    # Convert the value function to a 2D array for plotting
    V_reshaped = np.reshape(V, (map_size, map_size))

    # Map the policy to directions for visualization
    direction_map = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    policy_directions = np.array([direction_map[a] for a in policy]).reshape(map_size, map_size)

    # Plot the value function and policy
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render(mode='rgb_array'))  # Assuming the environment has a render mode that returns an image
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    # Plot the value function and policy
    sns.heatmap(
        V_reshaped,
        annot=policy_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Value Function and Policy\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")

    plt.show()
