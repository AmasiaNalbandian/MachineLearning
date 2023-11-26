import matplotlib.pyplot as plt
from mdptoolbox.mdp import ValueIteration, PolicyIteration

import gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from frozenlakes import MyMDPConverter
import gymnasium as gym
from pathlib import Path
from frozenlakes import Params
import numpy as np
import pandas as pd

params = Params(
    total_episodes=2000,
    learning_rate=0.01,
    gamma=0.9,
    epsilon=1,
    seed=26,
    is_slippery=True,
    n_runs=800000,
    action_size=None,
    state_size=None,
    proba_frozen=0.9,
    savefig_folder=Path("./_static/img/tutorials/"),
)
np.random.seed(params.seed)
rng = np.random.default_rng(params.seed)
map_sizes = [16]
res_all = pd.DataFrame()
st_all = pd.DataFrame()

# Create the figure folder if it doesn't exists
params.savefig_folder.mkdir(parents=True, exist_ok=True)

map_size=20
def setup_env():

    # Create the Frozen Lake environment
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=params.is_slippery,
        render_mode="rgb_array",
        desc=generate_random_map(
            size=map_size, p=params.proba_frozen, seed=params.seed
        ),
    )

    env.action_space.seed(
        params.seed
    )
    env.reset()

    # Access the desc attribute
    lake_map = env.unwrapped.desc

    # Initialize and use the converter
    converter = MyMDPConverter(env)
    converter.convert_PR()

    # Function to convert 2D position to state index
    def pos_to_state(row, col, size):
        return row * size + col

    # Modify the R matrix for custom rewards
    for row in range(lake_map.shape[0]):
        for col in range(lake_map.shape[1]):
            state = pos_to_state(row, col, lake_map.shape[0])
            
            if lake_map[row, col] == b'H':
                converter.R[state, :] = -1  # Penalty for falling into a hole
            elif lake_map[row, col] == b'G':
                converter.R[state, :] = 1   # Reward for reaching the goal
            else:
                converter.R[state, :] = -0.01  # Step penalty for other states

    # Now you can use converter.P and converter.R with MDPToolbox algorithms
    P = converter.P
    R = converter.R

    return env, P, R


# Modify the pos_to_state function as needed for your environment
def pos_to_state(row, col, size):
    return row * size + col

def evaluate_policy_quality(env, policy, size):
    total_reward = 0
    num_episodes = 5

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = policy[state]
            next_state, reward, done, _, _ = env.step(action)  # Adjusted unpacking here
            state = next_state  # Update state for the next iteration

            total_reward += reward

    average_reward = total_reward / num_episodes
    return average_reward

map_sizes = [4, 8, 12, 16, 20]  # Example different map sizes
convergence_speeds = {'VI': [], 'PI': []}
policy_qualities = {'VI': [], 'PI': []}

for map_size in map_sizes:
    print("Map size:", map_size)  # Add this line in the loop before calling evaluate_policy_quality

    env, P, R = setup_env()  # Your setup_env function, adjusted for different map_sizes
    state = env.reset()
    print("State after reset:", state)  # Add this line

    # Run Value Iteration
    vi = ValueIteration(P, R, params.gamma)
    vi.run()
    convergence_speeds['VI'].append(vi.iter)
    policy_quality = evaluate_policy_quality(env, vi.policy, map_size)
    policy_qualities['VI'].append(policy_quality)
    # Evaluate policy quality and add to policy_qualities['VI']

    # Run Policy Iteration
    pi = PolicyIteration(P, R, params.gamma)
    pi.run()
    convergence_speeds['PI'].append(pi.iter)
    policy_quality = evaluate_policy_quality(env, pi.policy, map_size)
    policy_qualities['PI'].append(policy_quality)

    # Evaluate policy quality and add to policy_qualities['PI']


# Assuming you have data collected in the following format
# state_sizes = [size1, size2, ...]
# convergence_speeds = {'VI': [speed1, speed2, ...], 'PI': [...], 'QL': [...]}
# policy_qualities = {'VI': [quality1, quality2, ...], 'PI': [...], 'QL': [...]}
print("starting to plot")
plt.figure(figsize=(12, 5))

# Plot for Convergence Speed
plt.subplot(1, 2, 1)
for algo in convergence_speeds:
    plt.plot(map_sizes, convergence_speeds[algo], label=algo)
plt.xlabel('State Space Size')
plt.ylabel('Iterations/Episodes to Convergence')
plt.title('Convergence Speed vs State Space Size')
plt.legend()

# Plot for Policy Quality
plt.subplot(1, 2, 2)
for algo in policy_qualities:
    plt.plot(map_sizes, policy_qualities[algo], label=algo)
plt.xlabel('State Space Size')
plt.ylabel('Policy Quality Metric')
plt.title('Policy Quality vs State Space Size')
plt.legend()

plt.tight_layout()
plt.show()
