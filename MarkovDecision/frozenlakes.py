# Code for the frozen lakes example was used by Andrea Pierré under the MIT License
# This can be found here: https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/#map-size-4-times-4
from pathlib import Path
from typing import NamedTuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from enum import Enum, auto

sns.set_theme()
np.random.seed(26)
class LearnerType(Enum):
    VALUE_ITERATION = auto()
    POLICY_ITERATION = auto()
    Q_LEARNING = auto()

    def __str__(self):
        # Replace underscores with spaces and capitalize each word
        return ' '.join(word.capitalize() for word in self.name.split('_'))

class MyMDPConverter:
    def __init__(self, env):
        self.env = env
        self.states = env.observation_space.n
        self.actions = env.action_space.n
        self.transitions = env.P  # or however your environment represents transitions
        self.P = np.zeros((self.actions, self.states, self.states))
        self.R = np.zeros((self.states, self.actions))

    def convert_PR(self):
        """Converts the transition probabilities to MDPToolbox-compatible P and R arrays"""
        for state in range(self.states):
            for action in range(self.actions):
                for transition_prob, next_state, reward, _ in self.transitions[state][action]:
                    self.R[state][action] += transition_prob * reward
                    self.P[action, state, next_state] += transition_prob



class Params(NamedTuple):
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    seed: int  # Define a seed so that we get reproducible results
    is_slippery: bool  # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    proba_frozen: float  # Probability that a tile is frozen
    savefig_folder: Path  # Root folder where plots are saved

class Qlearning:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update

    def reset_qtable(self):
        """Reset the Q-table."""
        self.qtable = np.zeros((self.state_size, self.action_size))

class EpsilonGreedy:
    def __init__(self, epsilon, rng):
        self.epsilon = epsilon
        self.rng = rng

    def choose_action(self, action_space, state, qtable):
        """Choose an action `a` in the current world state (s)."""
        # First we randomize a number
        explor_exploit_tradeoff = self.rng.uniform(0, 1)

        # Exploration
        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample()

        # Exploitation (taking the biggest Q-value for this state)
        else:
            # Break ties randomly
            # If all actions are the same for this state we choose a random one
            # (otherwise `np.argmax()` would always take the first one)
            if np.all(qtable[state, :]) == qtable[state, 0]:
                action = action_space.sample()
            else:
                action = np.argmax(qtable[state, :])
        return action
        
def run_env(params, learner, env, explorer):
    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, params.state_size, params.action_size))
    all_states = []
    all_actions = []

    for run in range(params.n_runs):  # Run several times to account for stochasticity
        learner.reset_qtable()  # Reset the Q-table between runs

        for episode in tqdm(
            episodes, desc=f"Run {run}/{params.n_runs} - Episodes", leave=False
        ):
            state = env.reset(seed=42)[0]  # Reset the environment
            step = 0
            done = False
            total_rewards = 0

            while not done:
                action = explorer.choose_action(
                    action_space=env.action_space, state=state, qtable=learner.qtable
                )

                # Log all states and actions
                all_states.append(state)
                all_actions.append(action)

                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated

                learner.qtable[state, action] = learner.update(
                    state, action, reward, new_state
                )

                total_rewards += reward
                step += 1

                # Our new state is state
                state = new_state

            # Log all rewards and steps
            rewards[episode, run] = total_rewards
            steps[episode, run] = step
        qtables[run, :, :] = learner.qtable

    return rewards, steps, episodes, qtables, all_states, all_actions

def postprocess(episodes, params, rewards, steps, map_size):
    """Convert the results of the simulation in dataframes."""
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=params.n_runs),
            "Rewards": rewards.flatten(),
            "Steps": steps.flatten(),
        }
    )
    res["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    res["map_size"] = np.repeat(f"{map_size}x{map_size}", res.shape[0])

    st = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=1)})
    st["map_size"] = np.repeat(f"{map_size}x{map_size}", st.shape[0])
    return res, st

def qtable_directions_map(qtable, map_size):
    """Get the best learned action & map it to arrows."""
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            # Assign an arrow only if a minimal Q-value has been learned as best action
            # otherwise since 0 is a direction, it also gets mapped on the tiles where
            # it didn't actually learn anything
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions

def qtable_directions_allmap(qtable, map_size):
    """Get the best learned action & map it to arrows."""
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    
    for idx, val in enumerate(qtable_best_action.flatten()):
        # Assign an arrow based on the best action regardless of Q-value
        qtable_directions[idx] = directions[val]
    
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_directions


def plot_q_values_map(qtable, env, map_size, params):
    """Plot the last frame of the simulation and the policy learned."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    # Plot the policy
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    img_title = f"frozenlake_q_values_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()

def plot_states_actions_distribution(states, actions, map_size, params):
    """Plot the distributions of states and actions."""
    labels = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=states, ax=ax[0], kde=True)
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title("Actions")
    fig.tight_layout()
    img_title = f"frozenlake_states_actions_distrib_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()

def plot_steps_and_rewards(rewards_df, steps_df, params):
    """Plot the steps and rewards from dataframes."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.lineplot(
        data=rewards_df, x="Episodes", y="cum_rewards", hue="map_size", ax=ax[0]
    )
    ax[0].set(ylabel="Cumulated rewards")

    sns.lineplot(data=steps_df, x="Episodes", y="Steps", hue="map_size", ax=ax[1])
    ax[1].set(ylabel="Averaged steps number")

    for axi in ax:
        axi.legend(title="map size")
    fig.tight_layout()
    img_title = "frozenlake_steps_and_rewards.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()

# #TODO: rewrite
# def derive_state_matrices(env):
#     state_total = env.observation_space.n
#     action_total = env.action_space.n
#     matrix_T = np.zeros((action_total, state_total, state_total))
#     matrix_R = np.zeros_like(matrix_T)

#     for act in range(action_total):
#         for st in range(state_total):
#             for outcome in env.P[st][act]:
#                 trans_prob, following_state, reward_val, _ = outcome
#                 matrix_T[act, st, following_state] += trans_prob
#                 matrix_R[act, st, following_state] += reward_val

#     # Normalize the transition matrix
#     matrix_T = adjust_transitions(matrix_T)
#     return matrix_T, matrix_R

def adjust_transitions(matrix_T):
    for a_matrix in matrix_T:
        sums = a_matrix.sum(axis=1, keepdims=True)  # Sum over axis 1, not 2
        sums[sums == 0] = 1
        a_matrix /= sums
    return matrix_T


# def derive_state_matrices(env):
#     n_states = env.observation_space.n
#     n_actions = env.action_space.n
#     P = np.zeros((n_states, n_actions, n_states))
#     R = np.zeros((n_states, n_actions))

#     for s in range(n_states):
#         for a in range(n_actions):
#             transitions = env.P[s][a]
#             for prob, next_state, reward, done in transitions:
#                 P[s][a][next_state] += prob
#                 # As we want a single reward for (s,a), we either take the max or mean
#                 R[s][a] = max(R[s][a], reward)  # or use np.mean([R[s][a], reward])

#     return P, R


# import numpy as np

def derive_state_matrices(env):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    P = np.zeros((n_actions, n_states, n_states))
    R = np.zeros((n_states, n_actions))

    for s in range(n_states):
        for a in range(n_actions):
            transitions = env.P[s][a]
            for prob, next_state, reward, _ in transitions:
                P[a][s][next_state] += prob
                # We take the reward for the transition with the highest probability
                R[s][a] = max(R[s][a], reward)
                
    # Make sure that probabilities sum to 1 for each s, a pair
    for s in range(n_states):
        for a in range(n_actions):
            total = np.sum(P[a][s, :])
            if total > 0:  # Avoid division by zero
                P[a][s, :] /= total

    return P, R

