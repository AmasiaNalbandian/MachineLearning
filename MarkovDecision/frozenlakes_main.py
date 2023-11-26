import numpy as np
from hiive.mdptoolbox import mdp

def generate_frozen_lake_environment(size, slip_chance=0.1):
    """
    Generate a Frozen Lake-like environment.
    size: Size of the grid (size x size)
    slip_chance: Probability of slipping to the wrong state
    """
    n_states = size * size
    n_actions = 4  # up, down, left, right

    # Initialize transition and reward matrices
    P = np.zeros((n_actions, n_states, n_states))
    R = np.full((n_states, n_actions), -1.0)  # Default reward is -1

    for state in range(n_states):
        row, col = divmod(state, size)

        transitions = {
            'up':    max(row - 1, 0) * size + col,
            'down':  min(row + 1, size - 1) * size + col,
            'left':  row * size + max(col - 1, 0),
            'right': row * size + min(col + 1, size - 1)
        }

        for action, next_state in enumerate(['left', 'up', 'right', 'down']):
            if state == n_states - 1:  # Goal state
                P[action, state, state] = 1.0
                R[state, action] = 0.0
            else:
                target = transitions[next_state]
                P[action, state, target] += 1.0 - slip_chance

                # Distribute slip chance
                for slip_state in transitions.values():
                    if slip_state != target:
                        P[action, state, slip_state] += slip_chance / 3

    return P, R

def run_policy_iteration(P, R):
    print("P shape:", np.array(P).shape)
    print("R shape:", np.array(R).shape)
    print("P type:", type(P))
    print("R type:", type(R))

    pi = mdp.PolicyIteration(P, R, discount=0.9)
    pi.run()
    return pi

def run_value_iteration(P, R):
    vi = mdp.ValueIteration(P, R, discount=0.9)
    vi.run()
    return vi

def run_q_learning(P, R):
    ql = mdp.QLearning(P, R, discount=0.9, n_iter=10000)
    ql.run()
    return ql

def main():
    sizes = [4, 6, 8]  # Different sizes of the grid

    for size in sizes:
        print(f"Running MDPs for Frozen Lake environment of size {size}x{size}")
        P, R = generate_frozen_lake_environment(size)


        print("Policy Iteration:")
        pi_result = run_policy_iteration(P, R)
        print(pi_result.policy)

        print("Value Iteration:")
        vi_result = run_value_iteration(P, R)
        print(vi_result.policy)

        print("Q-Learning:")
        ql_result = run_q_learning(P, R)
        print(ql_result.policy)

if __name__ == "__main__":
    main()
