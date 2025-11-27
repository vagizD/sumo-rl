"""Epsilon Greedy Exploration Strategy."""

import numpy as np
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy


class EpsilonGreedyWithState(EpsilonGreedy):
    """Epsilon Greedy Exploration Strategy with availability to make frozen choices."""

    def choose(self, q_table, state, action_space, do_update=True):
        """Choose action based on epsilon greedy strategy."""
        if np.random.rand() < self.epsilon:
            action = int(action_space.sample())
        else:
            action = np.argmax(q_table[state])

        if do_update:
            self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
        # print(self.epsilon)
        return action
