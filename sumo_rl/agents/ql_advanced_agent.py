"""Q-learning Advanced Agent"""


from enum import Enum

# Agent.learn can use classic Q-Learning or SARSA
class ValueUpdateStrategy(Enum):
    QLearning = 1
    SARSA = 2


from sumo_rl.exploration import EpsilonGreedyWithState  # frozen choices if SARSA is used

class QLAdvancedAgent:
    def __init__(
            self,
            starting_state,
            state_space,
            action_space,
            alpha=0.5,
            gamma=0.95,
            exploration_strategy=EpsilonGreedyWithState(),
            value_update_strategy=ValueUpdateStrategy.QLearning
    ):
        """Initialize Advanced Q-learning agent."""
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {self.state: [0 for _ in range(action_space.n)]}
        self.exploration = exploration_strategy
        self.value_update_strategy = value_update_strategy
        self.acc_reward = 0

    def act(self):
        """Choose action based on Q-table."""
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space, do_update=True)
        return self.action

    def learn(self, next_state, reward, done=False):
        """Update Q-table with new experience."""
        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]

        s = self.state
        s1 = next_state
        a = self.action
        if done:
            td_error = reward - self.q_table[s][a]
        else:
            if self.value_update_strategy == ValueUpdateStrategy.QLearning:
                next_value = self.gamma * max(self.q_table[s1])
                td_error = reward + next_value - self.q_table[s][a]

            elif self.value_update_strategy == ValueUpdateStrategy.SARSA:
                a1 = self.exploration.choose(self.q_table, next_state, self.action_space, do_update=False)
                next_value = self.gamma * self.q_table[s1][a1]
                td_error = reward + next_value - self.q_table[s][a]
            else:
                raise ValueError("Wrong argument value_update_strategy.")

        self.q_table[s][a] += self.alpha * td_error
        self.state = s1
        self.acc_reward += reward
