import numpy as np
import BanditEnv 
"""
Please implement a sample-average method to estimate the expected reward of each action.
The agent should have the following methods:
• select_action(self): Choose an action based on the estimated expected reward for
each action.
• update_q(self, action, reward): Update the estimated expected reward of the
chosen action.
• reset(self): Reset the agent.

"""

class Agent:
    def __init__(self, k, epsilon, step_size = None):
        self.k = k
        self.epsilon = epsilon
        self.step_size = step_size
        self.reset()

    def select_action(self):
        if np.random.rand() < self.epsilon:
            # randomly select with p < epsilon
            return np.random.randint(0, self.k)
        else:
            # else, select action with max expected reward
            return np.argmax(self.q_values)
        
    def update_q(self, action, reward):
        # update the number of times action was selected
        self.action_cnt[action] += 1

        if self.step_size is None:
            self.reward_sum[action] += reward
            self.q_values[action] = self.reward_sum[action] / self.action_cnt[action]
        else: 
             # update the estimated expected reward of the chosen action
            self.q_values[action] += self.step_size * (reward - self.q_values[action])
            # update the total reward received for the chosen action
            self.reward_sum[action] += reward
            
        

    def reset(self):
        # store estimated expected reward of each action
        self.q_values = np.zeros(self.k)
        # store number of times each action was selected
        self.action_cnt = np.zeros(self.k)
        # store total reward received for each action
        self.reward_sum = np.zeros(self.k)
