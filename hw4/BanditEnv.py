import numpy as np

class BanditEnv:
    def __init__(self, k, stationary=True):
        self.k = k
        self.stationary = stationary
        self.reset()

    def reset(self):
        # true mean of the k arms, from N(0, 1)
        self.true_mean = np.random.normal(0, 1, self.k)
        self.action_history = []
        self.reward_history = []
        # there might be multiple optimal actions
        self.optimal_action = np.argmax(self.true_mean)


    def step(self, action):
        # reward = N(true_mean[ith_ bandit], 1)
        reward = np.random.normal(self.true_mean[action], 1)
        self.action_history.append(action)
        self.reward_history.append(reward)

        # if not self.stationary:
        if not self.stationary:
            # random walk
            self.true_mean += np.random.normal(0, 0.01, self.k)
            # update optimal action
            self.optimal_action = np.argmax(self.true_mean)
        return reward
    

    def check_optimal_action(self, action):
        if self.true_mean[action] == self.true_mean[self.optimal_action]:
            return True
        return False

    def export_history(self):
        return self.action_history, self.reward_history

