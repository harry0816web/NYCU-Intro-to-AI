from BanditEnv import BanditEnv
from Agent import Agent
import numpy as np
import matplotlib.pyplot as plt

k = 10
epsilon_list = [0, 0.1, 0.01]  # List of epsilon values to test
exp_times = 2000
steps_per_exp = 10000

plt.figure(figsize=(12, 5))

for epsilon in epsilon_list:
    # sum up for exp_times and take average over time
    avg_reward = np.zeros(steps_per_exp)
    optimal_select = np.zeros(steps_per_exp)

    # test for epsilon 0, 0.1, 0.01 (k=10)
    for time in range(exp_times):
        env = BanditEnv(10, stationary=False)
        agent = Agent(k, epsilon)
        actions = []
        rewards = []

        for i in range(steps_per_exp):
            action = agent.select_action()
             # Optimal action selected
            if env.check_optimal_action(action):
                optimal_select[i] += 1     

            # average reward for each step over time
            reward = env.step(action)
            avg_reward[i] += reward

            agent.update_q(action, reward)
            actions.append(action)
            rewards.append(reward)

        action_history, reward_history = env.export_history()

        for i in range(steps_per_exp):
            assert actions[i] == action_history[i]
            assert rewards[i] == reward_history[i]

    avg_rewards_over_time = [reward / exp_times for reward in avg_reward]
    optimal_actions_over_time = [select_times / exp_times for select_times in optimal_select]

    # plot avg rewards for each epsilon
    plt.subplot(1, 2, 1)
    plt.plot(avg_rewards_over_time, label=f"ε = {epsilon}")

    # plot optimal action selection for each epsilon
    plt.subplot(1, 2, 2)
    plt.plot(optimal_actions_over_time, label=f"ε = {epsilon}")

plt.subplot(1, 2, 1)
plt.xlabel("Step")
plt.ylabel("Avg Reward (total reward / exp_times)")
plt.title("Average Reward Over Time")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.xlabel("Step")
plt.ylabel("Optimal Action (selected times/ exp_times)")
plt.title("Optimal Action Selection Over Time")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()