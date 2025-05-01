from BanditEnv import BanditEnv
from Agent import Agent


avg_reward = []
optimal_select = []

# test for epsilon 0, 0.1, 0.01 (k=10)
for times in range(2000):
    env = BanditEnv(10)
    env.reset()
    actions = []
    rewards = []
    avg_reward_per_round = 0
    optimal_select_cnt = 0

    k = 10
    epsilon = 0.01
    agent = Agent(k, epsilon)
    reward = 0

    for i in range(1000):
        action = agent.select_action()

        # Optimal action
        if action == env.optimal_action:
            optimal_select_cnt += 1

        reward = env.step(action)
        agent.update_q(action, reward)
        actions.append(action)
        rewards.append(reward)

    action_history, reward_history = env.export_history()

    for i in range(1000):
        assert actions[i] == action_history[i]
        assert rewards[i] == reward_history[i]

    # average reward per round
    avg_reward_per_round = sum(rewards) / 1000
    avg_reward.append(avg_reward_per_round)

    # optimal select rate
    optimal_select_rate = optimal_select_cnt / 1000
    optimal_select.append(optimal_select_rate)

    # print("All tests passed!\n\n")

    # print("epsilon: ", epsilon)
    # # pritn total reward
    # print("Total reward: ", sum(rewards))
    # # print average reward
    # print("Average reward: ", sum(rewards) / len(rewards))


# plot the average reward and optimal select rate
import matplotlib.pyplot as plt


# Plot the average reward over time
plt.figure(figsize=(12, 6))

# Plot average reward
plt.subplot(1, 2, 1)
print(avg_reward)
print(optimal_select)
plt.plot(avg_reward, label="Average Reward")
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.title("Average Reward Over Time")
plt.legend()

# Plot optimal action selection percentage
plt.subplot(1, 2, 2)
plt.plot(optimal_select, label="Optimal Action Selection %", color="orange")
plt.xlabel("Episodes")
plt.ylabel("Optimal Action Selection Percentage")
plt.title("Optimal Action Selection Over Time")
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()









