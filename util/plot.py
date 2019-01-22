import numpy as np
import matplotlib.pyplot as plt


def plotLearningCurve(episode_rewards, window=100, label='reward', color='b', shadow=True):
    xs = []
    moving_avg = []
    moving_std = []
    for i in range(len(episode_rewards)):
        xs.append(i)
        moving_avg.append(np.average(episode_rewards[max(0, i-window/2):i+window/2]))
        moving_std.append(np.std(episode_rewards[max(0, i-window/2):i+window/2]))

    moving_avg = np.array(moving_avg)
    moving_std = np.array(moving_std)
    if shadow:
        plt.fill_between(xs, moving_avg-moving_std, moving_avg+moving_std, alpha=0.2, color=color)
    plt.plot(xs, moving_avg, label=label, color=color)
    plt.legend(loc=4)


# def plotLearningCurve(episode_rewards, window=100, label='reward', color='b'):
#     xs = []
#     moving_avg = []
#     moving_q1 = []
#     moving_q3 = []
#
#     for i in range(len(episode_rewards)):
#         xs.append(i)
#         moving_avg.append(np.average(episode_rewards[max(0, i-window/2):i+window/2]))
#         moving_q1.append(np.percentile(episode_rewards[max(0, i-window/2):i+window/2], 25))
#         moving_q3.append(np.percentile(episode_rewards[max(0, i-window/2):i+window/2], 75))
#
#     moving_avg = np.array(moving_avg)
#     moving_q1 = np.array(moving_q1)
#     moving_q3 = np.array(moving_q3)
#
#     plt.fill_between(xs, moving_q1, moving_q3, alpha=0.2, color=color)
#     plt.plot(xs, moving_avg, label=label, color=color)
#     plt.legend(loc=4)


if __name__ == '__main__':
    # a = np.load('reward.npy')
    # plotLearningCurve(a, label='lstm', color='b')
    b = np.load('/home/ur5/thesis/simple_task/scoop_1d/data/q_learning/reward_1000')
    plotLearningCurve(b, label='Q-Learning')
    plt.show()

    b = np.load('/home/ur5/thesis/simple_task/scoop_1d/data/q_learning/length_1000')
    plotLearningCurve(b, label='Q-Learning')
    plt.show()



