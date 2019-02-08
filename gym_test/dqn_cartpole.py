import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('..')
from agent.dqn_agent import DQNAgent

from util.utils import LinearSchedule
from util.plot import *
import gym


class CartPoleDRQNAgent(DQNAgent):
    def __init__(self, *args, **kwargs):
        DQNAgent.__init__(self, *args, **kwargs)

    def takeAction(self, action):
        """
        take given action and return response
        :param action: int, action to take
        :return: obs_, r, done, info
        """
        obs, r, done, info = self.env.step(action)
        if done:
            r = -1
        return obs, r, done, info


class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train():
    env = gym.make("CartPole-v1")
    agent = CartPoleDRQNAgent(DQN, model=DQN(), env=env,
                          exploration=LinearSchedule(10000, initial_p=1.0, final_p=0.1),
                          batch_size=128, target_update_frequency=20)
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/gym_test/data/dqn_cartpole'
    agent.train(10000, 200, 100, False)


def plot(checkpoint):
    agent = CartPoleDRQNAgent(DQN)
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/gym_test/data/dqn_cartpole'
    agent.load_checkpoint(checkpoint)
    plotLearningCurve(agent.episode_rewards, window=100)
    plt.show()


if __name__ == '__main__':
    train()