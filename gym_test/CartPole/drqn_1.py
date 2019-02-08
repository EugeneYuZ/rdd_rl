import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('..')

from util.utils import LinearSchedule
from util.plot import *
# from scoop_discrete.scripts.drqn_lrud import ConvDRQNAgent
# from agent.drqn_agent import DRQNAgent
from drqn_agent import CartPoleDRQNAgent

import gym


class DRQN(torch.nn.Module):
    def __init__(self):
        super(DRQN, self).__init__()
        self.lstm = nn.LSTM(4, 128, batch_first=True)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x, hidden=None, episode_size=None):
        if episode_size is None:
            episode_size = [1]
        x = nn.utils.rnn.pack_padded_sequence(x, episode_size, True)
        if hidden is None:
            x, hidden = self.lstm(x)
        else:
            x, hidden = self.lstm(x, hidden)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, True)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, hidden


def train():
    env = gym.make("CartPole-v1")
    agent = CartPoleDRQNAgent(DRQN, model=DRQN(), env=env,
                          exploration=LinearSchedule(10000, initial_p=1.0, final_p=0.1),
                          batch_size=32, target_update_frequency=20)
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/gym_test/data/drqn_cartpole_1'
    agent.train(100000, 200, 100, False)


def plot(checkpoint):
    env = None
    agent = CartPoleDRQNAgent(DRQN)
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/gym_test/data/drqn_cartpole_1'
    agent.load_checkpoint(checkpoint, data_only=True)
    plotLearningCurve(agent.episode_rewards, window=100)
    plt.show()


if __name__ == '__main__':
    # train()
    plot('20190208141418')
