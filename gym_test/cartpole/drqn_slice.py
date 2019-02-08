import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../..')

from util.utils import LinearSchedule
from util.plot import *
# from scoop_discrete.scripts.drqn_lrud import ConvDRQNAgent
from agent.drqn_slice_agent import DRQNSliceAgent

import gym


class DRQN(torch.nn.Module):
    def __init__(self):
        super(DRQN, self).__init__()

        self.fc1 = nn.Linear(4, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x, hidden=None):
        x = F.relu(self.fc1(x))
        if hidden is None:
            x, hidden = self.lstm(x)
        else:
            x, hidden = self.lstm(x, hidden)
        x = self.fc2(x)
        return x, hidden


def train():
    env = gym.make("CartPole-v1")
    agent = DRQNSliceAgent(DRQN, model=DRQN(), env=env,
                           exploration=LinearSchedule(10000, initial_p=1.0, final_p=0.02),
                           batch_size=32, memory_size=10000, min_mem=100, sequence_len=64)
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/gym_test/cartpole/data/drqn_slice'
    agent.train(10000, 300, 100, False, print_step=False)


def plot(checkpoint):
    env = None
    agent = DRQNSliceAgent(DRQN)
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/gym_test/cartpole/data/drqn_slice'
    agent.loadCheckpoint(checkpoint, data_only=True)
    plotLearningCurve(agent.episode_rewards, window=100)
    plt.show()


if __name__ == '__main__':
    train()
    # plot('20190208212151')