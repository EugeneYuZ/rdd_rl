import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../..')

from util.utils import LinearSchedule
from util.plot import *
from agent.drqn_agent import DRQNAgent

import gym


class DRQN(torch.nn.Module):
    def __init__(self):
        super(DRQN, self).__init__()

        self.fc1 = nn.Linear(2, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x, hidden=None, episode_size=None):
        if episode_size is None:
            episode_size = [1]
        x = F.relu(self.fc1(x))
        x = nn.utils.rnn.pack_padded_sequence(x, episode_size, True)
        if hidden is None:
            x, hidden = self.lstm(x)
        else:
            x, hidden = self.lstm(x, hidden)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, True)
        x = self.fc2(x)
        return x, hidden


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    agent = DRQNAgent(DRQN, model=DRQN(), env=env,
                      exploration=LinearSchedule(100000, initial_p=1.0, final_p=0.02),
                      batch_size=8, memory_size=1000, min_mem=100)
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/gym_test/mountain_car/data/drqn'
    agent.train(10000, 2000, 100, False, False)
