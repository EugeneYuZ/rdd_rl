import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../..')

from util.utils import LinearSchedule
from util.plot import *
from agent.drqn_slice_agent import DRQNSliceAgent

import gym


class DRQN(torch.nn.Module):
    def __init__(self):
        super(DRQN, self).__init__()

        self.fc1 = nn.Linear(2, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x, hidden=None):
        x = F.relu(self.fc1(x))
        if hidden is None:
            x, hidden = self.lstm(x)
        else:
            x, hidden = self.lstm(x, hidden)
        x = self.fc2(x)
        return x, hidden


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    agent = DRQNSliceAgent(DRQN, model=DRQN(), env=env,
                           exploration=LinearSchedule(100000, initial_p=1.0, final_p=0.02),
                           batch_size=32, sequence_len=10)
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/gym_test/mountain_car/data/drqn_slice'
    agent.train(10000, 2000, 100, False)
    # agent.loadCheckpoint('20190211161243', data_only=True)
    # plotLearningCurve(agent.episode_rewards)