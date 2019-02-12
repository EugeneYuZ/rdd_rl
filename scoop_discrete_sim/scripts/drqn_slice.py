import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../..')

from util.utils import LinearSchedule
from scoop_discrete.scripts.drqn_slice_lrud import ConvDRQNAgent
from env import SimScoopEnv


class DRQN(torch.nn.Module):
    def __init__(self):
        super(DRQN, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, 3)
        self.conv2 = nn.Conv1d(8, 16, 3)
        self.lstm = nn.LSTM(96+4, 64)
        self.fc1 = nn.Linear(64, 4)

        self.hidden_size = 64

    def forward(self, inputs, hidden=None):
        x, action = inputs
        shape = x.shape
        x = x.view(shape[0]*shape[1], shape[2], shape[3])
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(shape[0], shape[1], -1)
        x = torch.cat((x, action), 2)
        if hidden is None:
            x, hidden = self.lstm(x)
        else:
            x, hidden = self.lstm(x, hidden)
        x = self.fc1(x)
        return x, hidden


if __name__ == '__main__':
    agent = ConvDRQNAgent(DRQN, model=DRQN(), env=SimScoopEnv(),
                          exploration=LinearSchedule(10000, initial_p=1.0, final_p=0.1), min_mem=1000)
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/scoop_discrete_sim/data/drqn_slice'
    agent.train(100000, max_episode_steps=200)
