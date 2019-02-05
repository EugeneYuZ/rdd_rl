import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../..')

from util.utils import LinearSchedule
from agent.drqn_agent_one import OneStepDRQNAgent

from gym.envs.toy_text.frozen_lake import FrozenLakeEnv


class DRQN(torch.nn.Module):
    def __init__(self):
        super(DRQN, self).__init__()
        self.fc1 = nn.Linear(16, 32)
        self.lstm = nn.LSTMCell(32, 64)
        self.fc2 = nn.Linear(64, 4)

        self.hidden_size = 64

    def forward(self, inputs):
        x, (hx, cx) = inputs
        x = F.relu(self.fc1(x))
        hx, cx = self.lstm(x, (hx, cx))
        x = F.relu(hx)
        x = self.fc2(x)
        return x, (hx, cx)


