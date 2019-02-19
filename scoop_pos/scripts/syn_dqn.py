import sys
import numpy as np
sys.path.append('../..')

from util.utils import LinearSchedule
from util.plot import *
from agent.syn_agent.syn_dqn_agent import SynDQNAgent as Agent
import torch.nn as nn
from env import ScoopEnv


class DQN(torch.nn.Module):
    def __init__(self, n_inputs, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = nn.ReLU(self.fc1(x))
        x = nn.ReLU(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    envs = []
    for i in range(8):
        env = ScoopEnv(19997 + i)
        envs.append(env)

