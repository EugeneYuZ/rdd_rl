import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('..')

from util.utils import LinearSchedule
from util.plot import *
# from scoop_discrete.scripts.drqn_lrud import ConvDRQNAgent
from drqn_os_agent import CartPoleDRQNAgent

import gym

class DRQN(torch.nn.Module):
    def __init__(self):
        super(DRQN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fc1 = nn.Linear(4, 64)
        self.lstm = nn.LSTMCell(64, 128)
        self.fc2 = nn.Linear(128, 2)

        self.hidden_size = 128

    def forward(self, x, hidden=None):
        x = F.relu(self.fc1(x))
        if hidden is None:
            hx, cx = self.lstm(x)
        else:
            hx, cx = self.lstm(x, hidden)
        x = self.fc2(hx)
        return x, (hx, cx)


if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    agent = CartPoleDRQNAgent(DRQN, model=DRQN(), env=env,
                             exploration=LinearSchedule(10000, initial_p=1.0, final_p=0.1),
                             batch_size=10, target_update_frequency=20)
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/gym_test/data/drqn_os_cartpole'
    agent.train(100000, 200, 100, False)