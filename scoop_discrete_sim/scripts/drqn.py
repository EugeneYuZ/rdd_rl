import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../..')

from util.utils import LinearSchedule
from scoop_discrete.scripts.drqn_lrud import ConvDRQNAgent
from env import SimScoopEnv

class DRQN(torch.nn.Module):
    def __init__(self):
        super(DRQN, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, 5)
        self.conv2 = nn.Conv1d(8, 16, 3)
        self.lstm = nn.LSTMCell(64+4, 16)
        self.fc1 = nn.Linear(16, 4)

        self.hidden_size = 16

    def forward(self, inputs):
        (x, action), (hx, cx) = inputs
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64)
        x = torch.cat((x, action), 1)
        hx, cx = self.lstm(x, (hx, cx))
        x = F.relu(hx)
        x = self.fc1(x)
        return x, (hx, cx)


if __name__ == '__main__':
    agent = ConvDRQNAgent(DRQN, model=DRQN(), env=SimScoopEnv(),
                          exploration=LinearSchedule(100000, initial_p=1.0, final_p=0.1),
                          batch_size=2)
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/scoop_discrete_sim/data/drqn_lrud'

    agent.train(100000, max_episode_steps=200)
