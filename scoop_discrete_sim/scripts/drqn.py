import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../..')

from util.utils import LinearSchedule
# from scoop_discrete.scripts.drqn_lrud import ConvDRQNAgent
from agent.drqn_agent import DRQNAgent
from env import SimScoopEnv


# class DRQN(torch.nn.Module):
#     def __init__(self):
#         super(DRQN, self).__init__()
#         self.conv1 = nn.Conv1d(1, 8, 5)
#         self.conv2 = nn.Conv1d(8, 16, 3)
#         self.lstm = nn.LSTMCell(64+4, 16)
#         self.fc1 = nn.Linear(16, 4)
#
#         self.hidden_size = 16
#
#     def forward(self, inputs):
#         (x, action), (hx, cx) = inputs
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = x.view(-1, 64)
#         x = torch.cat((x, action), 1)
#         hx, cx = self.lstm(x, (hx, cx))
#         x = F.relu(hx)
#         x = self.fc1(x)
#         return x, (hx, cx)

class DRQN(torch.nn.Module):
    def __init__(self):
        super(DRQN, self).__init__()
        self.fc1 = nn.Linear(10, 16)
        self.lstm = nn.LSTM(16, 32, batch_first=True)
        self.fc2 = nn.Linear(32, 4)
        self.hidden_size = 32

    def forward(self, x, hidden):
        hx, cx = hidden
        x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        # x = nn.utils.rnn.pack_padded_sequence(x, [1], True)
        x, (hx, cx) = self.lstm(x, (hx, cx))
        x = self.fc2(x)
        x = x.squeeze(0)
        return x, (hx, cx)

    def forwardSequence(self, x, hidden, episode_size):
        hx, cx = hidden
        x = F.relu(self.fc1(x))
        x = nn.utils.rnn.pack_padded_sequence(x, episode_size, True)
        x, (hx, cx) = self.lstm(x, (hx, cx))
        x, _ = nn.utils.rnn.pad_packed_sequence(x, True)
        x = self.fc2(x)
        return x, (hx, cx)


if __name__ == '__main__':
    agent = DRQNAgent(DRQN, model=DRQN(), env=SimScoopEnv(),
                          exploration=LinearSchedule(100000, initial_p=1.0, final_p=0.1),
                          batch_size=2)
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/scoop_discrete_sim/data/drqn_lrud'
    # agent.load_checkpoint('20190206133550')
    agent.train(100000, max_episode_steps=200)
