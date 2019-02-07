import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../..')

from util.utils import LinearSchedule
from util.plot import *
# from scoop_discrete.scripts.drqn_lrud import ConvDRQNAgent
from agent.drqn_agent import DRQNAgent

import gym


class DRQN(torch.nn.Module):
    def __init__(self):
        super(DRQN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fc1 = nn.Linear(4, 16)
        self.lstm = nn.LSTM(16, 32, batch_first=True)
        self.fc2 = nn.Linear(32, 2)
        self.hidden_size = 32

    def forward(self, x, hidden=None, episode_size=None):
        if episode_size is None:
            episode_size = [1]
        if hidden is None:
            hidden = (torch.zeros(1, len(episode_size), self.hidden_size, device=self.device, requires_grad=False),
                      torch.zeros(1, len(episode_size), self.hidden_size, device=self.device, requires_grad=False))
        hx, cx = hidden
        x = F.relu(self.fc1(x))
        x = nn.utils.rnn.pack_padded_sequence(x, episode_size, True)
        x, (hx, cx) = self.lstm(x, (hx, cx))
        x, _ = nn.utils.rnn.pad_packed_sequence(x, True)
        x = self.fc2(x)
        return x, (hx, cx)

    # def forwardSequence(self, x, hidden, episode_size):
    #     hx, cx = hidden
    #     x = F.relu(self.fc1(x))
    #     x = nn.utils.rnn.pack_padded_sequence(x, episode_size, True)
    #     x, (hx, cx) = self.lstm(x, (hx, cx))
    #     x, _ = nn.utils.rnn.pad_packed_sequence(x, True)
    #     x = self.fc2(x)
    #     return x, (hx, cx)


if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    # env = None
    agent = DRQNAgent(DRQN, model=DRQN(), env=env,
                      exploration=LinearSchedule(10000, initial_p=1.0, final_p=0.1),
                      batch_size=4, target_update_frequency=20)
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/gym_test/data/drqn_cartpole'

    # agent.load_checkpoint('20190207140411')
    # plotLearningCurve(agent.episode_rewards)
    # plt.show()
    # plotLearningCurve(agent.episode_lengths, label='length', color='r')
    # plt.show()

    agent.train(100000, 200, 100, True)