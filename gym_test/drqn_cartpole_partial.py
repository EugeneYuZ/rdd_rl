import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('..')

from util.utils import LinearSchedule
from util.plot import *
# from scoop_discrete.scripts.drqn_lrud import ConvDRQNAgent
# from agent.drqn_agent import DRQNAgent
from cartpole_partial_drqn_agent import CartPoleDRQNAgent

import gym


class DRQN(torch.nn.Module):
    def __init__(self):
        super(DRQN, self).__init__()

        self.fc1 = nn.Linear(2, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc2 = nn.Linear(128, 2)

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


def train():
    env = gym.make("CartPole-v1")
    agent = CartPoleDRQNAgent(DRQN, model=DRQN(), env=env,
                          exploration=LinearSchedule(100000, initial_p=1.0, final_p=0.1),
                          batch_size=32, target_update_frequency=20, memory_size=1000)
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/gym_test/data/drqn_cartpole_partial'
    agent.train(100000, 200, 100, False)


def plot(checkpoint):
    env = None
    agent = CartPoleDRQNAgent(DRQN, model=DRQN(), env=env,
                              exploration=LinearSchedule(100000, initial_p=1.0, final_p=0.1),
                              batch_size=32, target_update_frequency=20)
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/gym_test/data/drqn_cartpole_partial'
    agent.load_checkpoint(checkpoint)
    plotLearningCurve(agent.episode_rewards, window=20)
    plt.show()


if __name__ == '__main__':
    train()
