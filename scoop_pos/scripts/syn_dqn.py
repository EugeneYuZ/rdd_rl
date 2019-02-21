import sys
import numpy as np
sys.path.append('../..')

from util.utils import LinearSchedule
from util.plot import *
from agent.syn_agent.syn_dqn_agent import SynDQNAgent as Agent
from agent.syn_agent.syn_dqn_agent import *
from env import ScoopEnv


class DQN(torch.nn.Module):
    def __init__(self, n_inputs, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    envs = []
    for i in range(4):
        env = ScoopEnv(19997 + i)
        envs.append(env)

    agent = Agent(DQN(envs[0].observation_space.shape[0], envs[0].nA), envs, LinearSchedule(10000, 0.1),
                  batch_size=128, min_mem=1000)
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/scoop_pos/data/syn_dqn_dense'
    agent.loadCheckpoint('20190221191448')
    agent.train(100000, 200, save_freq=500)
