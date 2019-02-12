import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gym

import sys
sys.path.append('../..')
from agent.drqn_slice_agent import DQNAgent
from gym_test.wrapper import wrap_dqn
from util.utils import LinearSchedule
from util.plot import *


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)

        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))

        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.float() / 256
        conv_out = self.conv(x)
        x = conv_out.view(x.size()[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class DQNStackAgent(DQNAgent):
    def __init__(self, *args, **kwargs):
        DQNAgent.__init__(self, *args, **kwargs)

    def resetEnv(self):
        obs = np.array(self.env.reset())
        self.state = torch.tensor(obs, device=self.device, dtype=torch.float).unsqueeze(0)
        return

    def getNextState(self, obs):
        return torch.tensor(np.array(obs), device=self.device, dtype=torch.float).unsqueeze(0)


if __name__ == '__main__':
    env = gym.make('PongNoFrameskip-v4')
    env = wrap_dqn(env)

    agent = DQNStackAgent(DQN, model=DQN(env.observation_space.shape, env.action_space.n), env=env,
                          exploration=LinearSchedule(100000, 0.02),
                          batch_size=32, target_update_frequency=1000, memory_size=100000)
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/gym_test/pong/data/dqn'
    agent.train(10000, 10000, save_freq=50)
