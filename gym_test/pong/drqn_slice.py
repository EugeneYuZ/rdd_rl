import torch
import torch.nn as nn
import numpy as np
import gym

import sys
sys.path.append('../..')
from agent.drqn_slice_agent import DRQNSliceAgent
from gym_test.wrapper import wrap_drqn, wrap_dqn
from util.utils import LinearSchedule
from util.plot import *


class DRQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DRQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)

        self.lstm = nn.LSTM(conv_out_size, 512, batch_first=True)
        self.fc = nn.Linear(512, n_actions)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))

        return int(np.prod(o.size()))

    def forward(self, x, hidden=None):
        x = x.float() / 256
        shape = x.shape
        x = x.view(shape[0]*shape[1], shape[2], shape[3], shape[4])
        conv_out = self.conv(x)
        x = conv_out.view(shape[0], shape[1], -1)
        if hidden is None:
            x, hidden = self.lstm(x)
        else:
            x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden


# class DRQNSliceStackAgent(DRQNSliceAgent):
#     def __init__(self, *args, **kwargs):
#         DRQNSliceAgent.__init__(self, *args, **kwargs)
#
#     def resetEnv(self):
#         obs = np.array(self.env.reset())
#         self.state = torch.tensor(obs, device=self.device, dtype=torch.float).unsqueeze(0)
#         return
#
#     def getNextState(self, obs):
#         return torch.tensor(np.array(obs), device=self.device, dtype=torch.float).unsqueeze(0)


if __name__ == '__main__':
    env = gym.make('PongNoFrameskip-v4')
    env = wrap_dqn(env)
    # env = wrap_drqn(env)

    agent = DRQNSliceAgent(DRQN, model=DRQN(env.observation_space.shape, env.action_space.n), env=env,
                           exploration=LinearSchedule(100000, 0.02),
                           batch_size=32, target_update_frequency=1000, memory_size=500, sequence_len=10)
    # agent = DRQNSliceStackAgent(DRQN, model=DRQN(env.observation_space.shape, env.action_space.n), env=env,
    #                        exploration=LinearSchedule(100000, 0.02),
    #                        batch_size=32, target_update_frequency=1000, memory_size=500, sequence_len=10)
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/gym_test/pong/data/drqn_slice'
    # agent.loadCheckpoint('20190210193936')
    agent.train(10000, 10000, print_step=False, save_freq=50)
    # plotLearningCurve(agent.episode_rewards, window=5)
    # plt.show()