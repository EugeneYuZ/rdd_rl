import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

sys.path.append('../..')

from util.utils import LinearSchedule
from util.plot import *
from agent.drqn_slice_agent import DRQNSliceAgent

import gym


class DRQN(torch.nn.Module):
    def __init__(self):
        super(DRQN, self).__init__()

        self.fc1 = nn.Linear(2, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x, hidden=None):
        x = F.relu(self.fc1(x))
        if hidden is None:
            x, hidden = self.lstm(x)
        else:
            x, hidden = self.lstm(x, hidden)
        x = self.fc2(x)
        return x, hidden


class CartPoleDRQNSliceAgent(DRQNSliceAgent):
    def __init__(self, *args, **kwargs):
        DRQNSliceAgent.__init__(self, *args, **kwargs)
        self.state_padding = torch.zeros(2).unsqueeze(0)

    def resetEnv(self):
        """
        reset the env and set self.state
        :return: None
        """
        obs = self.env.reset()
        obs = obs[[0, 2]]
        self.state = torch.tensor(obs, device=self.device, dtype=torch.float).unsqueeze(0)
        return

    def takeAction(self, action):
        """
        take given action and return response
        :param action: int, action to take
        :return: obs_, r, done, info
        """
        obs, r, done, info = self.env.step(action)
        obs = obs[[0, 2]]
        if done:
            r = -1
        return obs, r, done, info

    def train(self, num_episodes, max_episode_steps=100, save_freq=100, render=False):
        while self.episodes_done < num_episodes:
            self.trainOneEpisode(num_episodes, max_episode_steps, save_freq, render)
            if len(self.episode_rewards) > 100:
                avg = np.average(self.episode_rewards[-100:])
                tqdm.write('avg reward in 100 episodes: {}'.format(avg))
                if avg > 195:
                    print 'solved'
                    return


def train():
    env = gym.make("CartPole-v1")
    agent = CartPoleDRQNSliceAgent(DRQN, model=DRQN(), env=env,
                                   exploration=LinearSchedule(100000, initial_p=1.0, final_p=0.02),
                                   batch_size=32, memory_size=100000, min_mem=10000, sequence_len=32)
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/gym_test/cartpole/data/partial_drqn_slice'
    agent.loadCheckpoint('20190212204507')
    agent.train(100000, 500, 200, False)


def plot(checkpoint):
    env = None
    agent = DRQNSliceAgent(DRQN)
    agent.saving_dir = '/home/ur5/thesis/rdd_rl/gym_test/cartpole/data/partial_drqn_slice'
    agent.loadCheckpoint(checkpoint, data_only=True)
    plotLearningCurve(agent.episode_rewards, window=100)
    plt.show()


if __name__ == '__main__':
    train()
    # plot('20190210114727')